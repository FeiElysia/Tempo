import io
import os
import time
import torch
import argparse
import numpy as np
import multiprocessing
from decord import cpu, VideoReader

from tempo.builder import load_pretrained_model
from tempo.conversation import conv_templates, SeparatorStyle
from tempo.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from tempo.mm_datautils import (
    compute_segment_timestamp,
    KeywordsStoppingCriteria,
    process_qwen_content,
    tokenizer_image_token,
)

def get_real_cpu_cores():
    """use multiple threads for video decoding"""
    try:
        # HF Spaces
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        # Local environments
        cores = multiprocessing.cpu_count()
    return cores

def compute_sample_indices(
    total_frames: int,
    original_fps: float,
    video_fps: float = 2.0,
    min_frames_num: int = 4,
    max_frames_num: int = 1024,
) -> list[int]:
 
    start_frame, end_frame = 0, total_frames - 1
    clip_frames = end_frame - start_frame + 1
    if clip_frames <= 1:
        return [start_frame]
    
    if original_fps is None or original_fps <= 0:
        original_fps = video_fps
        
    clip_duration = clip_frames / original_fps
    target_num_frames = max(1, round(clip_duration * video_fps))
    final_num_frames = min(max(target_num_frames, min_frames_num), max_frames_num)
    
    if final_num_frames == 1:
        return [end_frame]
        
    indices = np.round(np.linspace(start_frame, end_frame, final_num_frames)).astype(int)
    indices = np.clip(indices, start_frame, end_frame)

    return indices.tolist()

def load_video(video_path: str, video_fps: float = 2.0, max_frames: int = 1024) -> tuple:

    available_cores = get_real_cpu_cores()
    optimal_threads = min(max(1, available_cores - 1), 16)
    print(f"[Profiling] Detected {available_cores} CPU cores. Decord using {optimal_threads} threads.")

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=optimal_threads)
    total_frames = len(vr)
    original_fps = vr.get_avg_fps()
    frame_idx = compute_sample_indices(total_frames, original_fps, video_fps, max_frames_num=max_frames)
    images = vr.get_batch(frame_idx).asnumpy()
    clip_duration = total_frames / original_fps

    real_fps = len(images) / clip_duration if clip_duration > 0 else video_fps
    print(f"[Info] Video loaded: {video_path} | Sampled Frames: {len(images)} | Real FPS: {real_fps:.2f}")

    return images, real_fps

def main(args):

    # load model, processor, tokenizer
    print(f"[Info] Loading Tempo model from {args.model_path}...")
    tokenizer, model, image_processor = load_pretrained_model(
        args.model_path,
        device_map="cuda",
        use_flash_attn=True,
    )

    # compression settings
    model.config.visual_token_budget = args.visual_token_budget
    model.config.tokenizer_model_max_length = args.tokenizer_model_max_length
    print(f"[Info] Visual Token Budget: {model.config.visual_token_budget}")
    print(f"[Info] Tokenizer Max Length: {model.config.tokenizer_model_max_length}")

    model.get_vision_tower_aux_list()[0].dynamic_compress = not args.disable_dynamic_compress
    print(f"[Info] Dynamic Compress: {model.get_vision_tower_aux_list()[0].dynamic_compress}")

    model.eval()
    model.to(torch.bfloat16)
    print("[Info] Processor loaded successfully!")
    print("[Info] Model loaded successfully!")

    # video process
    start_prep_time = time.perf_counter()
    try:
        video_frames, real_fps = load_video(args.video_path, args.video_fps, args.max_frames)
    except Exception as e:
        return f"⚠️ Error loading video: {str(e)}"

    vlm_inputs, seg_timestamps, image_sizes, processed_video = None, None, None, None
    
    # process local compressor inputs
    vlm_inputs = process_qwen_content(
        video_frames, "video", args.query, image_processor[0], real_fps, args.frame_windows, args.frame_stride, is_eval=True
    )
    vlm_inputs = {key: v.cuda() for key, v in vlm_inputs.items()}

    # compute timestamp for each segment
    seg_timestamps = compute_segment_timestamp(
        len(vlm_inputs["video_grid_thw"]), tokenizer, real_fps, args.frame_stride, args.frame_windows
    )

    # stat info
    num_segments = len(vlm_inputs["video_grid_thw"])
    segment_duration = args.frame_windows / real_fps
    stats_info = f"🎬 Video Stats: Total Segments: {num_segments}  |  Segment Duration: {segment_duration:.2f}s  |  Real FPS: {real_fps:.2f}"    
    print(f"[Info] {stats_info}")

    # prompt
    if getattr(model.config, "mm_use_im_start_end", False):
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + args.query
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + args.query

    conv = conv_templates[args.conv_version].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    # tokenization
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    start_infer_time = time.perf_counter()

    # generating
    print("[Info] Generating response...")
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=processed_video,
            image_sizes=image_sizes,
            do_sample=(args.temperature > 0),
            temperature=args.temperature if args.temperature > 0 else None,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            vlm_inputs=vlm_inputs,
            seg_timestamps=seg_timestamps,
        )

    end_infer_time = time.perf_counter()

    if isinstance(output_ids, tuple):
        output_ids = output_ids[0]
    
    prep_duration = start_infer_time - start_prep_time
    infer_duration = end_infer_time - start_infer_time
    total_duration = end_infer_time - start_prep_time
    
    print("\n" + "="*40)
    print(f"🚀 [Profiling] Video Loading & Prep : {prep_duration:.2f} s")
    print(f"🧠 [Profiling] Model Inference     : {infer_duration:.2f} s")
    print(f"⏱️  [Profiling] Total Request Time  : {total_duration:.2f} s")
    print("="*40 + "\n")

    pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if pred.endswith(stop_str):
        pred = pred[: -len(stop_str)].strip()

    print("\n" + "="*60)
    print("🤔 [User Query]:")
    print(args.query)
    print("-" * 60)
    print("🤖 [Tempo Response]:")
    print(pred)
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tempo Inference Script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Tempo model weights")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the input video")
    parser.add_argument("--query", type=str, required=True, help="Question to ask about the video")
    parser.add_argument("--video_fps", type=float, default=2.0)
    parser.add_argument("--max_frames", type=int, default=1024)
    parser.add_argument("--frame_windows", type=int, default=8)
    parser.add_argument("--frame_stride", type=int, default=8)
    parser.add_argument("--conv_version", type=str, default="qwen")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--disable_dynamic_compress", action="store_true", help="Disable dynamic visual compression")
    parser.add_argument("--tokenizer_model_max_length", type=int, default=16384, help="Max context length for the model")
    parser.add_argument("--visual_token_budget", type=int, default=8192, help="Budget for visual tokens during compression")

    
    args = parser.parse_args()
    main(args)