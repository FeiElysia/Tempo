import io
import os
import time
import torch
import numpy as np
import gradio as gr
import multiprocessing
from decord import cpu, VideoReader

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from scipy.interpolate import make_interp_spline
from PIL import Image

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
    max_frames_num: int = 1024
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

    return images, real_fps

def generate_allocation_plot(allocations):
    """
    Token allocation visualization function
    """
    if allocations is None or len(allocations) == 0:
        # if disable_dynamic_compress is True, we return a blank image
        return Image.new('RGB', (1600, 350), color='white')

    allocations = np.array(allocations)
    num_segments = len(allocations)

    plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})
    fig = plt.figure(figsize=(16, 3.5), layout='constrained')
    gs = fig.add_gridspec(2, 1, height_ratios=[0.15, 1.0], hspace=0.05)


    ax_heat = fig.add_subplot(gs[0])
    ax_heat.set_title(" ", pad=50)

    colors = ["#EBF5FB", "#85C1E9", "#F2D7D5", "#E74C3C", "#641E16"]
    cmap_custom = mcolors.LinearSegmentedColormap.from_list("custom_heat", colors)

    vmax_val = max(128, allocations.max())
    ax_heat.imshow([allocations], cmap=cmap_custom, aspect='auto', extent=[0.5, num_segments + 0.5, 0, 1], vmin=4, vmax=vmax_val)
    ax_heat.set_yticks([]) 
    ax_heat.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) 
    for spine in ax_heat.spines.values():
        spine.set_linewidth(1.2)

    ax_line = fig.add_subplot(gs[1], sharex=ax_heat)

    x = np.arange(1, num_segments + 1)

    if num_segments > 3:
        spl = make_interp_spline(x, allocations, k=3)
        x_smooth = np.linspace(1, num_segments, 800)
        y_smooth = spl(x_smooth)
        y_smooth = np.clip(y_smooth, 4, vmax_val)
    else:
        x_smooth = x
        y_smooth = allocations

    line_color = '#1A252C'
    fill_color = '#D5D8DC'

    ax_line.plot(x_smooth, y_smooth, color=line_color, linewidth=2.0)
    ax_line.fill_between(x_smooth, y_smooth, color=fill_color, alpha=0.4)

    ax_line.axhline(vmax_val, color='#C0392B', linestyle='--', linewidth=1.2, alpha=0.8)
    ax_line.axhline(4, color='#2980B9', linestyle='--', linewidth=1.2, alpha=0.8)

    ax_line.set_xlim(0.5, num_segments + 0.5)
    ax_line.set_ylim(0, vmax_val + 12)
    ax_line.set_ylabel("Tokens / Seg", fontsize=14, fontweight='bold')
    ax_line.set_xlabel("Temporal Segments", fontsize=14, fontweight='bold')
    ax_line.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax_line.spines['top'].set_visible(False)
    ax_line.spines['right'].set_visible(False)
    ax_line.spines['bottom'].set_linewidth(1.2)
    ax_line.spines['left'].set_linewidth(1.2)
    ax_line.grid(axis='y', linestyle=':', color='gray', alpha=0.5)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100, transparent=True)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# ==========================================
#            Load global model
# ==========================================
# Hugging Face Weights "Vision-CAIR/Tempo-6B"
# MODEL_PATH = os.environ.get("MODEL_PATH", "Vision-CAIR/Tempo-6B") 

# if deploying locally, set MODEL_PATH to your local checkpoint directory, e.g., "./checkpoints/Tempo-6B"
MODEL_PATH = "./checkpoints/Tempo-6B"

print(f"[Init] Loading Tempo model from {MODEL_PATH}...")
tokenizer, model, image_processor = load_pretrained_model(
    MODEL_PATH,
    device_map="cuda",
    use_flash_attn=True,
)

FIXED_MAX_LENGTH = 16384
model.config.tokenizer_model_max_length = FIXED_MAX_LENGTH
tokenizer.model_max_length = FIXED_MAX_LENGTH
model.eval()
model.to(torch.bfloat16)
print(f"[Init] Model loaded! Max context length set to {FIXED_MAX_LENGTH}.")


# ==========================================
#              inference
# ==========================================
def predict(video_path, query, max_frames, visual_token_budget, temperature, max_new_tokens, disable_dynamic_compress):
    if not video_path:
        return "⚠️ Error: Please upload a video first."
    if not query:
        return "⚠️ Error: Please enter a question."

    print(f"\n[Request] Video: {video_path} | Query: {query}")
    
    model.config.visual_token_budget = int(visual_token_budget)
    model.get_vision_tower_aux_list()[0].dynamic_compress = not disable_dynamic_compress

    # video process
    start_prep_time = time.perf_counter()
    try:
        video_frames, real_fps = load_video(video_path, video_fps=2.0, max_frames=int(max_frames))
    except Exception as e:
        return f"⚠️ Error loading video: {str(e)}"

    # process local compressor inputs
    frame_windows, frame_stride = 8, 8
    vlm_inputs = process_qwen_content(
        video_frames, "video", query, image_processor[0], real_fps, frame_windows, frame_stride, is_eval=True
    )
    vlm_inputs = {key: v.cuda() for key, v in vlm_inputs.items()}

    # compute timestamp for each segment
    seg_timestamps = compute_segment_timestamp(
        len(vlm_inputs["video_grid_thw"]), tokenizer, real_fps, frame_stride, frame_windows
    )

    # stat info
    num_segments = len(vlm_inputs["video_grid_thw"])
    segment_duration = frame_windows / real_fps
    stats_info = f"🎬 Video Stats: Total Segments: {num_segments}  |  Segment Duration: {segment_duration:.2f}s  |  Real FPS: {real_fps:.2f}"    
    
    # prompt
    if getattr(model.config, "mm_use_im_start_end", False):
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + query
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + query

    conv_version = "qwen"
    conv = conv_templates[conv_version].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

    # tokenization
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    model._demo_count_allocations = []

    start_infer_time = time.perf_counter()

    # generating
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=None, # Qwen-VL architecture usually uses vlm_inputs instead of raw images in kwargs if projector is vlm
            image_sizes=None,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            max_new_tokens=int(max_new_tokens),
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
    stats_info += f"\n⚡ Profiling  : Prep Time: {prep_duration:.2f}s  |  Inference Time: {infer_duration:.2f}s  |  Total: {total_duration:.2f}s"

    pred = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if pred.endswith(stop_str):
        pred = pred[: -len(stop_str)].strip()
    
    # token allocation plot
    allocations_data = model._demo_count_allocations
    plot_img = generate_allocation_plot(allocations_data)    
    
    return pred, plot_img, stats_info

# ==========================================
#                  UI
# ==========================================
with gr.Blocks(title="Tempo Video Understanding", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # ⏱️ Tempo: Small Vision-Language Models are Smart Compressors for Long Video Understanding
        Upload a video and ask any question! Tempo dynamically compresses visual tokens based on your query to achieve SOTA performance.
        **[🏠 Project Page](https://feielysia.github.io/)** | **[💻 GitHub](https://github.com/FeiElysia)** | **[📄 Paper](https://arxiv.org/abs/xxxx)** | **[👨‍💻 @Junjie Fei](https://feielysia.github.io/)**
        
        *⏳ **Slow preprocessing?** Try Examples 4 & 5 below, decrease `Max Sampled Frames` in Advanced Settings, or check our [GitHub](https://github.com/FeiElysia) for full-speed local deployment.*
        """
    )
    
    with gr.Row():
        # left column: inputs
        with gr.Column(scale=1):
            video_input = gr.Video(label="Upload Video")
            example_poster = gr.Image(label="Video Poster", interactive=False, height=150, visible=False)
            query_input = gr.Textbox(label="Your Question", placeholder="e.g., What is the person doing in the video?", lines=3)
            with gr.Row():
                clear_btn = gr.Button("🧹 Clear", variant="secondary")
                submit_btn = gr.Button("🚀 Generate Response", variant="primary")
            
            # hyperparameters
            with gr.Accordion("Advanced Settings", open=False):
                max_frames_slider = gr.Slider(minimum=16, maximum=2048, value=1024, step=16, label="Max Sampled Frames")
                budget_slider = gr.Slider(minimum=64, maximum=16384, value=8192, step=64, label="Visual Token Budget")
                temp_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.0, step=0.1, label="Temperature (0 = Greedy)")
                max_tokens_slider = gr.Slider(minimum=64, maximum=4096, value=1024, step=64, label="Max New Tokens")
                disable_compress_chk = gr.Checkbox(label="Disable Dynamic Compression (Baseline)", value=False)
                
        # right column: outputs
        with gr.Column(scale=1):
            output_text = gr.Textbox(label="Tempo Response", lines=12, interactive=False)
            stats_text = gr.Textbox(label="📊 Video Segment Stats", lines=1, interactive=False)
            output_plot = gr.Image(label="Query-Aware Visual Feature Intensity (Visual Token Allocation)", interactive=False, height=180)

    # clicking submit_btn or pressing enter in query_input will trigger prediction
    submit_btn.click(
        fn=predict,
        inputs=[video_input, query_input, max_frames_slider, budget_slider, temp_slider, max_tokens_slider, disable_compress_chk],
        outputs=[output_text, output_plot, stats_text]
    )
    
    query_input.submit(
        fn=predict,
        inputs=[video_input, query_input, max_frames_slider, budget_slider, temp_slider, max_tokens_slider, disable_compress_chk],
        outputs=[output_text, output_plot, stats_text]
    )

    clear_btn.click(
        fn=lambda: (None, None, None, None, None, None),
        inputs=None,
        outputs=[video_input, example_poster, query_input, output_text, stats_text, output_plot]
    )
    
    # Examples
    gr.Markdown("---")
    gr.Markdown("### 💡 Try an Example")
    gr.Examples(
        examples=[
            [
                "examples/hsr_helloworld.mp4", 
                "Task: Please examine the provided media and answer the following three questions regarding the specific puppy in the scene:\n"
                "Q1: What is the primary fur color of the puppy positioned on the swing?\n"
                "Q2: Specify the exact time interval (in seconds, e.g., XX-XXs) during which the puppy is seen sitting on the swing.\n"
                "Q3: Provide a brief description of the puppy's appearance and its surroundings.",
                "examples/meme_hsr_helloworld.png"
            ],
            [
                "examples/hsr_helloworld.mp4", 
                "Task: Please analyze the provided video and answer the following 7 questions precisely.\n"
                "Q1: How many performers are visible on the stage?\n"
                "Q2: Describe the architectural elements in the background. What historical civilization do they remind you of?\n"
                "Q3: What is happening in the night sky above the performers, and what does this suggest about the event?\n"
                "Q4: List the hair colors of the performers in order from left to right.\n"
                "Q5: Identify the specific musical instrument being played by the performer located on the far left of the stage.\n"
                "Q6: What is the specific time interval (in seconds, e.g., XX-XXs) during which this fireworks performance scene occurs in the video?\n"
                "Q7: Look at the audience in the foreground. How does their silhouette-like depiction affect the viewer's perspective of the stage?",
                "examples/performance_hsr_helloworld.png"
            ],
            [
                "examples/honkai3_becauseofyou.mp4",
                "What text appears in the center of the video behind a sea of pink flowers?",
                "examples/ocr_honkai3_becauseofyou.png"
            ],
            [
                "examples/videomme_fFjv93ACGo8.mp4",
                "How many red socks are above the fireplace at the end of this video?",
                "examples/cover_videomme_fFjv93ACGo8.png"
            ],
            [
                "examples/videomme_FjS2LzrHEO8.mp4",
                "What was the purpose of using a hammer to hit the car in the video?\n"
                "A. To show the hammer works well.\n"
                "B. To show the solidity of the car.\n"
                "C. To warn people not to hit cars with hammers.\n"
                "D. To illustrate that a hammer is harder than a bullet.",
                "examples/cover_videomme_FjS2LzrHEO8.png"
            ],
            [
                "examples/honkai3_becauseofyou.mp4",
                "Describe the video in detail.",
                "examples/description_honkai3_becauseofyou.png"
            ]
        ],
        inputs=[video_input, query_input, example_poster],
        cache_examples=False,
    )

if __name__ == "__main__":
    demo.queue().launch(share=True)

