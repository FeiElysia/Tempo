import os
from typing import Optional, Union, List, Tuple

import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image, ImageSequence
from tqdm import tqdm
from decord import cpu, VideoReader

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

from tempo.builder import load_pretrained_model
from tempo.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from tempo.conversation import conv_templates, SeparatorStyle
from tempo.mm_datautils import (
    compute_segment_timestamp,
    KeywordsStoppingCriteria,
    process_qwen_content,
    tokenizer_image_token,
)

@register_model("tempo")
class Tempo(lmms):
    """
    Tempo Video LLM Model for lmms_eval (Open-Source Version - Latest Compat)
    """

    def __init__(
        self,
        pretrained: str = "path/to/your/model",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        use_flash_attention_2: bool = True,
        video_fps: float = 2.0,
        min_frames_num: int = 4,
        max_frames_num: int = 1024,
        model_max_length: int = 32768,
        inference_max_length: int = 16,
        frame_windows: int = 8,
        frame_stride: int = 8,
        conv_version: str = "qwen",
        visual_token_budget: int = 8192,
        disable_dynamic_compress: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        eval_logger.info(f"Start to init Tempo Evaluator")
        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        self.video_fps = video_fps
        self.min_frames_num = min_frames_num
        self.max_frames_num = max_frames_num
        self.inference_max_length = inference_max_length
        self.frame_windows = frame_windows
        self.frame_stride = frame_stride
        self.conv_version = conv_version
        self.context_len = model_max_length

        eval_logger.info(f"Loading model from {pretrained}")
        self._tokenizer, self._model, self.image_processor = load_pretrained_model(
            pretrained,
            device_map=self.device_map,
            use_flash_attn=use_flash_attention_2,
        )

        self._model.eval()

        self._model.get_model().config.tokenizer_model_max_length = model_max_length
        self._model.get_model().config.inference_max_length = inference_max_length
        self._model.config.use_cache = use_cache

        # ATA config
        self._model.config.visual_token_budget = visual_token_budget
        self._model.get_vision_tower_aux_list()[0].dynamic_compress = not disable_dynamic_compress
        eval_logger.info(f"Visual Token Budget: {visual_token_budget}, Dynamic Compress: {not disable_dynamic_compress}")

        self._model.cuda()
        self._model.to(torch.bfloat16)

        self._config = self._model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ]
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self._model)
            else:
                self._model = accelerator.prepare_model(self._model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.process_index
            self._world_size = self.accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

        eval_logger.info(f"Model loaded successfully. Rank: {self._rank}, World size: {self._world_size}")

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self.context_len

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for this model")

    def _compute_sample_indices(
        self,
        total_frames: int,
        original_fps: float,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
    ) -> list[int]:
        if end_frame is None:
            end_frame = total_frames - 1
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(start_frame, min(end_frame, total_frames - 1))
        clip_frames = end_frame - start_frame + 1
        
        if clip_frames <= 1:
            return [start_frame]
            
        target_fps = self.video_fps
        min_frames = self.min_frames_num
        max_frames = self.max_frames_num
        
        if original_fps is None or original_fps <= 0:
            original_fps = target_fps
            
        clip_duration = clip_frames / original_fps
        target_num_frames = max(1, round(clip_duration * target_fps))
        final_num_frames = min(max(target_num_frames, min_frames), max_frames)
        
        if final_num_frames == 1:
            return [end_frame]
            
        indices = np.round(np.linspace(start_frame, end_frame, final_num_frames)).astype(int)
        indices = np.clip(indices, start_frame, end_frame)
        return indices.tolist()

    def process_video(
        self,
        video: str,
        clip_start: Optional[float] = None,
        clip_end: Optional[float] = None,
    ) -> tuple:
        video_duration = None
        clip_duration = None
        if video.endswith(".npy"):
            all_frames = np.load(video)
            total_frames = len(all_frames)
            original_fps = self.video_fps
            video_duration = total_frames / original_fps
            image_size = all_frames[0].shape[:2]
            frame_idx = self._compute_sample_indices(total_frames, original_fps)
            image = all_frames[frame_idx]
            clip_duration = video_duration
        elif video.endswith(".gif"):
            gif = Image.open(video)
            all_frames = []
            total_duration_ms = 0
            for frame in ImageSequence.Iterator(gif):
                frame_copy = frame.copy()
                all_frames.append(frame_copy.convert("RGB"))
                frame_duration = frame.info.get("duration", 100)
                if frame_duration <= 10 or frame_duration > 2000:
                    frame_duration = 100
                total_duration_ms += frame_duration
            total_frames = len(all_frames)
            video_duration = (
                total_duration_ms / 1000.0 if total_duration_ms > 0 else total_frames / 10.0
            )
            original_fps = total_frames / video_duration if video_duration > 0 else 10.0
            image_size = all_frames[0].size
            frame_idx = self._compute_sample_indices(total_frames, original_fps)
            image = [all_frames[i] for i in frame_idx]
            clip_duration = video_duration
        elif os.path.isdir(video):
            files = sorted(os.listdir(video))
            all_frames = [Image.open(os.path.join(video, f)).convert("RGB") for f in files]
            total_frames = len(all_frames)
            original_fps = self.video_fps
            video_duration = total_frames / original_fps
            image_size = all_frames[0].size
            frame_idx = self._compute_sample_indices(total_frames, original_fps)
            image = [all_frames[i] for i in frame_idx]
            clip_duration = video_duration
        else:
            try:
                cores = len(os.sched_getaffinity(0))
            except AttributeError:
                import multiprocessing
                cores = multiprocessing.cpu_count()
            optimal_threads = min(max(1, cores - 2), 16)
            
            vr = VideoReader(video, ctx=cpu(0), num_threads=optimal_threads)
            total_frames = len(vr)
            original_fps = vr.get_avg_fps()
            video_duration = total_frames / original_fps
            clip_duration = video_duration

            start_frame, end_frame = 0, total_frames - 1
            if clip_start is not None and clip_end is not None:
                clip_start = max(0, min(clip_start, video_duration))
                clip_end = max(clip_start, min(video_duration, clip_end))
                start_frame = int(clip_start * original_fps)
                end_frame = min(int(clip_end * original_fps), total_frames - 1)
                clip_duration = clip_end - clip_start
                
            frame_idx = self._compute_sample_indices(
                total_frames, original_fps, start_frame, end_frame
            )
            image = vr.get_batch(frame_idx).asnumpy()
            image_size = image[0].shape[:2]
            
        real_fps = (
            len(image) / clip_duration if clip_duration and clip_duration > 0 else self.video_fps
        )

        eval_logger.debug(
            f"Processed video: {video}, frames: {len(image)}, real_fps: {real_fps:.2f}"
        )
        return real_fps, image

    def _process_visual_input(self, video: np.ndarray, question: str, real_fps: float):
        vlm_inputs = None
        seg_timestamps = None
        image_sizes = None
        processed_video = None

        vlm_inputs = process_qwen_content(
            video,
            "video",
            question,
            self.image_processor[0],
            real_fps,
            self.frame_windows,
            self.frame_stride,
            is_eval=True,
        )
        vlm_inputs = {key: v.cuda() for key, v in vlm_inputs.items()}
        
        seg_timestamps = compute_segment_timestamp(
            len(vlm_inputs["video_grid_thw"]),
            self.tokenizer,
            real_fps,
            self.frame_stride,
            self.frame_windows,
        )
                    
        return processed_video, image_sizes, vlm_inputs, seg_timestamps

    def _build_prompt(self, question: str) -> tuple:
        if getattr(self.model.config, "mm_use_im_start_end", False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + question
        conv = conv_templates[self.conv_version].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        return prompt, stop_str

    def _get_video_from_doc(self, doc: dict, visual_data) -> tuple:
        bound = doc.get("bound", None)
        clip_start = None
        clip_end = None
        if bound:
            clip_start, clip_end = bound[0], bound[1]
        if isinstance(visual_data, list):
            if len(visual_data) == 0:
                return None, clip_start, clip_end
            visual_data = visual_data[0]
        if isinstance(visual_data, str) or isinstance(visual_data, Image.Image):
            return visual_data, clip_start, clip_end
        return visual_data, clip_start, clip_end

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]

            visual_list = [
                doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id
            ]
            gen_kwargs = all_gen_kwargs[0]

            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str, list], but got {type(until)}")

            until = [item for item in until if item != "\n\n"]

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            for i, context in enumerate(contexts):
                if "<image>" in context:
                    contexts[i] = context.replace("<image>", "")
                    context = contexts[i]

                doc = self.task_dict[task][split][doc_id[i]]
                visual_data = visual_list[i]

                video_input, clip_start, clip_end = self._get_video_from_doc(doc, visual_data)
                if video_input is None:
                    eval_logger.warning(f"No visual data for doc_id: {doc_id[i]}")
                    res.append("")
                    pbar.update(1)
                    continue
                try:
                    if isinstance(video_input, str):
                        if not os.path.exists(video_input):
                            eval_logger.warning(f"Video file not found: {video_input}")
                            res.append("")
                            pbar.update(1)
                            continue
                        real_fps, image = self.process_video(video_input)

                    elif isinstance(video_input, Image.Image):
                        image = np.array(video_input.convert("RGB"))[np.newaxis, :]
                        real_fps = self.video_fps

                    elif isinstance(video_input, np.ndarray):
                        if video_input.ndim == 3:
                            video_input = video_input[np.newaxis, :]
                        image = video_input
                        real_fps = self.video_fps

                    else:
                        eval_logger.warning(f"Unsupported visual type: {type(video_input)}")
                        res.append("")
                        pbar.update(1)
                        continue

                except Exception as e:
                    eval_logger.error(f"Error processing video: {e}")
                    res.append("")
                    pbar.update(1)
                    continue

                processed_video, image_sizes, vlm_inputs, seg_timestamps = self._process_visual_input(image, context, real_fps)

                prompt, stop_str = self._build_prompt(context)

                input_ids = tokenizer_image_token(
                    prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                ).unsqueeze(0).cuda()
                
                stopping_criteria = KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)
                
                max_new_tokens = gen_kwargs.get("max_new_tokens", self.inference_max_length)
                temperature = gen_kwargs.get("temperature", 0.0)
                do_sample = temperature > 0
                
                with torch.inference_mode():
                    output_ids = self.model.generate(
                        input_ids,
                        images=processed_video,
                        image_sizes=image_sizes if processed_video is not None else None,
                        do_sample=do_sample,
                        temperature=temperature if do_sample else None,
                        max_new_tokens=max_new_tokens,
                        use_cache=self.use_cache,
                        stopping_criteria=[stopping_criteria],
                        vlm_inputs=vlm_inputs,
                        seg_timestamps=seg_timestamps,
                    )
                    
                if isinstance(output_ids, tuple):
                    output_ids = output_ids[0]
                    
                pred = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                if pred.endswith(stop_str):
                    pred = pred[: -len(stop_str)].strip()
                for term in until:
                    if len(term) > 0:
                        pred = pred.split(term)[0]
                pred = pred.strip()
                
                res.append(pred)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), pred)
                pbar.update(1)

        res = re_ords.get_original(res)
        pbar.close()
        
        return res

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented")