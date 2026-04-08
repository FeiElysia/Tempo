import os
import copy
from typing import List
from packaging import version
from collections.abc import Sequence

import torch
import tokenizers
import transformers
import numpy as np
from PIL import Image
from transformers import StoppingCriteria
from qwen_vl_utils import process_vision_info
from torch import distributed as dist
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    StateDictType,
)

from tempo import conversation as conversation_lib
from tempo.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse(
    "0.14"
)

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if (
                len(cur_keyword_ids) > 1
                and cur_keyword_ids[0] == tokenizer.bos_token_id
            ):
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def call_for_batch(
        self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [
            keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids
        ]
        for keyword_id in self.keyword_ids:
            truncated_output_ids = output_ids[0, -keyword_id.shape[0] :]
            if torch.equal(truncated_output_ids, keyword_id):
                return True
        outputs = self.tokenizer.batch_decode(
            output_ids[:, -offset:], skip_special_tokens=True
        )[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False

    def __call__(
        self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)


def safe_save_model_for_hf_trainer(
    trainer: transformers.Trainer, output_dir: str
) -> None:
    """Collects the state dict and dump to disk."""
    global_rank = dist.get_rank()
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    if len(trainer.args.fsdp) == 0:
        cpu_state_dict = trainer.model.state_dict()
    else:
        with FSDP.state_dict_type(
            trainer.model, StateDictType.FULL_STATE_DICT, save_policy
        ):
            cpu_state_dict = trainer.model.state_dict()

    for key in cpu_state_dict.keys():
        cpu_state_dict[key] = cpu_state_dict[key].to(torch.bfloat16)

    if global_rank == 0:
        trainer.model.config.save_pretrained(output_dir)
        current_folder = output_dir.split("/")[-1]
        parent_folder = os.path.dirname(output_dir)
        save_path = os.path.join(output_dir, "pytorch_model.bin")
        if getattr(trainer.args, "tune_mm_mlp_adapter", False) and not getattr(
            trainer.args, "tune_text_decoder", False
        ):
            # Only save Adapter
            keys_to_match = ["mm_projector"]
            if getattr(trainer.args, "use_im_start_end", False):
                keys_to_match.extend(["embed_tokens", "embed_in"])

            freeze_layer_remove = []
            for key in cpu_state_dict.keys():
                remove = True
                for key_match in keys_to_match:
                    if key_match in key:
                        remove = False
                        break
                if remove:
                    freeze_layer_remove.append(key)
            for key in freeze_layer_remove:
                del cpu_state_dict[key]

            if current_folder.startswith("checkpoint-"):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                save_path = os.path.join(mm_projector_folder, f"{current_folder}.bin")
            else:
                save_path = os.path.join(output_dir, f"mm_projector.bin")
        torch.save(cpu_state_dict, save_path)


def _tokenize_fn(
    strings: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers) -> None:
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2 : cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation: bool = True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["value"] = (
            BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        )
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def crop2square(pil_img):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        left = (width - height) // 2
        right = left + height
        top = 0
        bottom = height
        return pil_img.crop((left, top, right, bottom))
    else:
        top = (height - width) // 2
        bottom = top + width
        left = 0
        right = width
        return pil_img.crop((left, top, right, bottom))


def perpare_input_for_qwen_input(chunk_dict, pad_token_ids):
    """Currently, only batch size = 1 is supported for evaluation."""

    qwenvl_input_dict = {}
    has_video = any(["video" in key for key in list(chunk_dict.keys())])

    qwenvl_input_dict["input_ids"] = torch.nn.utils.rnn.pad_sequence(
        chunk_dict["vlm_input_ids"],
        batch_first=True,
        padding_value=pad_token_ids,
    )
    qwenvl_input_dict["attention_mask"] = torch.nn.utils.rnn.pad_sequence(
        chunk_dict["vlm_attention_mask"],
        batch_first=True,
        padding_value=0,
    )

    if has_video:
        qwenvl_input_dict["pixel_values_videos"] = torch.cat(chunk_dict["pixel_values_videos"], dim=0)
        qwenvl_input_dict["video_grid_thw"] = torch.cat(chunk_dict["video_grid_thw"], dim=0)
    else:
        qwenvl_input_dict["pixel_values"] = torch.cat(chunk_dict["pixel_values"], dim=0)
        qwenvl_input_dict["image_grid_thw"] = torch.cat(chunk_dict["image_grid_thw"], dim=0)

    return qwenvl_input_dict


def construct_message(
    content_data,
    data_type,
    query,
    multimodal_processor,
    sample_fps=1,
    return_text=False,
):
    # # prompt 0 (using during training)
    # system_message = (
    #     "You are a query-conditioned visual compressor. "
    #     "Store in the provided memory tokens the minimal visual information needed to answer the Query. "
    #     "Ignore irrelevant details."
    # )
    # prompt 1 (using during inference)
    system_message = (
        "You are a query-conditioned visual compressor. "
        "Store in the provided memory tokens the minimal visual information needed to answer the Query. "
        "Ignore irrelevant details. "
        "Now, before compressing, answer exactly 'Yes' or 'No': is this segment relevant to the Query?"
    )

    user_message = f"\nQuery:\n{query}"
    # assistant_message = "Scanning for target features... The visual confidence representation is:"
    assistant_message = None

    if data_type == "image":
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": content_data},
                    {"type": "text", "text": user_message},
                ],
            },
        ]
    elif data_type == "video":
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": content_data,
                        "sample_fps": sample_fps,
                    },
                    {"type": "text", "text": user_message},
                ],
            },
        ]
    else:
        raise ValueError(f"Unknown data type: {data_type}")

    if return_text:
        messages = multimodal_processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        if assistant_message is not None:
            messages = messages + assistant_message

    return messages


def video_process_with_frame_idx(
    chunk_frames, multimodal_processor, query, sample_fps=2, frame_offset=0
):
    messages = construct_message(
        chunk_frames, "video", query, multimodal_processor, sample_fps
    )
    text = multimodal_processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        [messages],
        return_video_kwargs=True,
        image_patch_size=16,
        return_video_metadata=True,
    )
    if video_inputs is not None:
        video_inputs, video_metadatas = zip(*video_inputs)
        video_inputs, video_metadatas = (
            list(video_inputs),
            list(video_metadatas),
        )
    else:
        video_metadatas = None

    if video_metadatas is not None:
        video_metadatas[0]["frames_indices"] = [
            f + frame_offset for f in video_metadatas[0]["frames_indices"]
        ]

    inputs = multimodal_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        video_metadata=video_metadatas,
        **video_kwargs,
        do_resize=False,
        return_tensors="pt",
    )

    return inputs


def process_qwen_content(
    content_data,
    data_type,
    sources,
    multimodal_processor,
    real_fps=None,
    frame_windows=8,
    frame_stride=8,
    is_eval=False,
):
    """
    content_data:
        - 'image': PIL.Image
        - 'video': List[PIL.Image] or np.ndarray (T, H, W, C)
    data_type: 'text', 'image' or 'video'
    """

    # query process
    if is_eval:
        # for evaluation, please input only text string
        assert isinstance(sources, str), "During evaluation, sources should be a single query string."
        query = sources
    else:
        conversation = sources[0]
        if isinstance(conversation, list):
            # This is acceptable during training to learn better representations,
            # but cannot be used during inference as it may lead to data leakage.
            # Currently, only single-turn dialogue is supported during inference.
            # Set the maximum number of dialogue turns to 8.
            human_queries = []
            for turn in conversation[:16]:
                if turn.get("from") == "human":
                    clean_text = turn["value"].replace("<image>", "").strip()
                    if clean_text:
                        human_queries.append(clean_text)

            query = "\n".join(
                [f"Context turn {i + 1}: {q}" for i, q in enumerate(human_queries)]
            )
        else:
            query = "Describe this content."
    if not query.strip():
        query = "Describe this content."

    chunk_results = []

    def resize_images(frames, resolution=512):
        resized_frames = []
        for f in frames:
            w, h = f.size
            max_edge = max(w, h)
            if max_edge > resolution:
                ratio = resolution / max_edge
                new_w = int(round((w * ratio) / 16) * 16)
                new_h = int(round((h * ratio) / 16) * 16)
                new_w = max(16, new_w)
                new_h = max(16, new_h)
                f = f.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)
            resized_frames.append(f)
        return resized_frames

    # === Text ===
    if data_type == "text":
        content_data = Image.new("RGB", (336, 336), color=(255, 255, 255))  # dummy image
        messages = construct_message(
            content_data, "image", query, multimodal_processor, return_text=True
        )  # str
        inputs = multimodal_processor(
            text=[messages],
            images=[content_data],
            padding=False,
            return_tensors="pt",
        )
        chunk_results.append(inputs)

    # === Image ===
    elif data_type == "image":
        if isinstance(content_data, list):
            # multi-image
            content_data = resize_images(content_data, resolution=512)
            messages = construct_message(
                content_data[0], data_type, query, multimodal_processor, return_text=True,
            )  # str
            inputs = multimodal_processor(
                text=[messages] * len(content_data),
                images=content_data,
                padding=True,
                return_tensors="pt",
            )
        else:
            messages = construct_message(
                content_data, data_type, query, multimodal_processor, return_text=True
            )  # str
            inputs = multimodal_processor(
                text=[messages],
                images=[content_data],
                padding=True,
                return_tensors="pt",
            )
        chunk_results.append(inputs)

    # === Video ===
    elif data_type == "video":
        if isinstance(content_data, np.ndarray):
            frames = [Image.fromarray(f) for f in content_data]
        else:
            frames = content_data

        if frames:
            frames = resize_images(frames, resolution=512)

        total_frames = len(frames)
        window_size = frame_windows
        stride = frame_stride

        for i in range(0, total_frames, stride):
            start_idx = i
            frame_offset = i  # use to compute timestamp
            end_idx = min(start_idx + window_size, total_frames)

            chunk_frames = frames[start_idx:end_idx]
            # if len(chunk_frames) < window_size:
            #     chunk_frames.extend(
            #         [chunk_frames[-1]] * (window_size - len(chunk_frames))
            #     )

            if len(chunk_frames) < window_size and len(chunk_frames) == 1:
                chunk_frames.append(chunk_frames[-1])
                print(f"Qwen processor requires at least 2 frames as video input, copy last frame to {len(chunk_frames)}")

            inputs = video_process_with_frame_idx(
                chunk_frames, multimodal_processor, query, real_fps, frame_offset,
            )
            chunk_results.append(inputs)

    else:
        raise ValueError(f"Unknown data type: {data_type}")

    # ============Group for batch process===================
    chunk_dict = {}
    for key in chunk_results[0]:
        if key in ["input_ids", "attention_mask"]:
            chunk_dict[f"vlm_{key}"] = [r[key].squeeze(dim=0) for r in chunk_results]
        else:
            chunk_dict[key] = [r[key] for r in chunk_results]

    if is_eval:
        return perpare_input_for_qwen_input(
            chunk_dict, multimodal_processor.tokenizer.pad_token_id
        )

    return chunk_dict


def compute_segment_timestamp(
    num_segments,
    tokenizer,
    real_fps,
    stride=None,
    window_size=None,
    use_center_timestamp=True,
):
    """
    The current version only supports non-overlapping segments.
    You need to modify the timestamp computation to support overlapping segments.
    """
    step = stride if stride is not None else window_size
    fps = real_fps if real_fps and real_fps > 0 else 1.0

    seg_timestamps_ids = []
    for i in range(num_segments):
        start_frame_idx = i * step
        if use_center_timestamp:
            frame_idx = start_frame_idx + (window_size / 2)
        else:
            frame_idx = start_frame_idx
        cur_timestamp_sec = frame_idx / fps
        text = f"<{cur_timestamp_sec:.1f} seconds>"

        ids = tokenizer.encode(text, add_special_tokens=False)
        seg_timestamps_ids.append(ids)

    return seg_timestamps_ids


def compute_sample_indices(
    total_frames: int,
    original_fps: float,
    target_fps: float,
    min_frames: int,
    max_frames: int,
) -> List[int]:
    if total_frames <= 1:
        return [0]

    if original_fps is None or original_fps <= 0:
        original_fps = target_fps

    video_duration = total_frames / original_fps
    target_num_frames = max(1, round(video_duration * target_fps))

    final_num_frames = target_num_frames
    if final_num_frames < min_frames:
        print(
            f"Upsampling video from {target_num_frames} to {min_frames} frames (min_frames limit)."
        )
        final_num_frames = min_frames
    elif final_num_frames > max_frames:
        print(
            f"Downsampling video from {target_num_frames} to {max_frames} frames (max_frames limit)."
        )
        final_num_frames = max_frames

    if final_num_frames == 1:
        return [total_frames - 1]

    indices = np.linspace(0, total_frames - 1, final_num_frames).astype(int)
    indices = np.clip(indices, 0, total_frames - 1)

    return indices.tolist()


def process_images(images, image_processor, model_cfg):
    # if image_processor is None:
    #     raise ValueError("image_processor cannot be None")
    if isinstance(image_processor, list):
        image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
        processor_aux_list = image_processor
        new_images_aux_list = []
        for image in images:
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            image_aux_list = []
            for processor_aux in processor_aux_list:
                image_aux = image
                if hasattr(processor_aux, "image_mean"):
                    try:
                        target_resolution = processor_aux.crop_size["height"]
                    except:
                        target_resolution = processor_aux.size["height"]
                    # image_aux = expand2square(
                    #     image_aux, tuple(int(x * 255) for x in processor_aux.image_mean)
                    # ).resize((target_resolution, target_resolution))
                    if image_aspect_ratio == "pad":
                        image_aux = expand2square(
                            image_aux,
                            tuple(int(x * 255) for x in processor_aux.image_mean),
                        )
                    elif image_aspect_ratio == "crop":
                        image_aux = crop2square(image_aux)
                    image_aux = image_aux.resize((target_resolution, target_resolution))

                image_aux = processor_aux.preprocess(image_aux, return_tensors="pt")[
                    "pixel_values"
                ][0]
                image_aux_list.append(image_aux)
            new_images_aux_list.append(image_aux_list)
        new_images_aux_list = [
            list(batch_image_aux) for batch_image_aux in zip(*new_images_aux_list)
        ]
        new_images_aux_list = [
            torch.stack(image_aux).half().cuda() for image_aux in new_images_aux_list
        ]
        return new_images_aux_list
    else:
        image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
        new_images = []
        if image_aspect_ratio == "pad":
            for image in images:
                image = expand2square(
                    image, tuple(int(x * 255) for x in image_processor.image_mean)
                )
                image = image_processor.preprocess(image, return_tensors="pt")[
                    "pixel_values"
                ][0]
                new_images.append(image)
        elif image_aspect_ratio == "crop":
            for image in images:
                image = crop2square(image)
                image = image_processor.preprocess(image, return_tensors="pt")[
                    "pixel_values"
                ][0]
                new_images.append(image)
        else:
            return image_processor(images, return_tensors="pt")["pixel_values"]
        if all(x.shape == new_images[0].shape for x in new_images):
            new_images = torch.stack(new_images, dim=0)
        return new_images


def preprocess_multimodal(sources: Sequence[str], data_args) -> dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            num_im = sentence["value"].count(DEFAULT_IMAGE_TOKEN)
            if num_im == 1 or "<video>" in sentence["value"]:
                # process only when the vision info is not multi-images
                sentence["value"] = (
                    sentence["value"]
                    .replace(DEFAULT_IMAGE_TOKEN, "")
                    .replace("<video>", "")
                    .strip()
                )
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence["value"] = sentence["value"].replace(
                        DEFAULT_IMAGE_TOKEN,
                        "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>",
                    )
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = (
                    DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                )
            sentence["value"] = sentence["value"].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
) -> dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [
                tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for rou in rounds:
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
) -> dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [
                tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def tokenizer_image_token(
    prompt,
    tokenizer,
    image_token_index=IMAGE_TOKEN_INDEX,
    return_tensors=None,
):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


def tokenizer_image_token_llama3(
    prompt,
    tokenizer,
    image_token_index=IMAGE_TOKEN_INDEX,
    return_tensors=None,
):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    for x in insert_separator(prompt_chunks, [image_token_index]):
        input_ids.extend(x)

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


def preprocess_qwen(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    system_message: str = "You are a helpful assistant.",
) -> dict:
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)

    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")

    unmask_tokens_idx = [198, im_start, im_end]
    # nl_tokens = tokenizer("\n").input_ids

    # Reset Qwen chat templates so that it won't include system message every time we apply
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # _system = tokenizer("system").input_ids + nl_tokens
    # _user = tokenizer("user").input_ids + nl_tokens
    # _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for source in sources:
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}]
        )
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)

            conv = [{"role": role, "content": content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    system_message: str = "You are a helpful assistant.",
) -> dict:
    # roles = {"human": "<|start_header_id|>user<|end_header_id|>", "gpt": "<|start_header_id|>assistant<|end_header_id|>"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)
    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    bos_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    unmask_tokens = [
        "<|begin_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "\n\n",
    ]
    unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]

    # After update, calling tokenizer of llama3 will
    # auto add bos id for the tokens. ヽ(｀⌒´)ﾉ
    def safe_tokenizer_llama3(text):
        input_ids = tokenizer(text).input_ids
        if input_ids[0] == bos_token_id:
            input_ids = input_ids[1:]
        return input_ids

    nl_tokens = tokenizer.convert_tokens_to_ids("\n\n")

    # chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{%- if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{%- endif %}"
    chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    tokenizer.chat_template = chat_template

    # Apply prompt templates
    input_ids, targets = [], []
    for source in sources:
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}]
            # pyre-fixme[6]: For 1st argument expected `Union[int, str]` but got `slice`.
        )[:-4]

        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)

            conv = [{"role": role, "content": content}]
            # First is bos token we don't need here
            encode_id = tokenizer.apply_chat_template(conv)[1:-4]
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    # print("input_ids", input_ids, flush=True)
    # print("targets", targets, flush=True)
    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def preprocess_llama_3_1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
) -> dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for source in sources:
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for sentence in source:
            if sentence["from"] == "Answer":
                sentence["from"] = "gpt"  # data bug
            role = roles[sentence["from"]]
            # assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [
                tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    # remove the first bos token
    if input_ids[0][0] == input_ids[0][1] == tokenizer.bos_token_id:
        input_ids = input_ids[:, 1:]
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3_1

    # Mask targets
    sep = "<|start_header_id|>" + conv.roles[1] + "<|end_header_id|>" + "\n\n"
    # sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.shape[0])

        rounds = conversation.split(conv.tokenizer.eos_token)
        rounds = [rounds[0]] + [
            rounds[idx] + rounds[idx + 1] for idx in range(1, len(rounds) - 1, 2)
        ]

        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2 and i != 0:
                break

            if i == 0:
                round_len = len(tokenizer(rou, add_special_tokens=False).input_ids)
                instruction_len = len(
                    tokenizer(rou, add_special_tokens=False).input_ids
                )

            else:
                parts[0] += sep
                if has_image:
                    round_len = len(tokenizer_image_token(rou, tokenizer)) + 1
                    instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
                else:
                    round_len = len(tokenizer(rou).input_ids) + 1
                    instruction_len = len(tokenizer(parts[0]).input_ids)

            # if i > 0: round_len += 1
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX
        cur_len = cur_len + len(tokenizer(sep, add_special_tokens=False).input_ids)

        # if cur_len > tokenizer.model_max_length: print(f"WARNING: max length context")
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_llama_3_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
) -> dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [
                tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    # remove the first bos token
    if input_ids[0][0] == input_ids[0][1] == tokenizer.bos_token_id:
        input_ids = input_ids[:, 1:]
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3_2

    # Mask targets
    sep = "<|start_header_id|>" + conv.roles[1] + "<|end_header_id|>" + "\n\n"
    # sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.shape[0])

        rounds = conversation.split(conv.tokenizer.eos_token)
        rounds = [rounds[0]] + [
            rounds[idx] + rounds[idx + 1] for idx in range(1, len(rounds) - 1, 2)
        ]

        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2 and i != 0:
                break

            if i == 0:
                round_len = len(tokenizer(rou, add_special_tokens=False).input_ids)
                instruction_len = len(
                    tokenizer(rou, add_special_tokens=False).input_ids
                )

            else:
                parts[0] += sep
                if has_image:
                    round_len = len(tokenizer_image_token(rou, tokenizer)) + 1
                    instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
                else:
                    round_len = len(tokenizer(rou).input_ids) + 1
                    instruction_len = len(tokenizer(parts[0]).input_ids)

            # if i > 0: round_len += 1
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX
        cur_len = cur_len + len(tokenizer(sep, add_special_tokens=False).input_ids)

        # if cur_len > tokenizer.model_max_length: print(f"WARNING: max length context")
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_phi3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
) -> dict:
    conv = conversation_lib.conv_templates["phi3"].copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack(
            [
                tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(
                conv.sep.join(rounds[conv_idx : conv_idx + 2])
            )  # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i == 0:
                round_len += 1
                instruction_len += 1
            else:
                round_len -= 2
                instruction_len -= 2

            if (
                i != 0
                and getattr(tokenizer, "legacy", False)
                and IS_TOKENIZER_GREATER_THAN_0_14
            ):
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
) -> dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack(
            [
                tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(
                conv.sep.join(rounds[conv_idx : conv_idx + 2])
            )  # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if (
                i != 0
                and getattr(tokenizer, "legacy", False)
                and IS_TOKENIZER_GREATER_THAN_0_14
            ):
                round_len += 1
                instruction_len += 1
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]["value"]
        source[0]["value"] = DEFAULT_IMAGE_TOKEN
        conversation = (
            source[0]["value"]
            + source[1]["value"]
            + conversation_lib.default_conversation.sep
        )
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversations
    ]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]["value"], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
) -> dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "phi3":
        return preprocess_phi3(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "qwen":
        return preprocess_qwen(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [
            tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
            for prompt in conversations
        ]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn(
                [header] + [s["value"] for s in source],
                tokenizer,
            )["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)