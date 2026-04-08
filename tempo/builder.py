#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch
from tempo.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
)
from transformers import AutoTokenizer

from tempo.language_model.modeling_tempo_qwen import TempoQwenForCausalLM

def load_pretrained_model(
    model_path,
    device_map="auto",
    device="cuda",
    use_flash_attn=False,
    **kwargs,
):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs["device_map"] = {"": device}

    kwargs["dtype"] = torch.float16
    if use_flash_attn:
        kwargs["attn_implementation"] = "flash_attention_2"

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = TempoQwenForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    add_tokens_flag = False
    if getattr(model.config, "mm_use_im_patch_token", False):
        add_tokens_flag = True
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if getattr(model.config, "mm_use_im_start_end", False):
        add_tokens_flag = True
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    if add_tokens_flag:
        model.resize_token_embeddings(len(tokenizer))

    vision_tower_aux_list = model.get_vision_tower_aux_list()
    for vision_tower_aux in vision_tower_aux_list:
        if not vision_tower_aux.is_loaded:
            vision_tower_aux.load_model(device_map=device_map)
        vision_tower_aux.to(device=device, dtype=torch.float16)

    image_processor = None
    image_processor = [vision_tower_aux.image_processor for vision_tower_aux in vision_tower_aux_list]

    return tokenizer, model, image_processor
