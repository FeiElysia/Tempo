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

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from tempo.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)

from tempo.multimodal_encoder.builder import build_vision_tower_aux_list
from tempo.multimodal_projector.builder import build_vision_projector
from tempo.vlm_multimodal_processor import VLMMultimodalProcessor


class TempoMetaModel:
    def __init__(self, config):
        super(TempoMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower_aux_list"):
            self.vision_tower_aux_list = nn.ModuleList(
                build_vision_tower_aux_list(config, delay_load=True)
            )
            config.mm_hidden_size = sum(
                [
                    vision_tower_aux.hidden_size for vision_tower_aux in self.vision_tower_aux_list
                ]
            )
            self.mm_projector = build_vision_projector(config)
        else:
            raise NotImplementedError(
                "mm_vision_tower_aux_list is not found in config. Please initialize vision modules in the subclass of TempoMetaModel."
            )

    def get_vision_tower_aux_list(self):
        vision_tower_aux_list = getattr(self, "vision_tower_aux_list", None)
        return vision_tower_aux_list

    def initialize_vision_modules(self, model_args, fsdp=None):
        # vision_hidden_size = model_args.vision_hidden_size
        vision_tower_aux_list = model_args.vision_tower_aux_list
        # vision_tower_aux_token_len_list = model_args.vision_tower_aux_token_len_list
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        self.config.mm_vision_tower_aux_list = vision_tower_aux_list
        # self.config.mm_vision_tower_aux_token_len_list = vision_tower_aux_token_len_list

        if self.get_vision_tower_aux_list() is None:
            vision_tower_aux_list = build_vision_tower_aux_list(model_args)
            if model_args.unfreeze_mm_vision_tower:
                self.vision_tower_aux_list = nn.ModuleList(vision_tower_aux_list)
            else:
                self.vision_tower_aux_list = vision_tower_aux_list
        else:
            vision_tower_aux_list = self.vision_tower_aux_list
            for vision_tower_aux in vision_tower_aux_list:
                vision_tower_aux.load_model()

            if model_args.unfreeze_mm_vision_tower and not isinstance(self.vision_tower_aux_list, nn.ModuleList):
                self.vision_tower_aux_list = nn.ModuleList(self.vision_tower_aux_list)

        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        # self.config.vision_hidden_size = vision_hidden_size

        if getattr(self, "mm_projector", None) is None:
            self.config.mm_hidden_size = sum(
                [
                    vision_tower_aux.hidden_size for vision_tower_aux in vision_tower_aux_list
                ]
            )
            self.mm_projector = build_vision_projector(self.config)
        else:
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {
                    k.split(keyword + ".")[1]: v
                    for k, v in weights.items()
                    if keyword + "." in k
                }

            self.mm_projector.load_state_dict(
                get_w(mm_projector_weights, "mm_projector"), strict=True
            )


class TempoMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower_aux_list(self):
        return self.get_model().get_vision_tower_aux_list()

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images=None,
        image_sizes=None,
        vlm_inputs=None,
        seg_timestamps=None,
        batch_split_size=None,
        relevance=None,
    ):
        if input_ids.shape[1] == 1: # inference
            return (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                None,
                labels,
            )

        is_video = "pixel_values_videos" in vlm_inputs

        compressed_features, relevance_scores = VLMMultimodalProcessor.tokenize_vision_inputs(self.get_vision_tower_aux_list()[0], vlm_inputs, is_video)
        compressed_features, count_allocations = (
            VLMMultimodalProcessor.adaptive_linear_budget_allocation(
                compressed_features,
                relevance_scores,
                is_video,
                max_budget=self.config.visual_token_budget if hasattr(self.config, "visual_token_budget") else 8192,
                min_tokens=4,
                strategy="head",
            )
        )

        # for visulization of the allocation results, can be removed
        self._demo_count_allocations = count_allocations

        if isinstance(compressed_features, list):
            seg_lens = [feat.shape[0] for feat in compressed_features]
            compressed_features = torch.cat(compressed_features, dim=0)
            image_features = self.get_model().mm_projector(compressed_features)
            print(f"[Total Segments: {len(seg_lens)}], compression features after dynamic compress", image_features.shape)
            image_features = list(torch.split(image_features, seg_lens, dim=0))  
        else:
            image_features = self.get_model().mm_projector(compressed_features)  
            print("final compression features for the whole batch:", image_features.shape)

        if is_video:
            # add timestamp embeddings for video inputs
            if batch_split_size is not None and len(batch_split_size) > 1: # batch training
                print(f"Number of segments for each video is: {batch_split_size}")
                start_idx = 0
                final_image_features_list = []
                for b_split_size in batch_split_size:
                    current_image_features = image_features[start_idx : start_idx + b_split_size]
                    current_seg_timestamp = seg_timestamps[start_idx : start_idx + b_split_size]
                    final_image_features_list.append(
                        VLMMultimodalProcessor.add_seg_timestamp(
                            current_image_features,
                            self.get_model(),
                            current_seg_timestamp,
                            is_video,
                        )
                    )
                    start_idx += b_split_size
            else:
                final_image_features_list = [
                    VLMMultimodalProcessor.add_seg_timestamp(
                        image_features, self.get_model(), seg_timestamps, is_video
                    )
                ]
        else:
            final_image_features_list = [img for img in image_features]

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()

        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )

        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        attention_mask = attention_mask | (input_ids == IMAGE_TOKEN_INDEX)

        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask)
        ]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()

            if num_images == 0:
                cur_image_features = final_image_features_list[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat(
                    [cur_input_embeds_1, cur_image_features[0:0]], dim=0
                )
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = (
                [-1]
                + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
                + [cur_input_ids.shape[0]]
            )

            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []

            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(
                    cur_input_ids[
                        image_token_indices[i] + 1 : image_token_indices[i + 1]
                    ]
                )
                cur_labels_noim.append(
                    cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]]
                )

            split_sizes_text = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(
                torch.cat(cur_input_ids_noim)
            )
            cur_input_embeds_no_im = torch.split(
                cur_input_embeds, split_sizes_text, dim=0
            )

            # for multi-image inputs, there is a bug.
            cur_new_input_embeds = []
            cur_new_labels = []
            text_len = sum([x.shape[0] for x in cur_input_embeds_no_im])
            visual_len = len(final_image_features_list[cur_image_idx])
            max_visual_len = (
                self.get_model().config.tokenizer_model_max_length
                - getattr(self.get_model().config, "inference_max_length", 16)
                - text_len
            )

            if max_visual_len < visual_len:
                final_image_features_list[cur_image_idx] = final_image_features_list[cur_image_idx][:max_visual_len]

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])

                if i < num_images:
                    try:
                        cur_image_features = final_image_features_list[cur_image_idx]
                    except IndexError:  
                        print(f"cur_image_idx={cur_image_idx} is not ok, get {num_images} images!!!")
                        cur_image_features = final_image_features_list[cur_image_idx - 1]

                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        position_ids = torch.zeros(
            (batch_size, max_len),
            dtype=position_ids.dtype,
            device=position_ids.device,
        )

        for i, (cur_new_embed, cur_new_labels) in enumerate(
            zip(new_input_embeds, new_labels)
        ):
            cur_len = cur_new_embed.shape[0]

            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0,
                        cur_len,
                        dtype=position_ids.dtype,
                        device=position_ids.device,
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0,
                        cur_len,
                        dtype=position_ids.dtype,
                        device=position_ids.device,
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return (
            None,
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
        )

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(
                    model_args.pretrain_mm_mlp_adapter, map_location="cpu"
                )
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[
                        -num_new_tokens:
                    ]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False