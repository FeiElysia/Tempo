import gc
import random
random.seed(42)

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from transformers.utils import is_torchdynamo_compiling
from transformers import AutoConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

class Qwen3VLTower(nn.Module):
    def __init__(self, vision_tower_aux_name, args, **kwargs):
        super(Qwen3VLTower, self).__init__()

        self.is_loaded = True # for compatibility
        self.model_path = vision_tower_aux_name
        self.dynamic_compress = getattr(args, "dynamic_compress", False)

        # load processor
        self.image_processor = Qwen3VLProcessor.from_pretrained(self.model_path)

        # load config
        self.config = AutoConfig.from_pretrained(self.model_path)
        self.config._attn_implementation = "flash_attention_2"
        self.config.dtype = torch.bfloat16

        # load model
        with init_empty_weights():
            self.vlm = Qwen3VLForConditionalGeneration(self.config)
            if hasattr(self.vlm, "lm_head"):
                del self.vlm.lm_head

        self.vlm.requires_grad_(False)

        self.hidden_size = self.config.text_config.hidden_size
        self.num_compression_tokens = args.num_compression_tokens

        self.compression_tokens = nn.Parameter(
            torch.empty(1, self.num_compression_tokens, self.hidden_size)
        )

    def smart_init_vision_tower(self):
        """Load only during Stage 0"""

        temp_model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=self.vlm.dtype,
            device_map="cpu",  # avoid multiple nodes and gpu conflicit
        )

        missing_keys, unexpected_keys = self.vlm.load_state_dict(temp_model.state_dict(), strict=False)

        if len(missing_keys) > 0:
            print(f"[Warning] Missing keys in Qwen3-VL loading: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"[Warning] Unexpected keys keys in Qwen3-VL loading: {unexpected_keys}")

        del temp_model
        gc.collect()
        torch.cuda.empty_cache()

        self.vlm.requires_grad_(False)

        with torch.no_grad():
            embed_weights = self.vlm.model.language_model.embed_tokens.weight
            mean = embed_weights.mean(dim=0)
            std = embed_weights.std(dim=0)

        self.compression_tokens.data = torch.normal(mean=mean.repeat(self.num_compression_tokens, 1), std=std.repeat(self.num_compression_tokens, 1)).unsqueeze(0)
        print(f"[Smart Init] Done. Shape: {self.compression_tokens.shape}")

    def smart_init_dynamic_compress(self):
        if getattr(self, "vlm_head", None) is None:
            temp_model = Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_path,
                dtype=self.vlm.dtype,
                device_map="cpu",  # avoid multiple nodes and gpu conflicit
            )
            self.vlm_head = temp_model.lm_head

            del temp_model
            gc.collect()
            torch.cuda.empty_cache()

        token_true_id = self.image_processor.tokenizer.get_vocab()["Yes"]
        token_false_id = self.image_processor.tokenizer.get_vocab()["No"]

        lm_head_weights = self.vlm_head.weight.data
        weight_yes = lm_head_weights[token_true_id]
        weight_no = lm_head_weights[token_false_id]

        D = weight_yes.size()[0]
        self.linear_layer = nn.Linear(D, 1, bias=False)
        with torch.no_grad():
            self.linear_layer.weight[0] = weight_yes - weight_no

        del self.vlm_head
        self.linear_layer.to("cuda")

        print(f"[Smart Init Router] Done!")

    def compute_relevance(self, mask_compression_token, batch_size, last_hidden_state):
        first_compression_idx = mask_compression_token.int().argmax(
            dim=1
        )  # (batch_size,)
        prev_idx = first_compression_idx - 1  # (batch_size,)
        batch_indices = torch.arange(batch_size, device=last_hidden_state.device)
        prev_token_features = last_hidden_state[
            batch_indices, prev_idx
        ]  # (batch_size, hidden_dim)

        scores = self.linear_layer(prev_token_features.float())
        scores = torch.sigmoid(scores).squeeze(-1).cpu().detach().tolist()

        return scores

    def load_model(self):
        # for compatible with other encoder
        pass

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        cache_position=None,
        **kwargs,
    ):
        if self.dynamic_compress and not hasattr(self, "linear_layer"):
            self.smart_init_dynamic_compress()

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        used_compression_tokens = self.num_compression_tokens

        if inputs_embeds is None:
            # process input_ids to insert learnable token
            batch_size, n_seq = input_ids.shape
            valid_lengths = (
                input_ids != self.image_processor.tokenizer.pad_token_id
            ).sum(dim=1)
            input_ids = torch.cat(
                [
                    input_ids,
                    torch.full(
                        (batch_size, used_compression_tokens),
                        # self.config.pad_token_id,
                        self.image_processor.tokenizer.pad_token_id,
                        dtype=input_ids.dtype,
                        device=input_ids.device,
                    ),
                ],
                dim=1,
            )
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.zeros(
                        (batch_size, used_compression_tokens),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ],
                dim=1,
            )
            inputs_embeds = self.vlm.get_input_embeddings()(input_ids)
        else:
            raise NotImplementedError(
                "Current only support input_ids as vlm compressor inputs"
            )

        image_mask = None
        video_mask = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.vlm.get_image_features(
                pixel_values, image_grid_thw
            )
            image_embeds = torch.cat(image_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            image_mask, _ = self.vlm.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.vlm.get_video_features(
                pixel_values_videos, video_grid_thw
            )
            video_embeds = torch.cat(video_embeds, dim=0).to(
                inputs_embeds.device, inputs_embeds.dtype
            )
            _, video_mask = self.vlm.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            # aggregate visual_pos_masks and deepstack_visual_embeds
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(
                deepstack_image_embeds, deepstack_video_embeds
            ):
                embed_joint = img_embed.new_zeros(
                    visual_pos_masks.sum(), img_embed.shape[-1]
                ).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        # ------------------------------------------------------------------
        # inputs_embeds, [Text + Image + Video]，shape: (B, L, D)
        # concat Learnable Tokens
        position_compression_token = (
            torch.arange(n_seq + used_compression_tokens, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        mask_compression_token = (
            position_compression_token >= valid_lengths.unsqueeze(1)
        ) & (
            position_compression_token
            < (valid_lengths + used_compression_tokens).unsqueeze(1)
        )
        compression_tokens_expanded = self.compression_tokens[
            :, :used_compression_tokens, :
        ].expand(batch_size, -1, -1)
        inputs_embeds[mask_compression_token] = compression_tokens_expanded.reshape(
            -1, self.hidden_size
        ).to(inputs_embeds.dtype)
        attention_mask.masked_fill_(mask_compression_token, 1)
        # ------------------------------------------------------------------

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask
                if not isinstance(attention_mask, dict)
                else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(
                    attention_mask_tensor[:, 0], dim1=1, dim2=2
                )
                # Only apply conversion for floating point tensors (inverted masks)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = (
                        attention_mask_tensor
                        / torch.finfo(attention_mask_tensor.dtype).min
                    )
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            # Calculate RoPE index once per generation in the pre-fill stage only.
            # When compiling, we can't check tensor values thus we check only input length
            # It is safe to assume that `length!=1` means we're in pre-fill because compiled
            # models currently cannot do asssisted decoding
            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )
            if (
                prefill_compiled_stage or prefill_noncompiled_stage
            ) or self.rope_deltas is None:
                position_ids, rope_deltas = self.vlm.model.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.vlm.model.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        last_hidden_state = outputs.last_hidden_state
        compression_features_flat = last_hidden_state[mask_compression_token]
        compression_features = compression_features_flat.reshape(
            batch_size, used_compression_tokens, -1
        )

        relevance_scores = None
        if self.dynamic_compress:
            relevance_scores = self.compute_relevance(mask_compression_token, batch_size, last_hidden_state)

        return compression_features, relevance_scores