import random
import numpy as np
import torch

class VLMMultimodalProcessor:
    """SVLM-based vision compression."""

    @staticmethod
    def tokenize_vision_inputs(vision_language_model, vlm_inputs, is_video, chunk_size=4):
        return vision_language_model(**vlm_inputs)

    @staticmethod
    def add_seg_timestamp(vision_features, model, seg_timestamps, is_video):
        if not is_video:
            return vision_features

        device = vision_features[0].device
        dtype = vision_features[0].dtype

        max_len = max(len(ts) for ts in seg_timestamps)
        num_segments = len(seg_timestamps)
        # pad_token_id = getattr(model.config, "pad_token_id", 151643)
        pad_token_id = 151643
        timestamp_ids_tensor = torch.full(
            (num_segments, max_len), pad_token_id, dtype=torch.long, device=device
        )

        for i, ts in enumerate(seg_timestamps):
            length = len(ts)
            timestamp_ids_tensor[i, :length] = torch.tensor(ts, device=device)

        timestamp_embeds = model.get_input_embeddings()(timestamp_ids_tensor).to(dtype)

        final_vision_features = []
        for i in range(num_segments):
            if vision_features[i].shape[0] == 0:
                print("drop this segment directly.")
                continue
            final_vision_features.append(
                torch.cat(
                    [
                        timestamp_embeds[i][: len(seg_timestamps[i])],
                        vision_features[i],
                    ],
                    dim=0,
                )
            )

        # return torch.cat(final_vision_features, dim=0).unsqueeze(0)
        return torch.cat(final_vision_features, dim=0)  # (comp_frame1+comp_frame2+comp_frame3+..., d)

    @staticmethod
    def tome_merge(x: torch.Tensor, target_num: int) -> torch.Tensor:
        """
        Token Merging using bipartite soft matching.
        Reference: "Token Merging: Your ViT But Faster" (Bolya et al.)
        Args:
            x: (n, d) tensor of token features
            target_num: number of tokens to keep after merging
        Returns:
            merged tokens: (target_num, d) tensor
        """
        if target_num <= 0:
            raise ValueError("target_num must be positive")

        n, d = x.shape

        if target_num >= n:
            return x

        while x.shape[0] > target_num:
            current_n = x.shape[0]

            if current_n < 2:
                break

            t1 = (current_n + 1) // 2  # ceil(n/2) - source token
            t2 = current_n // 2  # floor(n/2) - target token

            if t2 == 0:
                break

            tokens_to_remove = current_n - target_num
            r = min(tokens_to_remove, t1)

            if r <= 0:
                break

            x_batch = x.unsqueeze(0)  # (1, n, d)
            k = x_batch / x_batch.norm(dim=-1, keepdim=True)
            a, b = k[..., ::2, :], k[..., 1::2, :]
            scores = a @ b.transpose(-1, -2)
            node_max, node_idx = scores.max(dim=-1)
            edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
            unm_idx = edge_idx[..., r:, :]
            src_idx = edge_idx[..., :r, :]
            dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)
            unm_idx = unm_idx.sort(dim=-2)[0]
            # merge
            src, dst = x_batch[..., ::2, :], x_batch[..., 1::2, :]
            batch, _, c = src.shape
            unm = src.gather(dim=-2, index=unm_idx.expand(batch, t1 - r, c))
            src_to_merge = src.gather(dim=-2, index=src_idx.expand(batch, r, c))
            dst = dst.scatter_add(-2, dst_idx.expand(batch, r, c), src_to_merge)
            # pooling
            ones = torch.ones(batch, r, 1, device=x.device, dtype=x.dtype)
            dst_counts = torch.ones(
                batch, dst.shape[1], 1, device=x.device, dtype=x.dtype
            )
            dst_counts = dst_counts.scatter_add(-2, dst_idx.expand(batch, r, 1), ones)
            dst = dst / dst_counts
            x = torch.cat([unm, dst], dim=-2).squeeze(0)  # (new_n, d)
        return x

    @staticmethod
    def topk_compress(
        vision_features,
        relevance_scores,
        is_video,
        k=0,
        drop_ratio=0.5,
        strategy="topk",
    ):
        """
        Drop/Truncate low-scoring segments to keep only first k tokens based on relevance scores
        Args:
            vision_features: (n_segment, n, d)
            relevance_scores: n_segment (list of scores between 0 and 1)
            k: number of tokens to keep, k=0 means drop directly
            drop_ratio: ratio of segments to drop/truncate
            strategy: "topk" (keep highest), "lastk" (keep lowest), "random" (random drop)
        Return a list of tensor with length of n_segment. Each tensor is (k, d) or (num_compression_token, d)
        """

        if not is_video or relevance_scores is None:
            return vision_features

        n_segment = vision_features.shape[0]
        if n_segment <= 1:
            print("video segment is equal/less than 1, not compressing")
            return vision_features

        if strategy == "topk":
            # in ascending order
            sorted_indices = sorted(range(n_segment), key=lambda i: relevance_scores[i])
        elif strategy == "lastk":
            # in descending order
            sorted_indices = sorted(
                range(n_segment), key=lambda i: relevance_scores[i], reverse=True
            )
        elif strategy == "random":
            # shuffle index, random
            sorted_indices = list(range(n_segment))
            random.shuffle(sorted_indices)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # truncate or drop ratio
        num_to_prune = int(n_segment * drop_ratio)
        print(
            f"[topk_compress] total segment: {n_segment}, drop/truncate segment: {num_to_prune}, keep first {k} token"
        )

        low_score_indices = set(sorted_indices[:num_to_prune])

        result = []
        for i in range(n_segment):
            if i in low_score_indices:
                result.append(vision_features[i, :k, :])  # shape: (k, d)
            else:
                result.append(vision_features[i])  # shape: (n, d)

        print(
            f"[topk_compress] segments: {n_segment} (pruned={num_to_prune}, kept={n_segment - num_to_prune}), "
            f"kept tokens for pruned segment={k}"
        )

        return result

    @staticmethod
    def adaptive_linear_budget_allocation(
        vision_features,
        relevance_scores,
        is_video,
        max_budget=8192,
        min_tokens=4,
        max_tokens=None,
        strategy="head",
    ):
        """
        soft token allocation based on min-max normalized relevance scores, Uses linear mapping instead of softmax, providing more aggressive sparsity
        Args:
            vision_features: (n_segment, n_tokens, d)
            relevance_scores: list/array of scores
            is_video: bool
            max_budget: the largest budget for a video
            min_tokens: minimum number of tokens to allocate to each segment
            max_tokens: maximum number of tokens to allocate to each segment
            strategy: "head", "tail", "random", "tome"
        Return a list of tensor with length of n_segment. Each tensor is (k, d), where k in [min_tokens, max_tokens]
        """
        if not is_video or relevance_scores is None:
            return vision_features, None

        n_segments = vision_features.shape[0]
        n_tokens_per_segment = vision_features.shape[1]
        max_tokens = min(max_tokens or n_tokens_per_segment, n_tokens_per_segment)

        base_budget = n_segments * min_tokens
        if base_budget > max_budget:
            actual_min = max(1, max_budget // n_segments)
            print(
                f"[adaptive_linear_budget_allocation] Warning: budget insufficient, "
                f"min_tokens: {min_tokens} -> {actual_min}"
            )
            min_tokens = actual_min
            base_budget = n_segments * min_tokens

        # Convert to tensor
        scores = torch.tensor(relevance_scores, dtype=torch.float32)
        score_min = scores.min()
        score_max = scores.max()
        score_range = score_max - score_min

        if score_range < 1e-8:
            k = min(max_tokens, max_budget // n_segments)
            return [vision_features[i, :k, :] for i in range(n_segments)], torch.full(
                (n_segments,), k, dtype=torch.long
            )

        # Normalize scores to [0, 1] range
        normalized_scores = (scores - score_min) / score_range

        # Linear mapping: [0, 1] -> [min_tokens, max_tokens]
        token_range = max_tokens - min_tokens
        ideal_allocations = (
            min_tokens + (normalized_scores * token_range).floor().long()
        )

        # ========== Budget Protection ==========
        total_desired = ideal_allocations.sum().item()

        if total_desired > max_budget:
            # Only scale down when total demand exceeds budget
            print(
                f"[adaptive_linear_budget_allocation] Warning: Desired budget "
                f"({total_desired}) exceeds max ({max_budget}). Scaling down..."
            )

            extra_budget = max_budget - base_budget

            # Prevent division by zero
            score_sum = normalized_scores.sum().item()
            if score_sum < 1e-8:
                # Fallback to uniform distribution if all normalized scores are ~0
                weights = torch.ones_like(normalized_scores) / n_segments
            else:
                weights = normalized_scores / score_sum

            # Distribute extra budget proportionally
            extra_allocation = (weights * extra_budget).floor().long()
            remainder = int(extra_budget - extra_allocation.sum().item())
            if remainder > 0:
                top_indices = torch.argsort(weights, descending=True)[:remainder]
                for idx in top_indices:
                    extra_allocation[idx] += 1

            allocations = min_tokens + extra_allocation
        else:
            allocations = ideal_allocations

        # Final clamp to ensure bounds
        allocations = allocations.clamp(min=min_tokens, max=max_tokens)

        # default, head truncation
        if strategy == "head":
            result = []
            for i in range(n_segments):
                k = min(allocations[i].item(), n_tokens_per_segment)
                result.append(vision_features[i, :k, :])

        elif strategy == "tail":
            result = []
            for i in range(n_segments):
                k = min(allocations[i].item(), n_tokens_per_segment)
                if k == 0:
                    result.append(vision_features[i, :0, :])
                else:
                    result.append(vision_features[i, -k:, :])

        elif strategy == "random":
            result = []
            for i in range(n_segments):
                k = min(allocations[i].item(), n_tokens_per_segment)
                if k == 0:
                    result.append(vision_features[i, :0, :])
                else:
                    indices = torch.randperm(
                        n_tokens_per_segment, device=vision_features.device
                    )[:k]
                    indices = indices.sort().values
                    result.append(vision_features[i, indices, :])

        elif strategy == "tome":
            result = []
            for i in range(n_segments):
                k = min(allocations[i].item(), n_tokens_per_segment)
                if k == 0:
                    result.append(vision_features[i, :0, :])
                elif k == n_tokens_per_segment:
                    result.append(vision_features[i])
                else:
                    # Token Merging
                    result.append(
                        VLMMultimodalProcessor.tome_merge(
                            vision_features[i], target_num=k
                        )
                    )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        total_used = allocations.sum().item()

        print(
            f"[adaptive_linear_budget_allocation] segments={n_segments}, "
            f"budget_used={total_used}/{max_budget}, "
            f"theoretical_range=[{min_tokens}, {max_tokens}], "
            f"actual_range=[{allocations.min().item():.0f}, {allocations.max().item():.0f}]",
            flush=True,
        )

        return result, allocations
