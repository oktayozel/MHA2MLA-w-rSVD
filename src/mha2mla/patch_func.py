import torch
import torch.nn as nn
from .svd_methods import SVD, low_rank_decomposition


def partial_rope_mask(model_args, mha2mla_args):
    """
    Generate different types of masks for partial rotary position embeddings (RoPE)
    based on configuration settings.
    Returns:
        Appropriate mask tensor based on the specified version
    """
    n_head = model_args.num_attention_heads
    n_k_head = model_args.num_key_value_heads
    if hasattr(model_args, "head_dim"):
        d_head = model_args.head_dim
    else:
        d_head = model_args.hidden_size // n_head
    d_head_half = d_head // 2
    rope_dim_for_mla = mha2mla_args.rope_dim_for_mla
    rope_dim_for_mla_half = rope_dim_for_mla // 2
    rope_version = mha2mla_args.partial_rope_version
    mask = torch.zeros(d_head)

    def select_high_frequency(mask):
        """
        Select high-frequency components (first rope_dim_for_mla dimensions)
        Returns:
            mask: Binary mask with 1s for the first rope_dim_for_mla dimensions
        """
        mask[:rope_dim_for_mla_half] = 1
        mask[d_head_half : d_head_half + rope_dim_for_mla_half] = 1
        q_masks = mask.repeat(n_head).bool()
        k_masks = mask.repeat(n_k_head).bool()
        return q_masks, k_masks

    def select_low_frequency(mask):
        """
        Select low-frequency components (last rope_dim_for_mla dimensions)
        Returns:
            mask: Binary mask with 1s for the last rope_dim_for_mla dimensions
        """
        mask[d_head - rope_dim_for_mla_half :] = 1
        mask[d_head_half - rope_dim_for_mla_half : d_head_half] = 1
        q_masks = mask.repeat(n_head).bool()
        k_masks = mask.repeat(n_k_head).bool()
        return q_masks, k_masks

    def select_uniform_frequency(mask, start_point):
        """
        Select uniformly distributed dimensions for RoPE
        Returns:
            mask: Binary mask with 1s at uniformly spaced positions
        """
        step = d_head // rope_dim_for_mla
        assert d_head_half % step == 0, "rope_dim_for_mla must be greater than 0"

        for i in range(start_point, d_head, step):
            mask[i] = 1
        q_masks = mask.repeat(n_head).bool()
        k_masks = mask.repeat(n_k_head).bool()
        return q_masks, k_masks

    def select_2norm_frequency(mask, rope_dim_for_mla):
        """
        Select dimensions based on 2-norm frequency importance
        Returns:
            mask: Binary mask with 1s for the top rope_dim_for_mla dimensions by 2-norm
        """
        # This is a placeholder implementation since the exact 2-norm selection
        # method was not detailed in the comments. In practice, this would
        # require statistics from the weight matrices to determine importance.
        with open(mha2mla_args.qk_tensor_path, "rb") as fin:
            qk_norm_rank = torch.load(fin, weights_only=True)

        k_masks = qk_norm_rank < rope_dim_for_mla_half
        if mha2mla_args.is_gqa2mha2mla:
            q_masks = k_masks
        else:
            q_masks = k_masks.repeat_interleave(n_head // n_k_head, dim=1)
        k_masks = k_masks.view(k_masks.size(0), -1)
        q_masks = q_masks.view(q_masks.size(0), -1)
        return q_masks, k_masks

    if rope_version == "high":
        return select_high_frequency(mask)
    elif rope_version == "low":
        return select_low_frequency(mask)
    elif rope_version == "uniform":
        return select_uniform_frequency(mask, mha2mla_args.uniform_start_point)
    elif rope_version == "2-norm":
        return select_2norm_frequency(d_head, rope_dim_for_mla)


class LowRankKVLinear(nn.Module):
    """
    A low-rank approximation of a linear layer.
    Instead of storing a full matrix W of shape (out_features, in_features),
    it stores two matrices: down of shape (in_features, low_rank) and
    up of shape (low_rank, out_features), such that W ≈ up.T @ down.T
    """

    def __init__(
        self,
        d_in,
        d_k_out,
        d_v_out,
        d_mid=0,
        k_approx=False,
        v_approx=False,
        kv_joint=False,
        bias=None,
    ):
        super().__init__()
        # TODO: add activations after down_kv
        if kv_joint:  # (x * W_q）* （x * down_kv * up_k）
            self.down_kv = nn.Linear(in_features=d_in, out_features=d_mid, bias=False)
        if not kv_joint and k_approx:
            self.down_k = nn.Linear(in_features=d_in, out_features=d_mid, bias=False)
        if not kv_joint and v_approx:
            self.down_v = nn.Linear(in_features=d_in, out_features=d_mid, bias=False)

        d_k_in = d_mid if k_approx else d_in
        self.up_k = nn.Linear(in_features=d_k_in, out_features=d_k_out, bias=bias)
        d_v_in = d_mid if v_approx else d_in
        self.up_v = nn.Linear(in_features=d_v_in, out_features=d_v_out, bias=bias)

    def reset_parameters(
        self,
        down_kv_weight=None,
        down_k_weight=None,
        down_v_weight=None,
        up_k_weight=None,
        up_v_weight=None,
        up_k_bias=None,
        up_v_bias=None,
    ):
        def _assign_weight(linear_layer, weight_tensor):
            target = linear_layer.weight.data
            if weight_tensor.shape == target.shape:
                target.copy_(weight_tensor)
            elif weight_tensor.T.shape == target.shape:
                target.copy_(weight_tensor.T)
            else:
                raise ValueError(
                    "Weight shape mismatch: got "
                    f"{tuple(weight_tensor.shape)} expected {tuple(target.shape)}"
                )

        if down_kv_weight is not None:
            _assign_weight(self.down_kv, down_kv_weight)
        if down_k_weight is not None:
            _assign_weight(self.down_k, down_k_weight)
        if down_v_weight is not None:
            _assign_weight(self.down_v, down_v_weight)
        if up_k_weight is not None:
            _assign_weight(self.up_k, up_k_weight)
        if up_v_weight is not None:
            _assign_weight(self.up_v, up_v_weight)
        if up_k_bias is not None:
            self.up_k.bias.data.copy_(up_k_bias)
        if up_v_bias is not None:
            self.up_v.bias.data.copy_(up_v_bias)

    def mha_forward(self, x):
        # x: (batch_size, seq_len, in_features)
        if hasattr(self, "down_kv"):
            k = v = self.down_kv(x)
        else:
            k = self.down_k(x) if hasattr(self, "down_k") else x
            v = self.down_v(x) if hasattr(self, "down_v") else x
        k = self.up_k(k)
        v = self.up_v(v)
        return k, v

    def mla_forward(self, x, q_hid, o_hid):
        kv = self.down_kv(x)
        # TODO: Triton kernel
        return kv


def svd_low_rank_approx(k_c_weight, k_c_bias, v_weight, v_bias, d_kv_mid, method, 
                        decomposition_method="svd", rsvd_oversampling=10, rsvd_n_iter=2):
    d_k_c, d_v, d_kv_in = k_c_weight.size(0), v_weight.size(0), v_weight.size(1)
    has_bias = k_c_bias is not None

    # Create decomposition function with configured parameters
    def decompose(X, r):
        return low_rank_decomposition(
            X,
            r,
            method=decomposition_method,
            oversampling=rsvd_oversampling,
            n_iter=rsvd_n_iter,
        )

    def effective_rank(*matrices):
        max_rank = min(min(mat.size(0), mat.size(1)) for mat in matrices if mat is not None)
        if max_rank == 0:
            raise ValueError("Cannot compute low-rank approximation for empty matrices")
        rank = min(d_kv_mid, max_rank)
        if rank == 0:
            raise ValueError(
                "Requested low-rank dimension is zero; adjust `low_rank` or mask settings"
            )
        return rank

    if method == "only_key":
        rank = effective_rank(k_c_weight)
        down_k, up_k = decompose(k_c_weight, rank)
        kv_proj = LowRankKVLinear(
            d_kv_in, d_k_c, d_v, d_mid=rank, k_approx=True, bias=has_bias
        )
        kv_proj.reset_parameters(
            down_k_weight=down_k,
            up_k_weight=up_k,
            up_v_weight=v_weight,
            up_k_bias=k_c_bias,
            up_v_bias=v_bias,
        )
    elif method == "only_value":
        rank = effective_rank(v_weight)
        down_v, up_v = decompose(v_weight, rank)
        kv_proj = LowRankKVLinear(
            d_kv_in, d_k_c, d_v, d_mid=rank, v_approx=True, bias=has_bias
        )
        kv_proj.reset_parameters(
            down_v_weight=down_v,
            up_k_weight=k_c_weight,
            up_v_weight=up_v,
            up_k_bias=k_c_bias,
            up_v_bias=v_bias,
        )
    elif method == "split":
        rank = effective_rank(k_c_weight, v_weight)
        down_k, up_k = decompose(k_c_weight, rank)
        down_v, up_v = decompose(v_weight, rank)
        kv_proj = LowRankKVLinear(
            d_kv_in,
            d_k_c,
            d_v,
            d_mid=rank,
            k_approx=True,
            v_approx=True,
            bias=has_bias,
        )
        kv_proj.reset_parameters(
            down_k_weight=down_k,
            down_v_weight=down_v,
            up_k_weight=up_k,
            up_v_weight=up_v,
            up_k_bias=k_c_bias,
            up_v_bias=v_bias,
        )
    elif method == "joint":
        joint_kv = torch.cat([k_c_weight, v_weight])
        rank = effective_rank(joint_kv)
        down_kv, up_kv = decompose(joint_kv, rank)
        up_k, up_v = up_kv.split([d_k_c, d_v])
        kv_proj = LowRankKVLinear(
            d_kv_in,
            d_k_c,
            d_v,
            d_mid=rank,
            k_approx=True,
            v_approx=True,
            kv_joint=True,
            bias=has_bias,
        )
        kv_proj.reset_parameters(
            down_kv_weight=down_kv,
            up_k_weight=up_k,
            up_v_weight=up_v,
            up_k_bias=k_c_bias,
            up_v_bias=v_bias,
        )
    elif method == "none":
        kv_proj = LowRankKVLinear(d_kv_in, d_k_c, d_v, bias=has_bias)
        kv_proj.reset_parameters(
            up_k_weight=k_c_weight,
            up_v_weight=v_weight,
            up_k_bias=k_c_bias,
            up_v_bias=v_bias,
        )
    return kv_proj
