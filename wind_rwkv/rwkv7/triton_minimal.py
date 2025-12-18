import torch
import triton
import triton.language as tl


@triton.jit
def rwkv_forward_block_kernel(
    r_ptr, w_ptr, k_ptr, v_ptr, a_ptr, b_ptr, S0_ptr, out_ptr, ST_ptr, sa_ptr,
    T: tl.constexpr, D: tl.constexpr, BS: tl.constexpr,
):
    """Forward pass: processes blocks of state matrix rows in parallel."""
    pid = tl.program_id(0)
    num_blocks = D // BS
    pid_bh = pid // num_blocks
    block_idx = pid % num_blocks
    row_start = block_idx * BS

    r_base = r_ptr + pid_bh * T * D
    w_base = w_ptr + pid_bh * T * D
    k_base = k_ptr + pid_bh * T * D
    v_base = v_ptr + pid_bh * T * D
    a_base = a_ptr + pid_bh * T * D
    b_base = b_ptr + pid_bh * T * D
    out_base = out_ptr + pid_bh * T * D
    sa_base = sa_ptr + pid_bh * T * D

    S0_base = S0_ptr + pid_bh * D * D
    ST_base = ST_ptr + pid_bh * D * D

    offsets_bs = tl.arange(0, BS)
    offsets_d = tl.arange(0, D)
    row_offsets = row_start + offsets_bs

    state_block = tl.load(S0_base + row_offsets[:, None] * D + offsets_d[None, :]).to(tl.float32)

    r_cur = r_base
    w_cur = w_base
    k_cur = k_base
    v_cur = v_base
    a_cur = a_base
    b_cur = b_base
    out_cur = out_base
    sa_cur = sa_base

    for t in range(T):
        r_t = tl.load(r_cur + offsets_d).to(tl.float32)
        w_t = tl.load(w_cur + offsets_d).to(tl.float32)
        k_t = tl.load(k_cur + offsets_d[None, :]).to(tl.float32)
        v_block = tl.load(v_cur + row_offsets[:, None]).to(tl.float32)
        a_t = tl.load(a_cur + offsets_d[:, None]).to(tl.float32)
        b_t = tl.load(b_cur + offsets_d[None, :]).to(tl.float32)

        sa_block = tl.dot(state_block, a_t)
        tl.store(sa_cur + row_offsets[:, None], sa_block.to(sa_ptr.dtype.element_ty))

        sab = sa_block * b_t
        kv_block = v_block * k_t
        state_block = state_block * w_t[None, :] + sab + kv_block

        out_block = tl.dot(state_block, r_t[:, None])
        tl.store(out_cur + row_offsets[:, None], out_block.to(out_ptr.dtype.element_ty))

        r_cur += D
        w_cur += D
        k_cur += D
        v_cur += D
        a_cur += D
        b_cur += D
        out_cur += D
        sa_cur += D

    tl.store(ST_base + row_offsets[:, None] * D + offsets_d[None, :], state_block.to(ST_ptr.dtype.element_ty))


@triton.jit
def rwkv_backward_block_kernel(
    r_ptr, w_ptr, k_ptr, v_ptr, a_ptr, b_ptr, sT_ptr, sa_ptr,
    dy_ptr, dsT_ptr,
    dr_ptr, dw_ptr, dk_ptr, dv_ptr, da_ptr, db_ptr, ds0_ptr,
    T: tl.constexpr, D: tl.constexpr, BS: tl.constexpr,
):
    """Backward pass: reverses state using stored sa values."""
    pid = tl.program_id(0)
    num_blocks = D // BS
    pid_bh = pid // num_blocks
    block_idx = pid % num_blocks
    row_start = block_idx * BS

    r_base = r_ptr + pid_bh * T * D
    w_base = w_ptr + pid_bh * T * D
    k_base = k_ptr + pid_bh * T * D
    v_base = v_ptr + pid_bh * T * D
    a_base = a_ptr + pid_bh * T * D
    b_base = b_ptr + pid_bh * T * D
    sa_base = sa_ptr + pid_bh * T * D
    dy_base = dy_ptr + pid_bh * T * D
    sT_base = sT_ptr + pid_bh * D * D
    dsT_base = dsT_ptr + pid_bh * D * D

    dr_base = dr_ptr + pid_bh * T * D
    dw_base = dw_ptr + pid_bh * T * D
    dk_base = dk_ptr + pid_bh * T * D
    dv_base = dv_ptr + pid_bh * T * D
    da_base = da_ptr + pid_bh * T * D
    db_base = db_ptr + pid_bh * T * D
    ds0_base = ds0_ptr + pid_bh * D * D

    offsets_d = tl.arange(0, D)
    row_offsets = row_start + tl.arange(0, BS)

    S_next = tl.load(sT_base + row_offsets[:, None] * D + offsets_d[None, :]).to(tl.float32)
    dS = tl.load(dsT_base + row_offsets[:, None] * D + offsets_d[None, :]).to(tl.float32)

    r_cur = r_base + (T - 1) * D
    w_cur = w_base + (T - 1) * D
    k_cur = k_base + (T - 1) * D
    v_cur = v_base + (T - 1) * D
    a_cur = a_base + (T - 1) * D
    b_cur = b_base + (T - 1) * D
    sa_cur = sa_base + (T - 1) * D
    dy_cur = dy_base + (T - 1) * D
    dr_cur = dr_base + (T - 1) * D
    dw_cur = dw_base + (T - 1) * D
    dk_cur = dk_base + (T - 1) * D
    dv_cur = dv_base + (T - 1) * D
    da_cur = da_base + (T - 1) * D
    db_cur = db_base + (T - 1) * D

    for t_rev in range(T):
        r_t = tl.load(r_cur + offsets_d).to(tl.float32)
        w_t = tl.load(w_cur + offsets_d).to(tl.float32)
        k_t = tl.load(k_cur + offsets_d).to(tl.float32)
        v_t = tl.load(v_cur + row_offsets).to(tl.float32)
        a_t = tl.load(a_cur + offsets_d).to(tl.float32)
        b_t = tl.load(b_cur + offsets_d).to(tl.float32)
        sa_t = tl.load(sa_cur + row_offsets).to(tl.float32)
        dy_t = tl.load(dy_cur + row_offsets).to(tl.float32)

        kv = v_t[:, None] * k_t[None, :]
        sab = sa_t[:, None] * b_t[None, :]
        S_t = (S_next - sab - kv) / w_t[None, :]

        dS += dy_t[:, None] * r_t[None, :]

        dr_t = tl.sum(S_next * dy_t[:, None], axis=0)
        tl.atomic_add(dr_cur + offsets_d, dr_t)

        dw_t = tl.sum(dS * S_t, axis=0)
        tl.atomic_add(dw_cur + offsets_d, dw_t)

        dk_t = tl.sum(dS * v_t[:, None], axis=0)
        tl.atomic_add(dk_cur + offsets_d, dk_t)

        dv_t = tl.sum(dS * k_t[None, :], axis=1, keep_dims=True)
        tl.atomic_add(dv_cur + row_offsets[:, None], dv_t)

        db_t = tl.sum(dS * sa_t[:, None], axis=0)
        tl.atomic_add(db_cur + offsets_d, db_t)

        dsa = tl.sum(dS * b_t[None, :], axis=1, keep_dims=True)

        da_t = tl.sum(S_t * dsa, axis=0)
        tl.atomic_add(da_cur + offsets_d, da_t)

        dS = dS * w_t[None, :] + dsa * a_t[None, :]
        S_next = S_t

        r_cur -= D
        w_cur -= D
        k_cur -= D
        v_cur -= D
        a_cur -= D
        b_cur -= D
        sa_cur -= D
        dy_cur -= D
        dr_cur -= D
        dw_cur -= D
        dk_cur -= D
        dv_cur -= D
        da_cur -= D
        db_cur -= D

    tl.store(ds0_base + row_offsets[:, None] * D + offsets_d[None, :], dS)


class RWKV7TritonMinimalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, w, k, v, a, b, s0):
        B, T, H, C = w.shape
        
        if s0 is None:
            s0 = torch.zeros(B, H, C, C, dtype=torch.float32, device=w.device)
        
        if s0.dtype != torch.float32:
            s0 = s0.float()
        
        w_preprocessed = torch.exp(-torch.exp(w))
        
        r_t = r.permute(0, 2, 1, 3).contiguous()
        w_t = w_preprocessed.permute(0, 2, 1, 3).contiguous()
        k_t = k.permute(0, 2, 1, 3).contiguous()
        v_t = v.permute(0, 2, 1, 3).contiguous()
        a_t = a.permute(0, 2, 1, 3).contiguous()
        b_t = b.permute(0, 2, 1, 3).contiguous()
        s0 = s0.contiguous()
        
        out_t = torch.empty_like(r_t)
        sT = torch.empty(B, H, C, C, dtype=torch.float32, device=r_t.device)
        sa_t = torch.empty(B, H, T, C, dtype=torch.float32, device=r_t.device)

        if C <= 64:
            BS = C
        elif C == 128:
            BS = 64
        elif C >= 256:
            BS = 32
        else:
            if C % 64 == 0:
                BS = 64
            elif C % 32 == 0:
                BS = 32
            elif C % 16 == 0:
                BS = 16
            else:
                raise ValueError(f"Head size C={C} must be divisible by 16, 32, or 64")

        def grid(_meta):
            num_blocks = C // BS
            return (B * H * num_blocks,)

        rwkv_forward_block_kernel[grid](
            r_t, w_t, k_t, v_t, a_t, b_t, s0, out_t, sT, sa_t,
            T=T, D=C, BS=BS,
        )
        
        out = out_t.permute(0, 2, 1, 3).contiguous()
        
        ctx.save_for_backward(r_t, w_t, k_t, v_t, a_t, b_t, sT, w, sa_t)
        ctx.shape_info = (B, T, H, C, BS)
        
        return out, sT
    
    @staticmethod
    def backward(ctx, dy, dsT):
        r_t, w_t, k_t, v_t, a_t, b_t, sT, w_raw, sa_t = ctx.saved_tensors
        B, T, H, C, BS = ctx.shape_info

        dy_t = dy.permute(0, 2, 1, 3).contiguous()

        if dsT is None:
            dsT_t = torch.zeros(B, H, C, C, device=r_t.device, dtype=torch.float32)
        else:
            dsT_t = dsT.float().contiguous() if dsT.dtype != torch.float32 else dsT.contiguous()

        dr_t = torch.zeros_like(r_t)
        dw_t = torch.zeros_like(w_t)
        dk_t = torch.zeros_like(k_t)
        dv_t = torch.zeros_like(v_t)
        da_t = torch.zeros_like(a_t)
        db_t = torch.zeros_like(b_t)
        ds0 = torch.empty(B, H, C, C, device=r_t.device, dtype=torch.float32)

        def grid(_meta):
            num_blocks = C // BS
            return (B * H * num_blocks,)

        rwkv_backward_block_kernel[grid](
            r_t, w_t, k_t, v_t, a_t, b_t, sT, sa_t,
            dy_t, dsT_t,
            dr_t, dw_t, dk_t, dv_t, da_t, db_t, ds0,
            T=T, D=C, BS=BS,
        )

        dr = dr_t.permute(0, 2, 1, 3).contiguous()
        dk = dk_t.permute(0, 2, 1, 3).contiguous()
        dv = dv_t.permute(0, 2, 1, 3).contiguous()
        da = da_t.permute(0, 2, 1, 3).contiguous()
        db = db_t.permute(0, 2, 1, 3).contiguous()

        dw_preprocessed = dw_t.permute(0, 2, 1, 3).contiguous()
        exp_w_raw = torch.exp(w_raw)
        w_preprocessed = torch.exp(-exp_w_raw)
        dw = dw_preprocessed * (-w_preprocessed * exp_w_raw)

        return dr, dw, dk, dv, da, db, ds0


def attn_triton_minimal(r, w, k, v, a, b, s0=None):
    """Forward pass only (no autograd)."""
    B, T, H, C = w.shape
    
    if s0 is None:
        s0 = torch.zeros(B, H, C, C, dtype=torch.float32, device=w.device)
    
    if s0.dtype != torch.float32:
        s0 = s0.float()
    
    w_preprocessed = torch.exp(-torch.exp(w))
    
    r_t = r.permute(0, 2, 1, 3).contiguous()
    w_t = w_preprocessed.permute(0, 2, 1, 3).contiguous()
    k_t = k.permute(0, 2, 1, 3).contiguous()
    v_t = v.permute(0, 2, 1, 3).contiguous()
    a_t = a.permute(0, 2, 1, 3).contiguous()
    b_t = b.permute(0, 2, 1, 3).contiguous()
    s0 = s0.contiguous()
    
    out_t = torch.empty_like(r_t)
    sT = torch.empty(B, H, C, C, dtype=torch.float32, device=r_t.device)
    sa_t = torch.empty(B, H, T, C, dtype=torch.float32, device=r_t.device)

    if C <= 64:
        BS = C
    elif C == 128:
        BS = 64
    elif C >= 256:
        BS = 32
    else:
        if C % 64 == 0:
            BS = 64
        elif C % 32 == 0:
            BS = 32
        elif C % 16 == 0:
            BS = 16
        else:
            raise ValueError(f"Head size C={C} must be divisible by 16, 32, or 64")

    def grid(_meta):
        num_blocks = C // BS
        return (B * H * num_blocks,)

    rwkv_forward_block_kernel[grid](
        r_t, w_t, k_t, v_t, a_t, b_t, s0, out_t, sT, sa_t,
        T=T, D=C, BS=BS,
    )
    
    out = out_t.permute(0, 2, 1, 3).contiguous()
    
    return out, sT


def attn_triton_minimal_with_grad(r, w, k, v, a, b, s0=None):
    """RWKV7 attention with automatic differentiation support."""
    return RWKV7TritonMinimalFunction.apply(r, w, k, v, a, b, s0)
