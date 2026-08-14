import torch

def mamba2_gated_scan(q, k, v, gamma):
    """
    Returns: dictionary containing sequence outputs and the final state
    """

    B,S, Dk = q.shape
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()

    Dv = v.shape[-1]

    state = torch.zeros(
        B,Dk, Dv,
        device = q.device,
        dtype = torch.float32
    )

    outputs = []

    for t in range(S):
        outer = (
            k_f[:,t, :, None] *
            v_f[:,t,None,:]
        )

        state = gamma[:,t,None, None] * state + outer
        y_t = torch.einsum('bd,bdv->bv', q_f[:,t], state)
        outputs.append(y_t)

    outputs = torch.stack(outputs,dim =1 )
    return {
        "outputs":outputs.to(dtype =q.dtype),
        "final_state":state.to(dtype=q.dtype)
    }

    