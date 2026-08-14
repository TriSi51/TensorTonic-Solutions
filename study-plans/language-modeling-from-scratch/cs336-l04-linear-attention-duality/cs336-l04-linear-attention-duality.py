import torch

def linear_attention_duality(q, k, v):
    """
    Returns: dictionary containing both outputs and the final state
    """

    # Parallel 
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()

    # scores = q_f @ k_f.transpose(-1,-2)
    # scores = torch.tril(scores)

    # output = scores @ v_f
    # return output.to(dtype = q.dtype)

    # k_t v_t^T for every t
    kv = k_f.unsqueeze(-1) * v_f.unsqueeze(-2)
    # (B, S, Dk, Dv)
    # S_t = sum{i<= t} k_i v_i^T
    states = torch.cumsum(kv, dim = 1)
    # (B, S, Dk, Dv)

    # y_t = q_t^T S_t
    parallel_outputs = torch.einsum('bsd,bsdv->bsv', q_f, states)
    final_state = states[:, -1]
    # (B,Dk, Dv)

    #Recurrent
    B, S, Dk = q.shape
    Dv = v.shape[-1]

    # S_(-1) = 0
    state = torch.zeros(
        B, Dk, Dv,
        device = q.device,
        dtype = torch.float32
    )

    recurrent_outputs = []
    for t in range(S):
        outer = (
            k_f[:, t, :, None] * 
            v_f[:, t, None, :]
        )

        state = state + outer

        #y_t = q_t^T S_t
        y_t = torch.einsum(
            'bd,bdv->bv',
            q_f[:,t],
            state
        )
        recurrent_outputs.append(y_t)
    recurrent_outputs = torch.stack(recurrent_outputs,dim = 1) # (B,S,V)

    return {
        "parallel_output": parallel_outputs.to(dtype = q.dtype),
        "recurrent_output": recurrent_outputs.to(dtype = q.dtype),
        "final_state": final_state.to(dtype= q.dtype)
    }