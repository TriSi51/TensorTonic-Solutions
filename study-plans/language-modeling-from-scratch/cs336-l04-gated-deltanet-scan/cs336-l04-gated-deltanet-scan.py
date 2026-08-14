import torch

def gated_deltanet_scan(q, k, v, gamma, beta):
    """
    Returns: dictionary containing sequence outputs and the final state
    """
    

    B,S, Dk = q.shape

    q_f = q.float()
    k_f = k.float()
    v_f = v.float()

    beta_f = beta.float()
    gamma_f = gamma.float()
    Dv = v.shape[-1]
    state = torch.zeros(
        B,Dk, Dv,
        device= q.device,
        dtype = torch.float32
    )
    outputs= []
    for t in range(S):

        # gt = gamma[:,t,None, None]
        # bt = beta[:,t , None, None]
        # outer_kv = (
        #     k_f[:,t, :, None] *
        #     v_f[:,t, None, :]
        # )

        # projection =  torch.einsum('bd,bdv->bv', k_f[:,t], state)
        # erase = (
        #     k_f[:, t, :, None] *  # (B, Dk, 1) 
        #     projection[:,None, :]    # (B, 1, Dv)
        # )  # (B, Dk, Dv)

        # state = gt * state - gt* bt *erase + bt * outer_kv

        kt = k_f[:, t] # (B, Dk)
        bt = beta_f[:, t] # (B)
        gt= gamma_f[:,t]
        kt_col = kt.unsqueeze(-1) # (B, Dk,1 )
        kt_row = kt.unsqueeze(1)

        kkT = torch.bmm(kt_col, kt_row)  # (B, Dk, Dk)

        I = torch.eye(
            Dk,
            dtype = torch.float32,
            device = q.device
        ).unsqueeze(0) #(1, Dk, Dk)

        erase_matrix = I - bt[:, None, None] * kkT  # (B, Dk ,Dk)
        carried = torch.bmm(erase_matrix, state)
        
        carried = gt[:, None, None] * carried
        kvT = torch.bmm(k_f[:,t,:,None], v_f[:, t, None, :])
        
        state =  carried + bt[:, None, None] * kvT
        
        
        y_t = torch.einsum('bd,bdv->bv', q_f[:,t], state)
        outputs.append(y_t)
    outputs = torch.stack(outputs, dim = 1)
    return {
        "outputs": outputs.to(dtype = q.dtype),
        "final_state": state.to(dtype = q.dtype)
    }
        
        


        
    