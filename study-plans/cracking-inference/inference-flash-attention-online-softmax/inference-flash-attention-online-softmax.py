import torch

def flash_attention_online_softmax(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_block_size: int,
    key_block_size: int,
    causal: bool = False,
) -> torch.Tensor:
    """
    Returns: attention output tensor of shape (batch, seq_q, d_v)
    """
    
    B,Sq, Dk = query.shape
    _, Sk, _ = key.shape
    Dv = value.shape[-1]

    scale = 1.0 / math.sqrt(Dk)

    output = torch.empty(
        (B,Sq, Dv),
        dtype = query.dtype
    )

    for q_start in range(0, Sq, query_block_size):
        q_end = min(q_start + query_block_size, Sq)

        # B, Bq, D
        Q = query[:, q_start:q_end, :]

        Bq = q_end - q_start

        # Running max per query row
        m = torch.full(
            (B, Bq, 1),
            float("-inf"),
            dtype = query.dtype,
        )
        # Running softmax denominator
        l = torch.zeros(
            (B, Bq, 1),
            dtype = query.dtype,
        )

        # Running unormalized output
        O = torch.zeros(
            (B, Bq, Dv),
            dtype = query.dtype,    
        )

        for k_start in range(0,Sk, key_block_size):
            k_end = min(k_start + key_block_size, Sk)
            
            # K_j: (B, Bk, D)
            K = key[:,k_start:k_end, :]

            # V_j: (B, Bk, Dv)
            V = value[:, k_start:k_end, :]

            scores = torch.matmul(Q, K.transpose(-2,-1) * scale)

            if causal:
                q_indices = torch.arange(
                    q_start,
                    q_end
                )[:, None]

                k_indices = torch.arange(
                    k_start,
                    k_end
                )[None,:]

                # mask[q,k]  = True when key is in the future
                mask = k_indices > q_indices
                scores.masked_fill_(mask.unsqueeze(0), float("-inf"))

            # B, Bq, 1
            block_max = scores.max(dim = -1, keepdim = True).values
            m_new = torch.maximum(m, block_max)
            alpha = torch.exp(m- m_new)

            # B, Bq, Bk
            P = torch.exp(scores-m_new)

            l_new = (
                l* alpha +
                P.sum(dim = -1, keepdim= True)
            )
            # numerator
            O = (
                O * alpha +
                torch.matmul(P,V)
            )

            m = m_new
            l = l_new

        output[:, q_start:q_end, :] = O / l
    return output