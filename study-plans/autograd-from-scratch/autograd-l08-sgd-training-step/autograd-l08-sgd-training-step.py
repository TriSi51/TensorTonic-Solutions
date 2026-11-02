import torch

def sgd_training_step(inputs: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor, learning_rate: float) -> tuple:
    """
    Returns old loss, new loss, updated weights, updated bias, weight gradients, and bias gradient.
    """
    p = torch.tanh(inputs @ weights + bias)
    y = targets
    sse = (p - y) ** 2
    loss = torch.sum(sse, dim = 0)
    batch_size= inputs.shape[0]
    updated_weights = weights.clone()
    updated_bias = bias.clone()

    gradients = 2 * (p-y) * (1-p**2)
    weights_gradients = inputs.transpose(0,1) @ gradients
    bias_gradients = torch.sum(gradients)
    updated_weights = updated_weights - learning_rate * weights_gradients
    updated_bias = updated_bias - learning_rate * bias_gradients

    new_p = torch.tanh(inputs @ updated_weights + updated_bias)
    new_loss = ((new_p - y) ** 2).sum()
    return loss, new_loss, updated_weights, updated_bias, weights_gradients, bias_gradients
    
    