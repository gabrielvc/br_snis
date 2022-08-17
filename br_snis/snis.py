from torch import logsumexp, Tensor


def snis(log_weights: Tensor,
         f_values: Tensor) -> [Tensor, Tensor]:
    normalized_weights = (log_weights - logsumexp(log_weights, dim=1)[:, None]).exp()
    return (normalized_weights[..., None] * f_values).sum(dim=1)