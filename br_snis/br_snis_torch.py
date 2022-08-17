from torch import cat, logsumexp, Tensor, rand, arange, randperm, vstack, stack
from tqdm.auto import tqdm


def generate_permutations(n_bootstrap: int,
                          length: int,
                          device: str):
    return vstack([arange(length, device=device).unsqueeze(0) if i == 0 else randperm(length, device=device).unsqueeze(0) for i in range(n_bootstrap)])

# Index only version. It is slower than the actual
# def br_snis(f_values: Tensor,
#             log_weights: Tensor,
#             n_particles: int,
#             k: int,
#             n_bootstrap: int,
#             progress_bar: bool = False) -> [Tensor, Tensor]:
#     dim_f = f_values.shape[-1]
#     device = f_values.device
#     n_chains = f_values.shape[0]
#
#     permutations = generate_permutations(n_bootstrap=n_bootstrap,
#                                          length=k*(n_particles-1),
#                                          device=device).long()
#     permutations = permutations
#     exps_f = empty(k, n_chains, n_bootstrap, dim_f, device='cpu')
#     indexes_conditioning_particles = permutations[:, 0].unsqueeze(0).repeat(n_chains, 1)
#
#     range_n_chains = arange(n_chains, device=device)
#     for round_index in tqdm(range(k), disable=(~progress_bar)):
#         index_low, index_high = round_index*(n_particles - 1), (round_index+1)*(n_particles - 1)
#         round_indexes = cat([indexes_conditioning_particles.unsqueeze(-1),
#                              permutations[:, index_low:index_high].unsqueeze(0).repeat(n_chains, 1, 1)],
#                             dim=-1)
#
#         round_log_weights = log_weights[range_n_chains[:, None, None], round_indexes[range_n_chains]]
#         round_weights = (round_log_weights - logsumexp(round_log_weights, dim=-1)[..., None]).exp()
#
#         indexes_conditioning_particles = sample_index(round_weights)
#         exps_f[round_index, :, :, :] = (round_weights.unsqueeze(-1) * f_values[range_n_chains[:, None, None],
#                                                                                round_indexes[range_n_chains]]).sum(dim=-2).cpu()
#
#     return exps_f


def br_snis(f_values: Tensor,
            log_weights: Tensor,
            n_particles: int,
            k_max: int,
            n_bootstrap: int,
            progress_bar: bool = False) -> [Tensor, Tensor]:
    dim_f = f_values.shape[-1]
    device = f_values.device

    permutations = generate_permutations(n_bootstrap=n_bootstrap,
                                         length=k_max*n_particles,
                                         device=device)

    log_weight_conditioning_particle = log_weights[:, permutations[:, 0]]
    f_conditioning_particle = f_values[:, permutations[:, 0]]
    exps_f = []

    for round_index in tqdm(range(k_max), disable=(~progress_bar)):
        index_low, index_high = round_index*n_particles, (round_index+1)*n_particles
        round_f_values = cat([
            f_conditioning_particle.unsqueeze(-2),
            f_values[:, permutations[:, index_low:index_high]]],
            dim=-2)
        round_log_weights = cat([log_weight_conditioning_particle.unsqueeze(-1),
                                 log_weights[:,
                                 permutations[:, index_low:index_high]]
                                 ], dim=-1)
        round_weights = (round_log_weights - logsumexp(round_log_weights, dim=-1)[..., None]).exp()
        new_conditioning_particle_index_in_round = sample_index(round_weights)

        exp_f = (round_weights.unsqueeze(-1) * round_f_values).sum(dim=-2)

        f_conditioning_particle = round_f_values.gather(dim=-2,
                                                        index=new_conditioning_particle_index_in_round[..., None, None].repeat(1, 1, 1, dim_f)).squeeze(-2)

        log_weight_conditioning_particle = round_log_weights.gather(dim=-1,
                                                                    index=new_conditioning_particle_index_in_round[..., None]).squeeze(-1)
        exps_f.append(exp_f.cpu())
    return stack(exps_f, dim=0)


# Faster than pytorch categorical
def sample_index(p):
    return (p.cumsum(-1) > rand(p.shape[:-1], device=p.device)[..., None]).byte().argmax(-1)
