from functools import partial

import jax.numpy as jnp
from jax import jit, vmap
from jax import random
from jax.lax import dynamic_slice_in_dim
from jax.scipy.special import logsumexp


def br_snis_inner_loop(batched_data,
                       conditioning_f_value,
                       conditioning_log_weights,
                       key):
    log_weights = batched_data[:, 0]
    f_values = batched_data[:, 1:]
    all_f_values = jnp.concatenate([conditioning_f_value,
                                    f_values], axis=0)
    all_log_weights = jnp.concatenate([conditioning_log_weights,
                                       log_weights], axis=0)
    new_index = random.categorical(key, all_log_weights)
    all_weights = jnp.exp(all_log_weights - logsumexp(all_log_weights))
    expectation = (all_weights[:, None] * all_f_values).sum(axis=0)
    new_conditioning_log_weights = dynamic_slice_in_dim(all_log_weights, start_index=new_index, slice_size=1, axis=0)
    new_conditioning_f_values = dynamic_slice_in_dim(all_f_values, start_index=new_index, slice_size=1, axis=0)

    return expectation, new_conditioning_f_values, new_conditioning_log_weights


def br_snis_once(original_log_weights,
                 original_f_values,
                 key,
                 k_max,
                 n_particles):
    shuffled_values = random.shuffle(key,
                                     jnp.concatenate([original_log_weights.reshape(original_log_weights.shape[0], 1),
                                                      original_f_values], axis=-1),
                                     axis=0)
    conditioning_log_weights = shuffled_values[0:1, 0]
    conditioning_f_value = shuffled_values[0:1, 1:]

    exps = jnp.empty(shape=(k_max, original_f_values.shape[-1]))
    subkey = key
    for k in range(k_max):
        _, subkey = random.split(key=subkey, num=2)
        shuffled_batch = dynamic_slice_in_dim(shuffled_values, start_index=k*n_particles, slice_size=n_particles, axis=0)
        exp, conditioning_f_value, conditioning_log_weights = br_snis_inner_loop(shuffled_batch,
                                                                                 conditioning_f_value,
                                                                                 conditioning_log_weights,
                                                                                 subkey)

        exps = exps.at[k].set(exp)
    return exps


@partial(jit, static_argnums=(2, 3, 4))
def br_snis(f_values,
            log_weights,
            k_max,
            n_particles,
            n_bootstrap,
            key):
    subkeys = random.split(key, n_bootstrap)
    br_snis_bootstraped = vmap(partial(br_snis_once,
                                       k_max=k_max,
                                       n_particles=n_particles),
                               in_axes=[None, None, 0])
    br_snis_all = vmap(br_snis_bootstraped, in_axes=[0, 0, None])
    return br_snis_all(log_weights, f_values, subkeys).mean(axis=1)


@partial(jit, static_argnums=(2, 3, 4))
def br_snis2(f_values,
             log_weights,
             k_max,
             n_particles,
             n_bootstrap,
             key):
    subkeys = random.split(key, n_bootstrap)
    br_snis_for_one_bootstrap = vmap(partial(br_snis_once,
                                             k_max=k_max,
                                             n_particles=n_particles),
                                     in_axes=[0, 0, None])
    br_snis_bootstraped = vmap(br_snis_for_one_bootstrap, in_axes=[None, None, 0])
    return br_snis_bootstraped(log_weights, f_values, subkeys).mean(axis=0)


@partial(jit, static_argnums=(2, 3, 4))
def br_snis_with_repeat(f_values,
                        log_weights,
                        k_max,
                        n_particles,
                        n_bootstrap,
                        key):
    subkeys = random.split(key, n_bootstrap)
    return vmap(partial(br_snis_once,
                        k_max=k_max,
                        n_particles=n_particles))(log_weights.repeat(n_bootstrap, axis=0),
                                                  f_values.repeat(n_bootstrap, axis=0),
                                                  jnp.tile(subkeys, (f_values.shape[0], 1))).reshape(n_bootstrap,
                                                                                                     f_values.shape[0],
                                                                                                     k,
                                                                                                     f_values.shape[-1]).mean(axis=0)


if __name__ == '__main__':
    from jax import default_device, devices
    from timeit import timeit

    import matplotlib.pyplot as plt
    import numpy as np

    default_device(devices('gpu')[-2])
    key = random.PRNGKey(0)
    times1 = []
    times2 = []
    times3 = []
    n_particles = 10
    k = 5
    n_bootstrap = 70
    n_repeats = 1_000
    batch_sizes = range(1, 100, 5)
    for batch_size in batch_sizes:
        weights = jnp.log(random.uniform(key, (batch_size, n_particles*k,)))
        f_values = random.uniform(key, (batch_size, n_particles*k, 2)) ** 2
        times1.append(timeit(stmt='br_snis(f_values, weights, k, n_particles, n_bootstrap, key).block_until_ready()',
                             setup='br_snis(f_values, weights, k, n_particles, n_bootstrap, key).block_until_ready()',
                             number=n_repeats,
                             globals=globals()) / n_repeats)
        times2.append(timeit(stmt='br_snis2(f_values, weights, k, n_particles, n_bootstrap, key).block_until_ready()',
                             setup='br_snis2(f_values, weights, k, n_particles, n_bootstrap, key).block_until_ready()',
                             number=n_repeats,
                             globals=globals()) / n_repeats)
        times3.append(timeit(stmt='br_snis_with_repeat(f_values, weights, k, n_particles, n_bootstrap, key).block_until_ready()',
                             setup='br_snis_with_repeat(f_values, weights, k, n_particles, n_bootstrap, key).block_until_ready()',
                             number=n_repeats,
                             globals=globals()) / n_repeats)

        print(f"Execution time jax : {times1[-1]:.4f} {times2[-1]:.4f} {times3[-1]:.4f}")

    res2 = br_snis2(f_values, weights, k, n_particles, n_bootstrap, key).block_until_ready()
    res1 = br_snis(f_values, weights, k, n_particles, n_bootstrap, key).block_until_ready()
    res3 = br_snis_with_repeat(f_values, weights, k, n_particles, n_bootstrap, key).block_until_ready()

    np.testing.assert_allclose(res1, res2, rtol=1e-4)
    np.testing.assert_allclose(res1, res3, rtol=1e-4)

    plt.plot(batch_sizes, times1)
    plt.plot(batch_sizes, times2)
    plt.plot(batch_sizes, times3)
    plt.show()
