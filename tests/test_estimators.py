import pytest
from torch import tensor
from torch.cuda import is_available
from torch.distributions import Normal, StudentT, Categorical, MixtureSameFamily

from br_snis import br_snis, snis


@pytest.fixture()
def toy_dataset():
    device = 'cuda:0' if is_available() else 'cpu'
    weights = tensor([.2, .8], device=device)
    mix = Categorical(weights)
    comp = Normal(tensor([-.5, .5], device=device), tensor([.25, .25], device=device))
    target = MixtureSameFamily(mix, comp)
    proposal = StudentT(df=3)
    samples = proposal.sample((100_000,)).to(device)
    return {
        "log_weights": (target.log_prob(samples) - proposal.log_prob(samples)).reshape(2, 50_000),
        "f_values": (samples**2).reshape(2, 50_000, 1)
    }


def test_snis(toy_dataset):
    estimations = snis(**toy_dataset)
    assert estimations.shape == (2, 1)


def test_brsnis(toy_dataset):
    estimations = br_snis(**toy_dataset,
                          n_particles=10,
                          k_max=5000,
                          n_bootstrap=500)
    assert estimations.shape == (5000, 2, 500, 1)

