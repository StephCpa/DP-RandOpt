import numpy as np
import pytest

from dprandopt import designs
from dprandopt.sim import SCENARIOS, simulate_table


@pytest.fixture(scope="module")
def sim():
    return simulate_table(SCENARIOS["countdown_like"], n_candidates=300, pool=400, public=50, test=200, seed=0)


@pytest.mark.parametrize("design", ["base", "random", "public", "nonprivate", "D0", "D1", "D2", "D3", "D7", "D8"])
def test_designs_run(sim, design):
    rng = np.random.default_rng(0)
    t = sim.table
    s = designs.select(t, design, k=5, pool_cols=np.arange(200), rng=rng, eps=4.0,
                       neighbor_fn=sim.neighbor_fn(0.5))
    assert s.answers.shape[1] == t.test_size
    assert 0 <= s.vote_acc <= 1


def test_d1_large_eps_equals_plain_randopt(sim):
    rng = np.random.default_rng(1)
    cols = np.arange(400)
    plain = designs.select(sim.table, "nonprivate", k=5, pool_cols=cols, rng=rng)
    d1 = designs.select(sim.table, "D1", k=5, pool_cols=cols, rng=rng, eps=1e6)
    assert set(plain.members) == set(d1.members)


def test_scoretable_roundtrip(sim, tmp_path):
    from dprandopt.scoretable import ScoreTable
    p = tmp_path / "t.npz"
    sim.table.save(str(p))
    t2 = ScoreTable.load(str(p))
    assert np.array_equal(t2.pool_reward, sim.table.pool_reward)
    assert t2.reward_max == pytest.approx(sim.table.reward_max)
