import numpy as np

from vn2.data.state_panel import state_at_split


def test_state_at_split_zero_demand():
    demands = np.zeros(3, dtype=int)
    oh, q1, q2 = state_at_split(demands, split_idx=3, i0=5, it0=2, it1=1)
    assert oh == 8
    assert q1 == 0
    assert q2 == 0
