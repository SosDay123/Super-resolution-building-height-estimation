import os
import sys
import numpy as np

# Ensure the project root is on the import path so that ``aggregate_utils``
# can be imported when tests run from a different working directory.
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from aggregate_utils import aggregate


def test_aggregate_shape():
    # Use dimensions divisible by the inverse of the scale to avoid edge
    # effects in the aggregation routine.
    data = np.ones((8, 20))
    scale = 0.25
    res = aggregate(data, scale)
    expected_shape = (int(8 * scale), int(20 * scale))
    assert res.shape == expected_shape
    # With all ones as input, the aggregated result should also be ones.
    assert np.allclose(res, np.ones(expected_shape))
