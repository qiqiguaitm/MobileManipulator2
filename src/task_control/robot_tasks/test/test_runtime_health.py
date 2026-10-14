from types import SimpleNamespace
import pytest
from robot_tasks.runtime_health import check_localization


def test_eighty_percent_inclusive_and_below_rejected():
    for value in (0.80, 0.893, 1.0):
        check_localization(
            SimpleNamespace(has_converged=True, inlier_fraction=value), 0.80
        )
    for value in (0.799, 0.0, float("nan")):
        with pytest.raises(RuntimeError, match="Localization unreliable"):
            check_localization(
                SimpleNamespace(has_converged=True, inlier_fraction=value), 0.80
            )
    with pytest.raises(RuntimeError, match="converged=False"):
        check_localization(
            SimpleNamespace(has_converged=False, inlier_fraction=0.99), 0.80
        )


@pytest.mark.parametrize("limit", [0.0, -1.0, 1.1, float("nan")])
def test_invalid_threshold_fails_closed(limit):
    with pytest.raises(RuntimeError, match="must be in"):
        check_localization(
            SimpleNamespace(has_converged=True, inlier_fraction=0.99), limit
        )
