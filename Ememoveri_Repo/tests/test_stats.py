import pytest


def test_signflip_and_effect_size():
    from ememoveri.stats import signflip_p, cohens_dz
    assert signflip_p([0.1] * 8) == pytest.approx(2 / 256)          # all positive: only the two extreme sign patterns
    assert signflip_p([0.1, -0.1, 0.1, -0.1]) == 1.0
    assert 0.0 < signflip_p([0.05, 0.02, -0.01, 0.04, 0.03]) < 1.0
    assert cohens_dz([1.0, 2.0, 3.0]) == pytest.approx(2.0)
