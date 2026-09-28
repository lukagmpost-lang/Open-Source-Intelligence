import importlib.util
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "reddit_halfyear_multislice.py"
_SPEC = importlib.util.spec_from_file_location("reddit_halfyear_multislice", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _index(month: tuple[int, int]) -> int:
    year, mon = month
    return year * 12 + mon


def test_six_months_step_fifteen_months():
    gaps = [_index(right) - _index(left) for left, right in zip(_MODULE.MONTHS, _MODULE.MONTHS[1:])]
    assert _MODULE.MONTHS[0] == (2006, 5)
    assert _MODULE.MONTHS[-1] == (2012, 8)
    assert gaps == [15, 15, 15, 15, 15]


def test_omega_sweep_starts_at_the_baseline():
    assert _MODULE.SWEEP_OMEGAS[0] == 0.0
    assert _MODULE.SWEEP_OMEGAS == (0.0, 0.1, 0.5, 1.0, 2.0)


def test_four_slice_fallback_keeps_both_ends():
    reduced = _MODULE.reduce_to_four(_MODULE.MONTHS)
    assert reduced[0] == (2006, 5)
    assert reduced[-1] == (2012, 8)
    assert len(reduced) == 4
    assert len(set(reduced)) == 4
