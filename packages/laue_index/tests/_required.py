"""Params lines for the keys every params file must now carry.

config_schema.REQUIRED_KEYS (0.8, decision D5): keys whose defaults used to
disagree between the RunImage, streaming and C parsers are fatal when absent.
Tests that write a minimal params file add them with :func:`with_required`;
the values are generic and a test's own lines for any of these keys win (they
are kept, and the helper does not add a second line for that key).
"""
from laue_index import config_schema as S

DEFAULTS = {
    "LatticeParameter": "0.352380 0.352380 0.352380 90 90 90",
    "P_Array": "0.028828 0.002715 0.512993",
    "R_Array": "-1.20161887 -1.21404493 -1.21852276",
    "NrPxX": "2048",
    "NrPxY": "2048",
    "PxX": "0.0002",
    "PxY": "0.0002",
    "MaxNrLaueSpots": "30",
    "MinIntensity": "50",
    "NMeadianPasses": "1",
    "MinGoodSpots": "2",
    "RobustFilter": "1",
    "BackgroundFile": "no_background_computed_from_frame1.bin",
}
assert set(DEFAULTS) == set(S.REQUIRED_KEYS), "update tests/_required.py"


def with_required(text: str = "", **override) -> str:
    """*text* plus a line for every required key it does not already set."""
    have = {ln.split("#", 1)[0].split()[0] for ln in text.splitlines()
            if ln.split("#", 1)[0].split()}
    vals = {**DEFAULTS, **{k: str(v) for k, v in override.items()}}
    extra = "".join(f"{k} {v}\n" for k, v in vals.items() if k not in have)
    if text and not text.endswith("\n"):
        text += "\n"
    return text + extra
