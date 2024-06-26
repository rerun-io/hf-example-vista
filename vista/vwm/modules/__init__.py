from __future__ import annotations

from .encoders.modules import GeneralConditioner

UNCONDITIONAL_CONFIG = {
    "target": "vista.vwm.modules.GeneralConditioner",
    "params": {"emb_models": list()}
}
