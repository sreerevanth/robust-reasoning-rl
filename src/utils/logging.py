"""Minimal structured console logging."""

import json
import logging
from typing import Any


def event(name: str, **fields: Any) -> None:
    logging.getLogger("robust_rl").info(json.dumps({"event": name, **fields}, allow_nan=False))
