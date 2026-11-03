"""LLM arms: pinned settings, prompts, caching, caps, transcripts, billed tokens, parsing (spec §6.1).

API keys come from environment variables and are never written to logs or transcripts.
"""

from __future__ import annotations

ARMS = (
    "LLM-W",
    "LLM-noW",
    "LLM-wrongW",
    "LLM-W-nosynth",
    "LLM-W-rescale",
    "LLM-W-canon",
    "LLM-W-strip",
    "LLM-noW-strip",
)


def run_session(arm: str, model_cfg: dict, weights_view, api, budget_fbe: int, seed: int):
    """Run one automated session; return up to 3 candidate patches plus the full transcript record."""
    raise NotImplementedError("P2")
