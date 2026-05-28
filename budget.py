"""Monthly token budget tracker for Yamanote.

Estimates per-launch cost from config.TOKENS_PER_LAUNCH × config.MODEL_PRICES_USD,
persists running spend to agents/budget.json, and exposes an `allows_launch`
gate the orchestrator checks before spawning a Claude subprocess.

The tracker is intentionally a coarse pre-flight estimator — actual token usage
is not visible from `claude -p` exit. Tune TOKENS_PER_LAUNCH from your own
agent logs over time. Resets at the start of each calendar month (UTC).
"""

import json
import os
import threading
import time
from typing import Optional

import config


def estimate_cost_usd(agent_name: str, model: str) -> float:
    """Estimate USD cost of one agent launch given the model it will run on.

    Train-suffixed names (e.g. "conductor:standard-0") map to their role.
    Returns 0.0 for unknown agent or unknown model — those launches are not
    counted against the budget, and not gated.
    """
    role = agent_name.split(":", 1)[0]
    tokens = config.TOKENS_PER_LAUNCH.get(role)
    prices = config.MODEL_PRICES_USD.get(model)
    if not tokens or not prices:
        return 0.0
    return (tokens["input"] * prices["input"] + tokens["output"] * prices["output"]) / 1_000_000


def _current_month() -> str:
    return time.strftime("%Y-%m", time.gmtime())


class BudgetTracker:
    """Thread-safe monthly spend tracker with file persistence."""

    def __init__(self, path: str, monthly_limit_usd: float):
        self.path = path
        self.monthly_limit_usd = float(monthly_limit_usd or 0)
        self._lock = threading.Lock()
        self._last_warned_at: float = 0.0
        self._state = self._load()
        # Roll over if the persisted month is stale at startup.
        self._maybe_roll_locked()

    # ── Persistence ─────────────────────────────────────────────────────

    def _load(self) -> dict:
        try:
            with open(self.path) as f:
                data = json.load(f)
            return {
                "month": str(data.get("month", _current_month())),
                "spend_usd": float(data.get("spend_usd", 0.0)),
                "launches_by_agent": dict(data.get("launches_by_agent", {})),
            }
        except (OSError, json.JSONDecodeError, ValueError):
            return {"month": _current_month(), "spend_usd": 0.0, "launches_by_agent": {}}

    def _save_locked(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self._state, f, indent=2, sort_keys=True)
            os.replace(tmp, self.path)
        except OSError:
            pass  # budget tracking is best-effort; don't crash the orchestrator

    def _maybe_roll_locked(self) -> None:
        now_month = _current_month()
        if self._state["month"] != now_month:
            self._state = {"month": now_month, "spend_usd": 0.0, "launches_by_agent": {}}
            self._last_warned_at = 0.0
            self._save_locked()

    # ── Public API ──────────────────────────────────────────────────────

    def allows_launch(self, projected_cost_usd: float) -> bool:
        """True if the projected cost fits inside the monthly cap (or no cap set)."""
        if self.monthly_limit_usd <= 0:
            return True
        with self._lock:
            self._maybe_roll_locked()
            return (self._state["spend_usd"] + projected_cost_usd) <= self.monthly_limit_usd

    def should_warn(self) -> bool:
        """Rate-limit the 'budget exhausted' log line. Call only when gated."""
        with self._lock:
            now = time.time()
            if now - self._last_warned_at >= config.BUDGET_WARN_INTERVAL:
                self._last_warned_at = now
                return True
            return False

    def record_launch(self, agent_name: str, cost_usd: float) -> None:
        """Record an actually-spawned launch. Called after the subprocess starts."""
        with self._lock:
            self._maybe_roll_locked()
            self._state["spend_usd"] += float(cost_usd)
            self._state["launches_by_agent"][agent_name] = (
                self._state["launches_by_agent"].get(agent_name, 0) + 1
            )
            self._save_locked()

    def snapshot(self) -> dict:
        """JSON-safe snapshot for /api/status and /metrics."""
        with self._lock:
            self._maybe_roll_locked()
            spend = self._state["spend_usd"]
            limit = self.monthly_limit_usd
            remaining: Optional[float] = max(0.0, limit - spend) if limit > 0 else None
            utilization = (spend / limit) if limit > 0 else 0.0
            return {
                "month": self._state["month"],
                "spend_usd": round(spend, 4),
                "limit_usd": limit if limit > 0 else None,
                "remaining_usd": round(remaining, 4) if remaining is not None else None,
                "utilization": round(utilization, 4),
                "enabled": limit > 0,
                "launches_by_agent": dict(self._state["launches_by_agent"]),
            }
