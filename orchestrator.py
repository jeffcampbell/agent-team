#!/usr/bin/env python3
"""Yamanote — multi-agent orchestrator for Claude Code agent personas."""

import glob
import json
import logging
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from collections import deque

import config
from metrics import METRICS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("orchestrator")


# ─── Activity log ────────────────────────────────────────────────────────────

def activity(msg: str):
    """Append a pretty-printed line to the activity log and also log it."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}]  {msg}\n"
    log.info(msg)
    try:
        with open(config.ACTIVITY_LOG, "a") as f:
            f.write(line)
    except OSError as e:
        print(f"[yamanote] activity log write failed: {e}", file=sys.stderr)


def _strip_markdown(text: str) -> str:
    """Remove markdown formatting characters from text for activity log readability."""
    text = re.sub(r'[*_`#>\[\]\-]+', '', text)
    return ' '.join(text.split())


def _strip_reason_prefix(text: str) -> str:
    """Strip verbose prefixes from agent decision reasons (e.g., 'Reasoning:', 'Why:')."""
    patterns = [
        r'^Why\s+.*?HOLD,\s+not\s+BUILD[\s:]*',
        r'^Why\s+.*?REJECT[\s:]*',
        r'^Why[\s:]*',
        r'^[\w\s]*?Reasoning[\s:.\d]*',
        r'^Reason[\s:]*',
        r'^Analysis[\s:]*',
    ]
    result = text
    for pattern in patterns:
        result = re.sub(pattern, '', result, flags=re.IGNORECASE)
        if result != text:
            break
    return result.strip()


def _first_line_truncated(text: str, limit: int = 150) -> str:
    """Extract first line of text and truncate to limit chars at word boundaries."""
    first_line = text.split('\n')[0].strip()
    if len(first_line) <= limit:
        return first_line
    truncated = first_line[:limit]
    last_space = truncated.rfind(' ')
    if last_space > limit - 30:
        return first_line[:last_space] + "..."
    return first_line[:limit - 3] + "..."


# ─── Failure / rejection log persistence ─────────────────────────────────────

def record_failure(title: str, reason: str):
    """Append a failure entry to the persistent failure log."""
    ts = time.strftime("%Y-%m-%d %H:%M")
    line = f"{ts} | {title} | {reason}\n"
    try:
        with open(config.FAILURE_LOG_PATH, "a") as f:
            f.write(line)
    except OSError:
        pass


def record_rejection(title: str, reason: str, project: str = ""):
    """Append a rejection entry to the persistent rejection log."""
    ts = time.strftime("%Y-%m-%d %H:%M")
    line = f"{ts} | {project} | {title} | {reason}\n"
    try:
        with open(config.REJECTION_LOG_PATH, "a") as f:
            f.write(line)
    except OSError:
        pass


def read_rejection_log(max_lines: int = 5, project: str = "", max_age_days: int = 7) -> str:
    """Read recent rejection log entries, filtered by project and age."""
    try:
        with open(config.REJECTION_LOG_PATH) as f:
            lines = f.readlines()
    except (OSError, FileNotFoundError):
        return "(none)"

    if project:
        lines = [l for l in lines if f"| {project} |" in l]

    if max_age_days > 0:
        cutoff = time.time() - (max_age_days * 86400)
        filtered = []
        for line in lines:
            try:
                date_str = line.split("|")[0].strip()
                entry_time = time.mktime(time.strptime(date_str, "%Y-%m-%d %H:%M"))
                if entry_time >= cutoff:
                    filtered.append(line)
            except (ValueError, IndexError):
                filtered.append(line)
        lines = filtered

    recent = lines[-max_lines:] if len(lines) > max_lines else lines
    return "".join(recent).strip() or "(none)"


def read_failure_log(max_lines: int = 15) -> str:
    """Read the last N lines from the failure log for dispatcher context."""
    try:
        with open(config.FAILURE_LOG_PATH) as f:
            lines = f.readlines()
        recent = lines[-max_lines:] if len(lines) > max_lines else lines
        return "".join(recent).strip() or "(none)"
    except (OSError, FileNotFoundError):
        return "(none)"


def _classify_spec_type(title: str) -> str:
    """Classify a spec title as feature/bugfix/hardening/refactor by keyword heuristic."""
    t = title.lower()
    if any(w in t for w in ("bug", "fix", "error", "crash", "fail", "broken", "patch", "hotfix")):
        return "bugfix"
    if any(w in t for w in ("test", "coverage", "assert", "unit-test", "integration")):
        return "hardening"
    if any(w in t for w in ("refactor", "cleanup", "reorganize", "restructure", "rename",
                             "simplify", "dedup", "remove-dead", "dead-code")):
        return "refactor"
    if any(w in t for w in ("monitor", "metric", "observ", "alert", "trace", "instrument")):
        return "hardening"
    return "feature"


# ─── Project scheduling ──────────────────────────────────────────────────────

def _is_in_schedule_window(schedule: str | None, now_hour: int | None = None) -> bool:
    """Check if current hour falls within a schedule string like '9-17' or '22-2'.

    None or empty means always eligible. Midnight wraparound is supported.
    Malformed strings are treated as always eligible (fail-open).
    """
    if not schedule:
        return True
    if now_hour is None:
        now_hour = time.localtime().tm_hour
    try:
        start, end = schedule.split("-")
        start_h, end_h = int(start), int(end)
    except (ValueError, AttributeError):
        return True
    if start_h <= end_h:
        return start_h <= now_hour <= end_h
    else:
        return now_hour >= start_h or now_hour <= end_h


# ─── Module-level log helpers (extracted from StationManager for reuse) ──────

def find_app_log(project_dir: str) -> str | None:
    """Resolve the project's application log file.

    Priority: AGENT_TEAM_APP_LOG_GLOB env var → common log file names.
    Returns the most-recently-modified match, or None.
    """
    patterns = []
    if config.APP_LOG_GLOB:
        patterns.append(config.APP_LOG_GLOB)
    patterns += ["logs/*.log", "*.log"]
    for pattern in patterns:
        matches = sorted(
            glob.glob(os.path.join(project_dir, pattern)),
            key=lambda p: os.path.getmtime(p),
            reverse=True,
        )
        if matches:
            return matches[0]
    return None


def fetch_railway_logs(environment: str, project_dir: str) -> str:
    """Fetch recent logs from Railway via CLI. Streams for RAILWAY_LOG_TIMEOUT seconds."""
    cmd = [
        "railway", "logs",
        "-e", environment,
        "-s", config.RAILWAY_SERVICE,
    ]
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            cwd=project_dir,
            start_new_session=True,
        )
        stdout, _ = proc.communicate(timeout=config.RAILWAY_LOG_TIMEOUT)
        return stdout
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, 9)
        stdout, _ = proc.communicate()
        return stdout
    except (OSError, FileNotFoundError) as e:
        log.warning("Railway CLI failed: %s", e)
        return ""


# ─── Pluggable log sources ────────────────────────────────────────────────────

class LogSource(ABC):
    """Abstract base for pluggable log sources used by the error watcher.

    Implement this to add new log backends (k8s, CloudWatch, Datadog, etc.)
    without modifying the orchestrator core. Register via
    StationManager.register_log_source().

    Each source tracks its own read position so independent callers (watcher
    vs. Signal) never interfere with each other's offsets.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier shown in log messages."""
        ...

    @abstractmethod
    def read_new_lines(self, project_dir: str) -> list[str]:
        """Return log lines written since the last call for this project_dir.

        On the very first call for a given project_dir, implementations should
        record the current position and return [] (don't scan old history).
        """
        ...


class FileLogSource(LogSource):
    """Read new lines from a local log file, using byte-offset tracking."""

    @property
    def name(self) -> str:
        return "file"

    def __init__(self):
        self._offsets: dict[str, int] = {}

    def read_new_lines(self, project_dir: str) -> list[str]:
        log_path = find_app_log(project_dir)
        if not log_path:
            return []

        if project_dir not in self._offsets:
            try:
                self._offsets[project_dir] = os.path.getsize(log_path)
            except OSError:
                pass
            return []

        stored = self._offsets[project_dir]
        try:
            size = os.path.getsize(log_path)
        except OSError:
            return []

        if size < stored:  # log rotation
            stored = 0
        if size == stored:
            return []

        try:
            with open(log_path, "r", errors="replace") as f:
                f.seek(stored)
                new_text = f.read()
            self._offsets[project_dir] = size
            return new_text.splitlines()
        except OSError:
            return []


class RailwayLogSource(LogSource):
    """Read new lines from Railway via CLI, throttled to once per minute."""

    _POLL_INTERVAL = 60  # seconds between Railway CLI invocations

    @property
    def name(self) -> str:
        return "railway"

    def __init__(self):
        self._last_seen: dict[str, str | None] = {}
        self._last_poll: float = 0.0

    def read_new_lines(self, project_dir: str) -> list[str]:
        if not config.RAILWAY_PROJECT:
            return []
        now = time.time()
        if now - self._last_poll < self._POLL_INTERVAL:
            return []
        self._last_poll = now

        output = fetch_railway_logs(config.RAILWAY_PRODUCTION_ENV, project_dir)
        if not output:
            return []

        lines = output.splitlines()
        if not lines:
            return []

        if project_dir not in self._last_seen:
            self._last_seen[project_dir] = lines[-1]
            return []

        last_seen = self._last_seen[project_dir]
        try:
            idx = lines.index(last_seen) if last_seen else -1
            new_lines = lines[idx + 1:]
        except ValueError:
            new_lines = lines

        if new_lines:
            self._last_seen[project_dir] = new_lines[-1]
        return new_lines


# Pattern: match lines containing ERROR/WARNING/CRITICAL/FATAL (case-insensitive)
_WATCHER_PATTERN = re.compile(r'\b(ERROR|WARNING|WARN|CRITICAL|FATAL)\b', re.IGNORECASE)


class AgentProcess:
    """Thin wrapper around a single Claude subprocess."""

    def __init__(self, name: str, prompt: str, cwd: str | None = None, model: str | None = None):
        self.name = name
        self.prompt = prompt
        self.cwd = cwd
        self.model = model or config.AGENT_MODELS.get(name, "claude-sonnet-4-5-20250929")
        self.proc: subprocess.Popen | None = None
        self.start_time: float | None = None
        self._output: str | None = None
        self._stderr: str | None = None
        self._live_log_path: str | None = None
        self._live_log_file = None

    def start(self) -> subprocess.Popen:
        cmd = [arg.format(prompt=self.prompt, model=self.model)
               for arg in config.CLAUDE_CMD_TEMPLATE]
        log.info("Starting agent %s (cwd=%s)", self.name, self.cwd or "default")
        # Strip CLAUDECODE env var so nested claude sessions are allowed
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        # Write stdout to a live log file so the dashboard can tail it during the run
        safe_name = self.name.replace(":", "-").replace("/", "-")
        self._live_log_path = f"/tmp/yamanote-{safe_name}-live.log"
        self._live_log_file = open(self._live_log_path, "w", buffering=1)
        self.proc = subprocess.Popen(
            cmd,
            stdout=self._live_log_file,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=self.cwd,
            start_new_session=True,  # own process group so timeout kills the whole tree
        )
        self.start_time = time.time()
        log.info("Agent %s started with PID %d", self.name, self.proc.pid)
        return self.proc

    def poll(self) -> bool:
        """Return True if the process has finished."""
        if self.proc is None:
            return True
        return self.proc.poll() is not None

    def is_timed_out(self) -> bool:
        if self.proc is None or self.start_time is None:
            return False
        return time.time() - self.start_time > config.AGENT_TIMEOUT_SECONDS

    def get_output(self) -> str:
        """Read stdout after completion. Blocks if still running."""
        if self._output is not None:
            return self._output
        if self.proc is None:
            return ""
        # Drain stderr pipe (blocks until process exits), then read stdout from file
        if self._stderr is None:
            self._stderr = self.proc.stderr.read() if self.proc.stderr else ""
        self.proc.wait()
        if self._live_log_file is not None:
            try:
                self._live_log_file.close()
            except Exception:
                pass
            self._live_log_file = None
        try:
            with open(self._live_log_path, "r") as f:
                self._output = f.read()
        except (OSError, TypeError):
            self._output = ""
        return self._output

    def get_stderr(self) -> str:
        if self._stderr is not None:
            return self._stderr
        if self.proc is None:
            return ""
        self.get_output()
        return self._stderr or ""

    def save_log(self, marker: str = ""):
        ts = time.strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(config.LOGS_DIR, f"{self.name}_{ts}.log")
        with open(log_path, "w") as f:
            if marker:
                f.write(f"{marker}\n")
            f.write(f"=== Agent: {self.name} ===\n")
            f.write(f"=== CWD: {self.cwd or 'default'} ===\n")
            f.write(f"=== Started: {time.ctime(self.start_time)} ===\n")
            f.write(f"=== Return code: {self.proc.returncode if self.proc else 'N/A'} ===\n\n")
            f.write("--- STDOUT ---\n")
            f.write(self.get_output())
            f.write("\n--- STDERR ---\n")
            f.write(self.get_stderr())
        log.info("Saved log for %s to %s", self.name, log_path)
        return log_path


class Train:
    """Encapsulates per-pipeline state for a single train."""

    def __init__(self, train_id: str, train_type: str, conductor_model: str, inspector_model: str, complexity: str):
        self.train_id = train_id
        self.train_type = train_type
        self.conductor_model = conductor_model
        self.inspector_model = inspector_model
        self.complexity = complexity

        # Pipeline state
        self.spec_path: str | None = None
        self.branch: str | None = None
        self.working_dir: str | None = None
        self.repo_dir: str | None = None  # original project repo (working_dir points to worktree)
        self.file_edits: dict[str, int] = {}
        self.edits_tallied: bool = False
        self.rework_count: int = 0
        self.spec_timeout_count: int = 0

        # Agent slots
        self.conductor: AgentProcess | None = None
        self.inspector: AgentProcess | None = None
        self.triage: AgentProcess | None = None

        # Triage gate state
        self.needs_triage: bool = False
        self.triage_failures: int = 0
        self.triage_cooldown_until: float = 0.0

        # Per-train cooldowns
        self.conductor_cooldown_until: float = 0.0
        self.inspector_cooldown_until: float = 0.0
        self.conductor_failures: int = 0
        self.inspector_failures: int = 0

        # SLA tracking
        self.spec_started_at: float = 0.0
        self.checkpoint_idle_since: float = 0.0

    def reset_pipeline(self):
        """Clear state after merge/cancel/entropy."""
        self.spec_path = None
        self.branch = None
        self.working_dir = None
        self.repo_dir = None
        self.file_edits.clear()
        self.edits_tallied = False
        self.rework_count = 0
        self.spec_timeout_count = 0
        self.spec_started_at = 0.0
        self.checkpoint_idle_since = 0.0
        self.needs_triage = False
        self.triage_failures = 0
        self.triage_cooldown_until = 0.0


class StationManager:
    """Main orchestration loop managing multi-train agent pipelines."""

    def __init__(self):
        # Ensure folder structure exists
        for d in (config.BACKLOG_DIR, config.REVIEW_DIR, config.LOGS_DIR):
            os.makedirs(d, exist_ok=True)

        # Build trains from config
        self.trains: list[Train] = []
        for train_type, cfg in config.TRAIN_CONFIG.items():
            for i in range(cfg["count"]):
                train = Train(
                    train_id=f"{train_type}-{i}",
                    train_type=train_type,
                    conductor_model=cfg["conductor_model"],
                    inspector_model=cfg["inspector_model"],
                    complexity=cfg["complexity"],
                )
                self.trains.append(train)

        # Global agents (not per-train)
        self.active_agents: dict[str, AgentProcess | None] = {
            "dispatcher": None,
            "signal": None,
            "station_manager": None,
            "ops": None,
            "triage": None,
        }
        self.last_merge_commit: str | None = None
        self._dispatcher_skip_logged_trains: set[str] = set()

        # Cost guardrail: track agent launches in a rolling window
        self.launch_times: deque[float] = deque()
        self.sleep_until: float = 0.0

        # Error cooldown: don't retry agents immediately after failures
        self.agent_cooldowns: dict[str, float] = {}  # agent name → earliest retry time
        self.consecutive_failures: dict[str, int] = {}  # agent name → failure streak
        self.last_launch_times: dict[str, float] = {}  # agent name → last launch timestamp

        # Signal high-water mark: only analyze new log lines since last run
        self.sre_log_offsets: dict[str, int] = {}  # project_dir → byte offset in app.log
        self._sre_prev_offsets: dict[str, int] = {}  # offset before last Signal read (for rollback on failure)
        self.last_merge_time: float = 0.0  # timestamp of last merge (to skip Signal during deployment)

        # Log watcher: pluggable sources + dedup state
        # watcher_log_offsets have moved into FileLogSource._offsets
        self._log_sources: list[LogSource] = [FileLogSource()]
        if config.RAILWAY_PROJECT:
            self._log_sources.append(RailwayLogSource())
        # Dedup: error signature → timestamp when we last filed a spec for it
        self.watcher_recent_specs: dict[str, float] = {}

        # Wake detection logging throttle (reduce log noise)
        self.last_wake_log_time: float = 0.0

        # Ops agent: track HEAD before ops launch to detect new commits
        self._ops_head_before: str | None = None
        # Deferred restart: set True when ops wants to restart but a conductor is mid-run
        self.restart_pending: bool = False

        # Uptime tracking (used by dashboard)
        self.start_time: float = time.time()

        # SLA: track when all trains went idle
        self.all_idle_since: float = 0.0

        # Stall detection: consecutive triage rejections per project, and stall resume times
        self._project_rejection_counts: dict[str, int] = {}
        self._stalled_projects: dict[str, float] = {}  # project → resume_time

        # Drafts recycler: last time stale HOLD specs were moved back to backlog
        self._last_draft_recycle: float = 0.0

        # Worktree GC: projects that have had worktrees created, and last GC time
        self._seen_project_dirs: set[str] = set()
        self._last_worktree_gc: float = 0.0

        # Log rotation: track last prune time to gate the expensive directory scan
        self._last_log_prune: float = 0.0

        # Per-tick caches — reset each tick to avoid redundant filesystem I/O
        self._tick_id: int = 0
        self._cached_backlog_count: int = 0
        self._cached_backlog_count_tick: int = -1
        self._cached_backlog_specs: list[str] = []
        self._cached_backlog_specs_tick: int = -1
        self._cached_backlog_specs_complexity: str | None = None
        self._cached_open_bugs: list[dict] = []
        self._cached_open_bugs_tick: int = -1

        # Don't run Dispatcher or Ops immediately on startup — wait for activity to accumulate
        self.last_launch_times["dispatcher"] = time.time()
        self.last_launch_times["ops"] = time.time()

        # Recover orphaned .in_progress specs from previous runs
        self._recover_orphaned_specs()

    # ─── Helpers ─────────────────────────────────────────────────────────

    def register_log_source(self, source: LogSource) -> None:
        """Register a custom log source for the error watcher.

        Example — add a CloudWatch source::

            sm.register_log_source(MyCloudWatchSource())

        Sources are polled in registration order on every watcher tick.
        """
        self._log_sources.append(source)
        log.info("Registered log source: %s", source.name)

    def _feedback_path(self, branch: str) -> str:
        """Return the path to an inspector feedback file for the given branch.

        Tries the canonical name first, then falls back to any *_feedback.md
        in the review dir (only one branch is in review at a time).
        """
        canonical = os.path.join(
            config.REVIEW_DIR,
            f"{branch.replace('/', '_')}_feedback.md",
        )
        if os.path.exists(canonical):
            return canonical
        matches = glob.glob(os.path.join(config.REVIEW_DIR, "*_feedback.md"))
        if len(matches) == 1:
            return matches[0]
        return canonical  # fall back to canonical (may not exist)

    def _recover_orphaned_specs(self):
        """On startup, rename any .in_progress specs back to .json so they re-enter the pipeline.

        Also cleans up stale worktrees and feedback files left from the
        previous run, so the conductor can start fresh without hitting
        worktree-already-checked-out errors.
        """
        pattern = os.path.join(config.BACKLOG_DIR, "*.json.in_progress")
        orphaned = glob.glob(pattern)
        for path in orphaned:
            original = path.removesuffix(".in_progress")
            # Read spec to derive branch name and working_dir for cleanup
            try:
                with open(path) as f:
                    spec_data = json.load(f)
            except (json.JSONDecodeError, OSError):
                spec_data = {}

            working_dir = spec_data.get("working_dir")
            if not working_dir:
                working_dir = os.path.join(config.DEVELOPMENT_DIR, config.DEFAULT_PROJECT)
            title = spec_data.get("title", "")

            # Clean up stale worktrees for all trains in this project
            if os.path.isdir(working_dir):
                worktree_base = os.path.join(working_dir, ".worktrees")
                if os.path.isdir(worktree_base):
                    for entry in os.listdir(worktree_base):
                        wt_path = os.path.join(worktree_base, entry)
                        if os.path.isdir(wt_path):
                            self._git("worktree", "remove", "--force", wt_path, cwd=working_dir)
                            # If git failed silently, force-remove the directory
                            if os.path.isdir(wt_path):
                                shutil.rmtree(wt_path, ignore_errors=True)
                            activity(f"CLEANUP stale worktree: {wt_path}")

            # Clean up stale feedback file for this branch
            if title:
                branch_name = f"feature/{title}"
                feedback_path = self._feedback_path(branch_name)
                if os.path.exists(feedback_path):
                    os.remove(feedback_path)
                    activity(f"CLEANUP stale feedback: {os.path.basename(feedback_path)}")

            os.rename(path, original)
            activity(f"RECOVERED orphaned spec: {os.path.basename(original)}")

    def _maybe_rotate_logs(self):
        """Rotate activity.log if it exceeds the size threshold; prune old agent log files hourly."""
        # Rotate activity.log: rename to .1, shift older rotations, keep at most 3
        if os.path.exists(config.ACTIVITY_LOG):
            try:
                if os.path.getsize(config.ACTIVITY_LOG) > config.LOG_MAX_SIZE_BYTES:
                    for i in range(2, 0, -1):
                        src = f"{config.ACTIVITY_LOG}.{i}"
                        dst = f"{config.ACTIVITY_LOG}.{i + 1}"
                        if os.path.exists(src):
                            try:
                                os.rename(src, dst)
                            except OSError:
                                pass
                    try:
                        os.rename(config.ACTIVITY_LOG, f"{config.ACTIVITY_LOG}.1")
                        log.info(
                            "Rotated activity.log (exceeded %dMB)",
                            config.LOG_MAX_SIZE_BYTES // (1024 * 1024),
                        )
                    except OSError:
                        pass
            except OSError:
                pass

        # Prune agent log files older than retention period (rate-limited to once per hour)
        now = time.time()
        if now - self._last_log_prune >= 3600:
            self._last_log_prune = now
            cutoff = now - (config.LOG_RETENTION_DAYS * 86400)
            pruned = 0
            for path in glob.glob(os.path.join(config.LOGS_DIR, "*.log")):
                try:
                    if os.path.getmtime(path) < cutoff:
                        os.remove(path)
                        pruned += 1
                except OSError:
                    pass
            if pruned:
                log.info("Pruned %d agent log files older than %d days", pruned, config.LOG_RETENTION_DAYS)

    def _backlog_specs(self, complexity: str | None = None) -> list[str]:
        """Return backlog specs sorted by priority (high first), then oldest first.

        If complexity is given, filter to specs matching that complexity.
        Specs without a complexity field default to "high".
        Specs with a corresponding .skip file are excluded.
        """
        PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}
        specs = [
            p for p in sorted(glob.glob(os.path.join(config.BACKLOG_DIR, "*.json")))
            if not os.path.exists(p + ".skip")
        ]
        if complexity is not None:
            filtered = []
            for path in specs:
                try:
                    with open(path) as f:
                        data = json.load(f)
                    spec_complexity = data.get("complexity", "high")
                    if spec_complexity == complexity:
                        filtered.append(path)
                except (json.JSONDecodeError, OSError):
                    if complexity == "high":
                        filtered.append(path)
            specs = filtered
        def sort_key(path: str) -> tuple[int, str]:
            try:
                with open(path) as f:
                    data = json.load(f)
                return (PRIORITY_ORDER.get(data.get("priority", "medium"), 1), path)
            except (json.JSONDecodeError, OSError):
                return (1, path)
        return sorted(specs, key=sort_key)

    def _signal_open_bugs(self) -> list[dict]:
        """Return spec dicts for open Signal-authored bugs (both .json and .json.in_progress)."""
        bugs = []
        for pattern in ("*.json", "*.json.in_progress"):
            for path in glob.glob(os.path.join(config.BACKLOG_DIR, pattern)):
                try:
                    with open(path) as f:
                        data = json.load(f)
                    if data.get("created_by") in ("sre", "signal"):
                        bugs.append(data)
                except (json.JSONDecodeError, OSError):
                    continue
        return bugs

    # ─── Per-tick caches ────────────────────────────────────────────────

    def _advance_tick(self):
        """Advance the tick counter, invalidating all per-tick caches."""
        self._tick_id += 1

    def _get_cached_backlog_count(self) -> int:
        """Return backlog spec count, cached for the current tick."""
        if self._cached_backlog_count_tick == self._tick_id:
            return self._cached_backlog_count
        count = len(glob.glob(os.path.join(config.BACKLOG_DIR, "*.json")))
        self._cached_backlog_count = count
        self._cached_backlog_count_tick = self._tick_id
        return count

    def _get_cached_backlog_specs(self, complexity: str | None = None) -> list[str]:
        """Return backlog specs, cached for the current tick and complexity."""
        if (self._cached_backlog_specs_tick == self._tick_id
                and self._cached_backlog_specs_complexity == complexity):
            return self._cached_backlog_specs
        specs = self._backlog_specs(complexity=complexity)
        self._cached_backlog_specs = specs
        self._cached_backlog_specs_tick = self._tick_id
        self._cached_backlog_specs_complexity = complexity
        return specs

    def _get_cached_open_bugs(self) -> list[dict]:
        """Return open Signal bugs, cached for the current tick."""
        if self._cached_open_bugs_tick == self._tick_id:
            return self._cached_open_bugs
        bugs = self._signal_open_bugs()
        self._cached_open_bugs = bugs
        self._cached_open_bugs_tick = self._tick_id
        return bugs

    def _is_agent_active(self, name: str) -> bool:
        agent = self.active_agents.get(name)
        if agent is None:
            return False
        if agent.poll():
            agent.save_log()
            rc = agent.proc.returncode if agent.proc else "?"
            summary = agent.get_output()[:200].replace("\n", " ").strip()
            activity(f"ARRIVED  [{agent.name}] rc={rc} — {summary or '(no output)'}")
            self.active_agents[name] = None
            # Set cooldown on non-zero exit with exponential backoff
            if rc != 0:
                METRICS.agent_failures_total.inc({"agent": name})
                # Detect API rate-limit responses and enter sleep mode to avoid
                # hammering a quota wall with repeated retries (seen as "out of extra usage")
                agent_output = agent.get_output() + agent.get_stderr()
                if "out of extra usage" in agent_output or "rate limit" in agent_output.lower():
                    self.sleep_until = time.time() + config.SLEEP_MODE_DURATION
                    activity(
                        f"RATE LIMIT [{name}] — API quota exhausted, "
                        f"entering SERVICE SUSPENDED for {config.SLEEP_MODE_DURATION}s"
                    )
                    return False
                self.consecutive_failures[name] = self.consecutive_failures.get(name, 0) + 1
                max_backoff = config.SIGNAL_MAX_BACKOFF if name == "signal" else config.MAX_ERROR_BACKOFF
                backoff = min(
                    config.AGENT_ERROR_COOLDOWN * (2 ** self.consecutive_failures[name]),
                    max_backoff,
                )
                cooldown_until = time.time() + backoff
                self.agent_cooldowns[name] = cooldown_until
                activity(f"DELAY [{name}] — failure #{self.consecutive_failures[name]}, retry after {backoff}s")
                # Signal failed — roll back log offsets so the same lines are retried next run
                if name == "signal":
                    self.sre_log_offsets.update(self._sre_prev_offsets)
            else:
                self.consecutive_failures.pop(name, None)
            self._sre_prev_offsets.clear()
            return False
        if agent.is_timed_out():
            self._kill_timed_out_agent(name, agent)
            return False
        return True

    def _kill_timed_out_agent(self, name: str, agent: AgentProcess):
        elapsed = time.time() - (agent.start_time or 0)
        activity(f"OVERDUE [{name}] after {elapsed:.0f}s — terminating")
        if agent.proc and agent.proc.poll() is None:
            try:
                os.killpg(agent.proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                agent.proc.terminate()
            try:
                agent.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(agent.proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    agent.proc.kill()
                agent.proc.wait()
        agent.save_log(marker="[OVERDUE]")
        self.active_agents[name] = None

        # Treat timeouts as failures for cooldown purposes
        self.consecutive_failures[name] = self.consecutive_failures.get(name, 0) + 1
        backoff = min(
            config.AGENT_ERROR_COOLDOWN * (2 ** self.consecutive_failures[name]),
            config.MAX_ERROR_BACKOFF,
        )
        self.agent_cooldowns[name] = time.time() + backoff
        activity(f"DELAY [{name}] — overdue #{self.consecutive_failures[name]}, retry after {backoff}s")

        # Signal timed out — roll back log offsets so those lines are retried next run
        if name == "signal":
            self.sre_log_offsets.update(self._sre_prev_offsets)
        self._sre_prev_offsets.clear()

    def _launch_agent(self, name: str, prompt: str, cwd: str | None = None) -> AgentProcess | None:
        # Error cooldown check
        now = time.time()
        if name in self.agent_cooldowns and now < self.agent_cooldowns[name]:
            remaining = int(self.agent_cooldowns[name] - now)
            log.info("Agent %s in cooldown (%ds remaining), skipping launch", name, remaining)
            return None

        # Clear expired cooldown
        self.agent_cooldowns.pop(name, None)

        # Minimum interval throttle
        min_interval = config.AGENT_MIN_INTERVALS.get(name, 0)
        if min_interval > 0:
            last_launch = self.last_launch_times.get(name, 0)
            elapsed = now - last_launch
            if elapsed < min_interval:
                remaining = int(min_interval - elapsed)
                log.info("Agent %s throttled (%ds until next allowed launch)", name, remaining)
                return None

        # Cost guardrail check
        self.launch_times.append(now)
        # Prune launches older than 1 hour
        while self.launch_times and self.launch_times[0] < now - 3600:
            self.launch_times.popleft()

        if len(self.launch_times) > config.MAX_AGENT_LAUNCHES_PER_HOUR:
            self.sleep_until = now + config.SLEEP_MODE_DURATION
            activity(
                f"FARE LIMIT — {len(self.launch_times)} launches in the last hour "
                f"(limit {config.MAX_AGENT_LAUNCHES_PER_HOUR}). "
                f"Entering SERVICE SUSPENDED until {time.ctime(self.sleep_until)}"
            )
            return None  # type: ignore[return-value]

        model = config.AGENT_MODELS.get(name, "claude-sonnet-4-5-20250929")
        agent = AgentProcess(name, prompt, cwd=cwd, model=model)
        agent.start()
        self.active_agents[name] = agent
        self.last_launch_times[name] = now
        METRICS.agent_launches_total.inc({"agent": name})
        activity(f"DEPARTED [{name}] PID {agent.proc.pid} model={model} cwd={cwd or 'default'}")
        return agent

    def _git(self, *args: str, cwd: str | None = None) -> str:
        """Run a git command and return stdout."""
        try:
            result = subprocess.run(
                ["git"] + list(args),
                capture_output=True, text=True,
                cwd=cwd or config.BASE_DIR,
                timeout=config.GIT_TIMEOUT,
            )
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            log.warning("git %s timed out after %ds (cwd=%s)", " ".join(args), config.GIT_TIMEOUT, cwd)
            return ""

    def _git_rc(self, *args: str, cwd: str | None = None) -> tuple[int, str, str]:
        """Run a git command and return (returncode, stdout, stderr)."""
        try:
            result = subprocess.run(
                ["git"] + list(args),
                capture_output=True, text=True,
                cwd=cwd or config.BASE_DIR,
                timeout=config.GIT_TIMEOUT,
            )
            return result.returncode, result.stdout.strip(), result.stderr.strip()
        except subprocess.TimeoutExpired:
            log.warning("git %s timed out after %ds (cwd=%s)", " ".join(args), config.GIT_TIMEOUT, cwd)
            return 1, "", "timeout"

    def _create_worktree(self, repo_dir: str, branch: str, train_id: str) -> str:
        """Create a git worktree for the given branch and return its path.

        Creates a new branch from HEAD if it doesn't exist, or checks out
        an existing branch. Adds .worktrees/ to the project's .gitignore.
        """
        worktree_base = os.path.join(repo_dir, ".worktrees")
        os.makedirs(worktree_base, exist_ok=True)
        worktree_path = os.path.join(worktree_base, train_id)

        # Ensure .worktrees/ is in the project's .gitignore
        gitignore_path = os.path.join(repo_dir, ".gitignore")
        ignore_entry = ".worktrees/"
        needs_add = True
        if os.path.exists(gitignore_path):
            with open(gitignore_path) as f:
                for line in f:
                    if line.strip() == ignore_entry:
                        needs_add = False
                        break
        if needs_add:
            with open(gitignore_path, "a") as f:
                f.write(f"\n{ignore_entry}\n")

        # Clean up stale worktree at this path if it exists
        if os.path.isdir(worktree_path):
            result = subprocess.run(
                ["git", "worktree", "remove", "--force", worktree_path],
                capture_output=True, text=True, cwd=repo_dir, timeout=config.GIT_TIMEOUT,
            )
            if result.returncode != 0 and os.path.isdir(worktree_path):
                shutil.rmtree(worktree_path, ignore_errors=True)

        # Create worktree: new branch if it doesn't exist, existing branch if it does
        if self._git_has_branch(branch, cwd=repo_dir):
            self._git("worktree", "add", worktree_path, branch, cwd=repo_dir)
        else:
            self._git("worktree", "add", "-b", branch, worktree_path, cwd=repo_dir)

        if not os.path.isdir(worktree_path):
            raise RuntimeError(
                f"Failed to create worktree at {worktree_path} for branch {branch}. "
                f"The branch may already be checked out in another worktree."
            )

        self._seen_project_dirs.add(repo_dir)
        activity(f"WORKTREE created: {worktree_path} (branch={branch})")
        return worktree_path

    def _remove_worktree(self, repo_dir: str | None, worktree_path: str | None):
        """Remove a git worktree. Safe to call even if already removed."""
        if not repo_dir or not worktree_path:
            return
        if os.path.isdir(worktree_path):
            result = subprocess.run(
                ["git", "worktree", "remove", "--force", worktree_path],
                capture_output=True, text=True, cwd=repo_dir, timeout=config.GIT_TIMEOUT,
            )
            if result.returncode != 0 and os.path.isdir(worktree_path):
                # Directory exists but isn't registered with git — remove it directly
                shutil.rmtree(worktree_path, ignore_errors=True)
                activity(f"WORKTREE force-removed (was not registered with git): {worktree_path}")
            else:
                activity(f"WORKTREE removed: {worktree_path}")
        # Prune any stale worktree references
        self._git("worktree", "prune", cwd=repo_dir)

    def _git_has_branch(self, branch: str, cwd: str | None = None) -> bool:
        result = self._git("branch", "--list", branch, cwd=cwd)
        return bool(result.strip())

    def _git_diff_trunk(self, branch: str, cwd: str | None = None) -> str:
        return self._git("diff", f"{config.TRUNK_BRANCH}..{branch}", cwd=cwd)

    def _git_last_commit(self, cwd: str | None = None) -> str:
        return self._git("rev-parse", "HEAD", cwd=cwd)

    def _find_app_log(self, project_dir: str) -> str | None:
        return find_app_log(project_dir)

    def _fetch_railway_logs(self, environment: str) -> str:
        project_dir = os.path.join(config.DEVELOPMENT_DIR, config.DEFAULT_PROJECT)
        return fetch_railway_logs(environment, project_dir)

    def _read_app_log_tail(self, project_dir: str, lines: int = 100) -> str:
        if config.RAILWAY_PROJECT:
            output = self._fetch_railway_logs(config.RAILWAY_PRODUCTION_ENV)
            if output:
                return "\n".join(output.splitlines()[-lines:])
            return ""

        log_path = self._find_app_log(project_dir)
        if not log_path:
            return ""
        result = subprocess.run(
            ["tail", "-n", str(lines), log_path],
            capture_output=True, text=True,
        )
        return result.stdout

    def _read_new_log_lines(self, project_dir: str) -> str:
        """Read only log lines written since the last Signal run (high-water mark)."""
        if config.RAILWAY_PROJECT:
            return self._read_new_railway_logs(project_dir)

        log_path = self._find_app_log(project_dir)
        if not log_path:
            return ""

        if project_dir not in self.sre_log_offsets:
            # First run (or after restart): set high-water mark to current EOF.
            # Don't re-analyze logs that were already seen before the restart.
            try:
                self.sre_log_offsets[project_dir] = os.path.getsize(log_path)
            except OSError:
                pass
            return ""

        stored_offset = self.sre_log_offsets[project_dir]
        try:
            file_size = os.path.getsize(log_path)
        except OSError:
            return ""

        # Log rotation: file shrank below stored offset → reset to start
        if file_size < stored_offset:
            stored_offset = 0

        if file_size == stored_offset:
            return ""  # No new content

        with open(log_path, "r") as f:
            f.seek(stored_offset)
            new_content = f.read()

        self._sre_prev_offsets[project_dir] = stored_offset
        self.sre_log_offsets[project_dir] = file_size
        return new_content

    def _read_new_railway_logs(self, project_dir: str) -> str:
        """Fetch Railway production logs and return only lines not seen before."""
        key = "_railway_"
        output = self._fetch_railway_logs(config.RAILWAY_PRODUCTION_ENV)
        if not output:
            return ""

        lines = output.splitlines()
        if not lines:
            return ""

        if key not in self.sre_log_offsets:
            # First run: set high-water mark, return empty (same semantics as local mode)
            self._sre_prev_offsets[key] = None
            self.sre_log_offsets[key] = lines[-1]
            return ""

        last_seen = self.sre_log_offsets[key]
        # Find where the last-seen line is in the new output
        try:
            idx = lines.index(last_seen)
            new_lines = lines[idx + 1:]
        except ValueError:
            # Last-seen line not found (log rotated or too much new output) — return all
            new_lines = lines

        if not new_lines:
            return ""

        self._sre_prev_offsets[key] = last_seen
        self.sre_log_offsets[key] = new_lines[-1]
        return "\n".join(new_lines)

    def _gather_ops_context(self) -> tuple[str, str]:
        """Collect diagnostic data for the ops agent."""
        activity_tail = ""
        if os.path.exists(config.ACTIVITY_LOG):
            result = subprocess.run(
                ["tail", "-n", "100", config.ACTIVITY_LOG],
                capture_output=True, text=True,
            )
            activity_tail = result.stdout
        git_log = self._git("log", "--oneline", "-10", cwd=config.BASE_DIR)
        return activity_tail, git_log

    def _gather_work_balance_digest(self, project_dir: str, window: int = 20) -> str:
        """Summarize recent merged spec types from the activity log for the dispatcher."""
        try:
            with open(config.ACTIVITY_LOG) as f:
                lines = f.readlines()
        except (OSError, FileNotFoundError):
            return "(no activity history)"

        terminus_re = re.compile(r"TERMINUS.*branch feature/([^\s]+)\s+approved")
        merged_titles = []
        for line in lines:
            if "TERMINUS" not in line:
                continue
            m = terminus_re.search(line)
            if m:
                merged_titles.append(m.group(1))

        recent = merged_titles[-window:] if len(merged_titles) > window else merged_titles
        if not recent:
            return "(no merged specs in activity history)"

        counts: dict[str, int] = {"feature": 0, "bugfix": 0, "hardening": 0, "refactor": 0}
        for title in recent:
            spec_type = _classify_spec_type(title)
            counts[spec_type] = counts.get(spec_type, 0) + 1

        total = len(recent)
        summary = f"Recent merged work ({total} specs): " + ", ".join(
            f"{v} {k}" for k, v in counts.items() if v > 0
        ) + "."

        feature_pct = counts["feature"] / total
        hardening_pct = (counts["hardening"] + counts["bugfix"]) / total
        refactor_pct = counts["refactor"] / total

        if feature_pct >= 0.7:
            signal = "FEATURE-HEAVY — consider a bug fix or hardening spec if the product has real issues to address."
        elif hardening_pct >= 0.6:
            signal = "HARDENING-HEAVY — consider a meaningful new feature."
        elif refactor_pct >= 0.4:
            signal = "REFACTOR-HEAVY — consider new user-facing functionality."
        else:
            signal = "BALANCED — choose based on what the product needs most right now."

        return f"{summary}\nBalance signal: {signal}"

    def _on_triage_rejection(self, train: Train) -> None:
        """Increment the per-project consecutive rejection counter; trigger stall if limit hit."""
        project = train.repo_dir or train.working_dir or ""
        if not project:
            return
        count = self._project_rejection_counts.get(project, 0) + 1
        self._project_rejection_counts[project] = count
        if count >= config.MAX_CONSECUTIVE_REJECTIONS:
            resume_at = time.time() + config.STALL_PAUSE_SECONDS
            self._stalled_projects[project] = resume_at
            self._project_rejection_counts[project] = 0
            hours = config.STALL_PAUSE_SECONDS // 3600
            activity(
                f"STALLED [{os.path.basename(project)}] — {count} consecutive triage rejections, "
                f"dispatcher paused for {hours}h"
            )

    def _recycle_stale_drafts(self) -> None:
        """Move HOLD specs older than DRAFTS_RECYCLE_AGE_SECONDS back to backlog."""
        now = time.time()
        if now - self._last_draft_recycle < config.DRAFTS_RECYCLE_AGE_SECONDS:
            return
        self._last_draft_recycle = now
        if not os.path.isdir(config.DRAFTS_DIR):
            return
        os.makedirs(config.BACKLOG_DIR, exist_ok=True)
        for fname in os.listdir(config.DRAFTS_DIR):
            if not fname.endswith(".json"):
                continue
            src = os.path.join(config.DRAFTS_DIR, fname)
            try:
                age = now - os.path.getmtime(src)
            except OSError:
                continue
            if age < config.DRAFTS_RECYCLE_AGE_SECONDS:
                continue
            dest = os.path.join(config.BACKLOG_DIR, fname)
            try:
                shutil.move(src, dest)
                activity(f"RECYCLED draft spec back to backlog: {fname}")
            except OSError as e:
                log.warning("Failed to recycle draft spec %s: %s", fname, e)

    def _gc_orphaned_worktrees(self) -> None:
        """Remove worktrees in known project dirs that no active train owns."""
        now = time.time()
        if now - self._last_worktree_gc < config.WORKTREE_GC_INTERVAL:
            return
        self._last_worktree_gc = now
        active_paths = {
            train.working_dir
            for train in self.trains
            if train.working_dir and train.branch
        }
        for project_dir in list(self._seen_project_dirs):
            worktree_base = os.path.join(project_dir, ".worktrees")
            if not os.path.isdir(worktree_base):
                continue
            try:
                entries = os.listdir(worktree_base)
            except OSError:
                continue
            for entry in entries:
                wt_path = os.path.join(worktree_base, entry)
                if not os.path.isdir(wt_path):
                    continue
                if wt_path in active_paths:
                    continue
                self._git("worktree", "remove", "--force", wt_path, cwd=project_dir)
                if os.path.isdir(wt_path):
                    shutil.rmtree(wt_path, ignore_errors=True)
                activity(f"GC stale worktree: {wt_path}")

    def _request_self_restart(self):
        """Gracefully terminate all agents and exit for systemd to restart."""
        activity("OPS RESTART — new commits detected, restarting orchestrator...")
        # Terminate global agents
        for name, agent in self.active_agents.items():
            if agent and agent.proc and agent.proc.poll() is None:
                activity(f"Terminating {name} (PID {agent.proc.pid})")
                try:
                    os.killpg(agent.proc.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError, OSError):
                    agent.proc.terminate()
                try:
                    agent.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(agent.proc.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError, OSError):
                        agent.proc.kill()
                agent.save_log()
        # Terminate per-train agents
        for train in self.trains:
            for role, agent in [("conductor", train.conductor), ("inspector", train.inspector)]:
                if agent and agent.proc and agent.proc.poll() is None:
                    activity(f"Terminating {role}:{train.train_id} (PID {agent.proc.pid})")
                    try:
                        os.killpg(agent.proc.pid, signal.SIGTERM)
                    except (ProcessLookupError, PermissionError, OSError):
                        agent.proc.terminate()
                    try:
                        agent.proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(agent.proc.pid, signal.SIGKILL)
                        except (ProcessLookupError, PermissionError, OSError):
                            agent.proc.kill()
                    agent.save_log()
        activity("All agents stopped. Exiting for restart.")
        sys.exit(0)

    def _is_self_project(self, working_dir: str | None) -> bool:
        """Return True if the spec targets the Yamanote orchestrator itself."""
        if not working_dir:
            return False
        return os.path.realpath(working_dir) == os.path.realpath(config.SELF_PROJECT_DIR)

    # ─── Per-train agent helpers ─────────────────────────────────────────

    def _is_train_agent_active(self, train: Train, role: str) -> bool:
        """Check if a train's conductor or inspector is still running. Handle completion."""
        agent = train.conductor if role == "conductor" else train.inspector
        if agent is None:
            return False
        if agent.poll():
            agent.save_log()
            rc = agent.proc.returncode if agent.proc else "?"
            summary = agent.get_output()[:200].replace("\n", " ").strip()
            activity(f"ARRIVED  [{role}:{train.train_id}] rc={rc} — {summary or '(no output)'}")
            if role == "conductor":
                train.conductor = None
            else:
                train.inspector = None
            if rc != 0:
                METRICS.agent_failures_total.inc({"agent": f"{role}:{train.train_id}"})
                agent_output = agent.get_output() + agent.get_stderr()
                if "out of extra usage" in agent_output or "rate limit" in agent_output.lower():
                    self.sleep_until = time.time() + config.SLEEP_MODE_DURATION
                    activity(
                        f"RATE LIMIT [{role}:{train.train_id}] — API quota exhausted, "
                        f"entering SERVICE SUSPENDED for {config.SLEEP_MODE_DURATION}s"
                    )
                    return False
                if role == "conductor":
                    train.conductor_failures += 1
                    backoff = min(config.AGENT_ERROR_COOLDOWN * (2 ** train.conductor_failures), config.MAX_ERROR_BACKOFF)
                    train.conductor_cooldown_until = time.time() + backoff
                    activity(f"DELAY [{role}:{train.train_id}] — failure #{train.conductor_failures}, retry after {backoff}s")
                else:
                    train.inspector_failures += 1
                    backoff = min(config.AGENT_ERROR_COOLDOWN * (2 ** train.inspector_failures), config.MAX_ERROR_BACKOFF)
                    train.inspector_cooldown_until = time.time() + backoff
                    activity(f"DELAY [{role}:{train.train_id}] — failure #{train.inspector_failures}, retry after {backoff}s")
            else:
                if role == "conductor":
                    train.conductor_failures = 0
                else:
                    train.inspector_failures = 0
            return False
        if agent.is_timed_out():
            self._kill_timed_out_train_agent(train, role, agent)
            return False
        return True

    def _launch_train_agent(self, train: Train, role: str, prompt: str, cwd: str | None = None) -> AgentProcess | None:
        """Launch a conductor or inspector for a specific train. Uses shared cost guardrail."""
        now = time.time()
        if role == "conductor" and now < train.conductor_cooldown_until:
            remaining = int(train.conductor_cooldown_until - now)
            log.info("Train %s conductor in cooldown (%ds remaining), skipping", train.train_id, remaining)
            return None
        if role == "inspector" and now < train.inspector_cooldown_until:
            remaining = int(train.inspector_cooldown_until - now)
            log.info("Train %s inspector in cooldown (%ds remaining), skipping", train.train_id, remaining)
            return None

        # Shared cost guardrail
        self.launch_times.append(now)
        while self.launch_times and self.launch_times[0] < now - 3600:
            self.launch_times.popleft()
        if len(self.launch_times) > config.MAX_AGENT_LAUNCHES_PER_HOUR:
            self.sleep_until = now + config.SLEEP_MODE_DURATION
            activity(
                f"FARE LIMIT — {len(self.launch_times)} launches in the last hour "
                f"(limit {config.MAX_AGENT_LAUNCHES_PER_HOUR}). "
                f"Entering SERVICE SUSPENDED until {time.ctime(self.sleep_until)}"
            )
            return None

        model = train.conductor_model if role == "conductor" else train.inspector_model
        agent_name = f"{role}:{train.train_id}"
        agent = AgentProcess(agent_name, prompt, cwd=cwd, model=model)
        agent.start()
        if role == "conductor":
            train.conductor = agent
        else:
            train.inspector = agent
        METRICS.agent_launches_total.inc({"agent": agent_name})
        activity(f"DEPARTED [{agent_name}] PID {agent.proc.pid} model={model} cwd={cwd or 'default'}")
        return agent

    def _kill_timed_out_train_agent(self, train: Train, role: str, agent: AgentProcess):
        """Handle timeout for a train's conductor or inspector."""
        elapsed = time.time() - (agent.start_time or 0)
        agent_name = f"{role}:{train.train_id}"
        activity(f"OVERDUE [{agent_name}] after {elapsed:.0f}s — terminating")
        if agent.proc and agent.proc.poll() is None:
            try:
                os.killpg(agent.proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError, OSError):
                agent.proc.terminate()
            try:
                agent.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(agent.proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    agent.proc.kill()
                agent.proc.wait()
        agent.save_log(marker="[OVERDUE]")

        if role == "conductor":
            train.conductor = None
            train.conductor_failures += 1
            backoff = min(config.AGENT_ERROR_COOLDOWN * (2 ** train.conductor_failures), config.MAX_ERROR_BACKOFF)
            train.conductor_cooldown_until = time.time() + backoff
            activity(f"DELAY [{agent_name}] — overdue #{train.conductor_failures}, retry after {backoff}s")
        else:
            train.inspector = None
            train.inspector_failures += 1
            backoff = min(config.AGENT_ERROR_COOLDOWN * (2 ** train.inspector_failures), config.MAX_ERROR_BACKOFF)
            train.inspector_cooldown_until = time.time() + backoff
            activity(f"DELAY [{agent_name}] — overdue #{train.inspector_failures}, retry after {backoff}s")

        if role == "conductor" and train.spec_path:
            train.spec_timeout_count += 1
            if train.spec_timeout_count >= config.MAX_SPEC_TIMEOUTS:
                activity(f"TERMINATED spec after {train.spec_timeout_count} overdue: {os.path.basename(train.spec_path)}")
                train.conductor_failures = 0
                train.conductor_cooldown_until = 0.0
                in_progress = train.spec_path + ".in_progress"
                if os.path.exists(in_progress):
                    os.remove(in_progress)
                repo = train.repo_dir
                branch = train.branch
                self._remove_worktree(train.repo_dir, train.working_dir)
                if repo and branch and self._git_has_branch(branch, cwd=repo):
                    self._git("branch", "-D", branch, cwd=repo)
                if branch:
                    fb = self._feedback_path(branch)
                    if os.path.exists(fb):
                        os.remove(fb)
                train.reset_pipeline()
            else:
                in_progress = train.spec_path + ".in_progress"
                if os.path.exists(in_progress):
                    original = train.spec_path
                    os.rename(in_progress, original)
                    activity(f"RE-ROUTED spec after Conductor overdue ({train.spec_timeout_count}/{config.MAX_SPEC_TIMEOUTS}): {os.path.basename(original)}")
                self._remove_worktree(train.repo_dir, train.working_dir)
                # Clean up the feature branch so retry starts fresh (prevents orphan recovery confusion)
                if train.branch and train.repo_dir and self._git_has_branch(train.branch, cwd=train.repo_dir):
                    self._git("branch", "-D", train.branch, cwd=train.repo_dir)
                train.branch = None
                train.spec_path = None
                train.working_dir = None
                train.repo_dir = None
                train.file_edits.clear()

    def _find_spec_for_train(self, train: Train) -> str | None:
        """Find a suitable spec for this train based on complexity.

        Regular trains:  high → medium → low
        Standard trains: medium → low
        Express trains:  low only
        """
        # With worktrees, multiple trains can work on the same project.
        # Collision is prevented by spec rename to .in_progress on pickup.

        # Primary complexity
        primary = self._backlog_specs(complexity=train.complexity)
        if primary:
            return primary[0]

        # Fallback chain: higher-capability trains can pick simpler specs
        if train.train_type == "regular":
            for fallback_complexity in ("medium", "low"):
                fallback = self._backlog_specs(complexity=fallback_complexity)
                if fallback:
                    return fallback[0]
        elif train.train_type == "standard":
            fallback = self._backlog_specs(complexity="low")
            if fallback:
                return fallback[0]

        return None

    # ─── Entropy check ───────────────────────────────────────────────────

    def _count_fix_commits_on_branch(self, branch: str, cwd: str | None = None) -> int:
        """Count commits on branch (not on trunk) whose message contains 'fix' or 'update'."""
        log_output = self._git(
            "log", f"{config.TRUNK_BRANCH}..{branch}",
            "--oneline", cwd=cwd,
        )
        if not log_output:
            return 0
        count = 0
        for line in log_output.splitlines():
            lower = line.lower()
            if "fix" in lower or "update" in lower:
                count += 1
        return count

    def _fire_conductor_entropy(self, train: Train, branch: str, cwd: str | None = None):
        """Fire the Conductor agent — nuke the branch and terminate the spec."""
        activity(
            f"DERAILED [conductor:{train.train_id}] — branch {branch} has too many fix/update commits. "
            f"Terminating spec (entropy threshold reached)."
        )
        # Kill conductor if still running
        conductor = train.conductor
        if conductor and conductor.proc and conductor.proc.poll() is None:
            conductor.proc.terminate()
            try:
                conductor.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                conductor.proc.kill()
                conductor.proc.wait()
            conductor.save_log(marker="[DERAILED — ENTROPY]")
        train.conductor = None

        # Delete any stale inspector feedback for this branch
        fb = self._feedback_path(branch)
        if os.path.exists(fb):
            os.remove(fb)

        # Remove worktree and delete branch from main repo
        self._remove_worktree(train.repo_dir, train.working_dir)
        repo = train.repo_dir or cwd
        if self._git_has_branch(branch, cwd=repo):
            self._git("branch", "-D", branch, cwd=repo)

        # Terminate spec (don't re-queue after entropy)
        if train.spec_path:
            in_progress = train.spec_path + ".in_progress"
            if os.path.exists(in_progress):
                os.remove(in_progress)
                activity(f"TERMINATED spec after entropy: {os.path.basename(train.spec_path)}")
                METRICS.specs_total.inc({"outcome": "entropy"})

        train.reset_pipeline()

    # ─── Phases ──────────────────────────────────────────────────────────

    def _pick_dispatcher_project(self) -> str | None:
        """Select which project the dispatcher should generate a spec for.

        If projects.json exists, scheduled projects in their active window get
        top priority, then unscheduled projects by priority. Falls back to
        DEFAULT_PROJECT when no projects.json is present.
        """
        projects = config.load_projects()
        if not projects:
            default_dir = os.path.join(config.DEVELOPMENT_DIR, config.DEFAULT_PROJECT)
            if os.path.isdir(default_dir):
                return default_dir
            if os.path.isdir(config.DEVELOPMENT_DIR):
                return config.DEVELOPMENT_DIR
            return None

        now_hour = time.localtime().tm_hour
        candidates = []
        for name, proj in projects.items():
            if proj.get("paused"):
                continue
            path = os.path.expanduser(proj.get("path", ""))
            if not os.path.isdir(path):
                continue
            priority = proj.get("priority", 999)
            schedule = proj.get("schedule")
            in_window = _is_in_schedule_window(schedule, now_hour)
            candidates.append((priority, name, path, schedule, in_window))

        if not candidates:
            return None

        scheduled_active = [(p, n, path) for p, n, path, sched, active in candidates if sched and active]
        if scheduled_active:
            scheduled_active.sort(key=lambda x: x[0])
            return scheduled_active[0][2]

        unscheduled = [(p, n, path) for p, n, path, sched, active in candidates if not sched]
        if unscheduled:
            unscheduled.sort(key=lambda x: x[0])
            return unscheduled[0][2]

        return None

    def _phase_dispatcher(self):
        """If backlog is empty and no Dispatcher running, launch Dispatcher agent."""
        if self._is_agent_active("dispatcher"):
            return
        if "dispatcher" in self.agent_cooldowns and time.time() < self.agent_cooldowns["dispatcher"]:
            return
        if self._backlog_specs():
            return

        project_dir = self._pick_dispatcher_project()
        if not project_dir:
            return

        # Stall check: too many consecutive rejections → pause this project
        now_stall = time.time()
        if project_dir in self._stalled_projects:
            if now_stall < self._stalled_projects[project_dir]:
                return
            del self._stalled_projects[project_dir]
            activity(f"UNSTALLED [{os.path.basename(project_dir)}] — stall period expired, dispatcher resuming")

        active_trains = [t for t in self.trains if t.branch]
        if active_trains:
            train_ids = ", ".join(t.train_id for t in active_trains)
            key = frozenset(t.train_id for t in active_trains)
            if key != self._dispatcher_skip_logged_trains:
                activity(f"Dispatcher — skipped, pipeline active on {train_ids}")
                self._dispatcher_skip_logged_trains = key
            return

        now = time.time()
        idle_types = set(t.train_type for t in self.trains if not t.branch)
        if idle_types:
            min_interval = min(
                config.TRAIN_CONFIG[tt]["dispatcher_interval"]
                for tt in idle_types
            )
        else:
            min_interval = config.AGENT_MIN_INTERVALS.get("dispatcher", 900)
        last_launch = self.last_launch_times.get("dispatcher", 0)
        if now - last_launch < min_interval:
            return

        ts = time.strftime("%Y%m%d_%H%M%S")
        app_logs = self._read_app_log_tail(project_dir) or "(no app.log found)"
        rejected_specs = read_rejection_log(max_lines=20, project=project_dir, max_age_days=30)
        work_balance_digest = self._gather_work_balance_digest(project_dir)
        prompt = config.DISPATCHER_PROMPT.format(
            timestamp=ts,
            working_dir=project_dir,
            backlog_dir=config.BACKLOG_DIR,
            app_logs=app_logs,
            rejected_specs=rejected_specs,
            work_balance_digest=work_balance_digest,
        )
        agent = self._launch_agent("dispatcher", prompt, cwd=project_dir)
        if agent is not None:
            activity(f"Dispatcher — backlog empty, generating spec for {project_dir}")

    def _train_phase_conductor(self, train: Train):
        """If backlog has specs and no Conductor running on this train, pick a spec and launch."""
        if self._is_train_agent_active(train, "conductor"):
            return
        if time.time() < train.conductor_cooldown_until:
            return

        # Track file edits once after Conductor finishes (not every tick)
        if not train.edits_tallied and train.conductor is None and train.branch:
            cwd = train.working_dir
            if self._git_has_branch(train.branch, cwd=cwd):
                diff_stat = self._git(
                    "diff", "--name-only",
                    f"{config.TRUNK_BRANCH}..{train.branch}",
                    cwd=cwd,
                )
                for fname in diff_stat.splitlines():
                    fname = fname.strip()
                    if fname:
                        train.file_edits[fname] = train.file_edits.get(fname, 0) + 1
            train.edits_tallied = True

        # Don't pick up a new spec while a branch is still in the review pipeline
        if train.branch:
            return

        spec_path = self._find_spec_for_train(train)
        if not spec_path:
            return

        try:
            with open(spec_path) as f:
                spec_data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Bad spec file %s: %s", spec_path, e)
            return

        # Read working_dir from spec, default to configured project
        working_dir = spec_data.get("working_dir")
        if not working_dir:
            working_dir = os.path.join(config.DEVELOPMENT_DIR, config.DEFAULT_PROJECT)

        # "Don't reinvent ourselves" guardrail
        if self._is_self_project(working_dir):
            activity(f"RESTRICTED spec {os.path.basename(spec_path)} — targets Yamanote itself. Removing.")
            os.remove(spec_path)
            return

        # Validate working_dir exists and is under Development dir
        if not os.path.isdir(working_dir):
            activity(f"RESTRICTED spec {os.path.basename(spec_path)} — working_dir {working_dir} does not exist. Removing.")
            os.remove(spec_path)
            return
        if not os.path.realpath(working_dir).startswith(os.path.realpath(config.DEVELOPMENT_DIR)):
            activity(f"RESTRICTED spec {os.path.basename(spec_path)} — working_dir outside {config.DEVELOPMENT_DIR}. Removing.")
            os.remove(spec_path)
            return

        spec_title = spec_data.get("title", "untitled")
        # Sanitize title into a valid git branch name
        safe_title = re.sub(r'[^a-zA-Z0-9_/-]', '-', spec_title).strip('-')
        safe_title = re.sub(r'-{2,}', '-', safe_title)
        branch_name = f"feature/{safe_title}"
        train.file_edits.clear()
        train.spec_path = spec_path
        train.spec_started_at = time.time()
        train.branch = branch_name
        train.repo_dir = working_dir  # original project repo
        train.rework_count = 0
        train.spec_timeout_count = 0

        # Rename spec to .in_progress before triage — this is the atomic claim step.
        # If two trains see the same spec, only one rename succeeds; the other aborts here.
        try:
            os.rename(spec_path, spec_path + ".in_progress")
        except OSError as e:
            log.warning("Failed to claim spec %s (%s) — aborting pickup", spec_path, e)
            train.reset_pipeline()
            return

        # Route to triage gate before building
        train.needs_triage = True
        return

    def _train_phase_triage(self, train: Train):
        """Triage gate: evaluate whether a spec is worth building before launching Conductor."""
        if not train.needs_triage:
            return
        if not train.spec_path or not train.branch:
            return
        if time.time() < train.triage_cooldown_until:
            return

        # Check if triage agent is still running
        if train.triage is not None:
            if train.triage.poll():
                # Triage finished — process verdict
                train.triage.save_log()
                output = train.triage.get_output()
                rc = train.triage.proc.returncode if train.triage.proc else "?"
                train.triage = None

                if rc != 0:
                    # Triage agent failed — let spec through (fail-open) but with backoff
                    activity(f"Triage:{train.train_id} — agent failed (rc={rc}), approving spec by default")
                    train.needs_triage = False
                    train.triage_failures += 1
                    backoff = min(config.AGENT_ERROR_COOLDOWN * (2 ** train.triage_failures), config.MAX_ERROR_BACKOFF)
                    train.triage_cooldown_until = time.time() + backoff
                    return

                # Parse verdict and reason from triage output
                first_line = output.strip().split("\n")[0].strip() if output else ""
                # Strip markdown formatting (**text**, *text*, etc.)
                first_line_clean = re.sub(r'[\*_`#]+', '', first_line)

                # Check if verdict and reason are on same line (e.g., "REJECT — reason here")
                if ' — ' in first_line_clean or ' - ' in first_line_clean:
                    separator = ' — ' if ' — ' in first_line_clean else ' - '
                    parts = first_line_clean.split(separator, 1)
                    verdict = parts[0].strip().upper()
                    reason = parts[1].strip() if len(parts) > 1 else ""
                else:
                    verdict = first_line_clean.upper()
                    # Try multi-line format (reason on subsequent lines)
                    reason = output.strip().split("\n", 1)[1].strip() if output and "\n" in output.strip() else ""

                # Fallback 1: search for "Verdict: BUILD" / "## Verdict: REJECT" etc.
                if not any(verdict.startswith(v) for v in ["BUILD", "REJECT", "HOLD"]):
                    verdict_match = re.search(r'##?\s*Verdict:?\s*(\w+)', output, re.MULTILINE | re.IGNORECASE)
                    if verdict_match:
                        verdict = verdict_match.group(1).upper()
                        verdict_pos = verdict_match.end()
                        remaining = output[verdict_pos:].strip()
                        if remaining and not remaining.startswith('\n'):
                            reason = remaining.split('\n')[0].strip()
                            if reason.startswith(('—', '-', ':')):
                                reason = reason[1:].strip()
                        if not reason:
                            reason = remaining

                # Fallback 2: scan every line for a standalone BUILD/REJECT/HOLD keyword.
                # Handles responses that open with analysis sections (e.g. "## ANALYSIS ...
                # ... BUILD — reason") where neither the first line nor a Verdict: header match.
                if not any(verdict.startswith(v) for v in ["BUILD", "REJECT", "HOLD"]):
                    for i, line in enumerate(output.split("\n")):
                        clean = re.sub(r'[\*_`#]+', '', line).strip()
                        m = re.match(r'^(BUILD|REJECT|HOLD)\b[\s:\-—]*(.*)?', clean, re.IGNORECASE)
                        if m:
                            verdict = m.group(1).upper()
                            rest = (m.group(2) or "").strip()
                            if not rest:
                                rest = "\n".join(output.split("\n")[i + 1:]).strip()
                            reason = rest
                            break

                if verdict.startswith("BUILD"):
                    train.needs_triage = False
                    train.triage_failures = 0  # Clear failures on successful triage
                    # Reset consecutive rejection counter — something finally passed
                    project = train.repo_dir or train.working_dir or ""
                    if project:
                        self._project_rejection_counts.pop(project, None)
                elif verdict.startswith("REJECT"):
                    spec_title = train.branch.removeprefix("feature/")
                    reason_clean = _strip_markdown(_strip_reason_prefix(reason))
                    reason_display = _first_line_truncated(reason_clean, limit=150)
                    activity(f"Triage:{train.train_id} — REJECTED spec '{spec_title}': {reason_display}")
                    project = train.repo_dir or train.working_dir or ""
                    record_rejection(spec_title, reason[:300], project=project)
                    METRICS.specs_total.inc({"outcome": "rejected"})
                    if train.spec_path and os.path.exists(train.spec_path):
                        os.remove(train.spec_path)
                    # Also remove .in_progress variant
                    in_progress = train.spec_path + ".in_progress"
                    if os.path.exists(in_progress):
                        os.remove(in_progress)
                    self._on_triage_rejection(train)
                    train.reset_pipeline()
                elif verdict.startswith("HOLD"):
                    spec_title = train.branch.removeprefix("feature/")
                    reason_clean = _strip_markdown(_strip_reason_prefix(reason))
                    reason_display = _first_line_truncated(reason_clean, limit=150)
                    activity(f"Triage:{train.train_id} — HOLD spec '{spec_title}': {reason_display}")
                    # Move spec to drafts
                    src = train.spec_path + ".in_progress"
                    if not os.path.exists(src):
                        src = train.spec_path
                    if os.path.exists(src):
                        dest = os.path.join(config.DRAFTS_DIR, os.path.basename(train.spec_path).removesuffix(".in_progress") + ".json")
                        os.makedirs(config.DRAFTS_DIR, exist_ok=True)
                        shutil.move(src, dest)
                    train.reset_pipeline()
                else:
                    # Unrecognized verdict — fail-secure, reject to force investigation
                    spec_title = train.branch.removeprefix("feature/")
                    reason_msg = f"Triage returned unrecognized verdict: {verdict[:50]}"
                    activity(f"Triage:{train.train_id} — REJECTED spec '{spec_title}': triage verdict parse error")
                    project = train.repo_dir or train.working_dir or ""
                    record_rejection(spec_title, reason_msg, project=project)
                    METRICS.specs_total.inc({"outcome": "rejected"})
                    if train.spec_path and os.path.exists(train.spec_path):
                        os.remove(train.spec_path)
                    in_progress = train.spec_path + ".in_progress"
                    if os.path.exists(in_progress):
                        os.remove(in_progress)
                    self._on_triage_rejection(train)
                    train.reset_pipeline()
            elif train.triage.is_timed_out():
                # Triage timed out — fail-open but with exponential backoff
                elapsed = time.time() - (train.triage.start_time or 0)
                activity(f"OVERDUE [triage:{train.train_id}] after {elapsed:.0f}s — approving spec by default")
                if train.triage.proc and train.triage.proc.poll() is None:
                    try:
                        os.killpg(train.triage.proc.pid, signal.SIGTERM)
                    except (ProcessLookupError, PermissionError, OSError):
                        train.triage.proc.terminate()
                train.triage = None
                train.needs_triage = False
                train.triage_failures += 1
                backoff = min(config.AGENT_ERROR_COOLDOWN * (2 ** train.triage_failures), config.MAX_ERROR_BACKOFF)
                train.triage_cooldown_until = time.time() + backoff
            return

        # No triage agent running yet — launch one
        in_progress_path = train.spec_path + ".in_progress"
        spec_read_path = in_progress_path if os.path.exists(in_progress_path) else train.spec_path
        try:
            with open(spec_read_path) as f:
                spec_data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Cannot read spec for triage %s: %s", spec_read_path, e)
            train.needs_triage = False  # fail-open
            return

        working_dir = spec_data.get("working_dir", "")
        working_dir_exists = os.path.isdir(working_dir)
        recent_merges = self._git("log", "--oneline", "-15", cwd=working_dir) if working_dir_exists else "(none)"
        rejected_specs = read_rejection_log(max_lines=10, max_age_days=30)
        failed_specs = read_failure_log()

        prompt = config.TRIAGE_PROMPT.format(
            working_dir=working_dir,
            spec_json=json.dumps(spec_data, indent=2),
            rejected_specs=rejected_specs,
            failed_specs=failed_specs,
            recent_merges=recent_merges,
        )
        agent = self._launch_agent("triage", prompt, cwd=working_dir if working_dir_exists else None)
        if agent is not None:
            train.triage = agent
            # Remove from active_agents since we track it on the train
            self.active_agents["triage"] = None
        else:
            # Launch failed — fail-open
            train.needs_triage = False

    def _train_phase_conductor_launch(self, train: Train):
        """Launch conductor for a triaged spec. Split from _train_phase_conductor for triage gate."""
        if train.needs_triage:
            return
        if not train.branch or not train.spec_path:
            return
        if train.conductor is not None or train.inspector is not None:
            return
        # Don't launch if worktree already exists (already in review pipeline)
        if train.working_dir and train.working_dir != train.repo_dir:
            return

        spec_path = train.spec_path
        branch_name = train.branch
        working_dir = train.repo_dir

        if not working_dir:
            return

        # Re-read spec data
        in_progress_path = spec_path + ".in_progress"
        spec_read_path = in_progress_path if os.path.exists(in_progress_path) else spec_path
        try:
            with open(spec_read_path) as f:
                spec_data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Cannot read spec for conductor launch %s: %s", spec_read_path, e)
            train.reset_pipeline()
            return

        spec_title = spec_data.get("title", "untitled")

        # If the feature branch already exists with changes, skip Conductor → inspector
        if self._git_has_branch(branch_name, cwd=working_dir) and self._git_diff_trunk(branch_name, cwd=working_dir):
            activity(f"Conductor:{train.train_id} — branch {branch_name} already has changes, routing to inspector (orphan recovery)")
            try:
                worktree_path = self._create_worktree(working_dir, branch_name, train.train_id)
            except RuntimeError as e:
                activity(f"RESTRICTED [{train.train_id}] — worktree creation failed: {e}")
                train.reset_pipeline()
                return
            train.working_dir = worktree_path
            return

        # Create worktree with the feature branch
        try:
            worktree_path = self._create_worktree(working_dir, branch_name, train.train_id)
        except RuntimeError as e:
            activity(f"RESTRICTED [{train.train_id}] — worktree creation failed: {e}")
            train.reset_pipeline()
            return
        train.working_dir = worktree_path

        spec_desc = spec_data.get("description", "")
        activity(f"Conductor:{train.train_id} — starting spec '{spec_title}' in {worktree_path}")
        spec_summary = spec_desc.split("\n")[0][:120].strip()
        activity(f"  SPEC: {spec_summary}")
        prompt = config.CONDUCTOR_PROMPT.format(
            spec_json=json.dumps(spec_data, indent=2),
            spec_title=spec_title,
            working_dir=worktree_path,
            repo_dir=train.repo_dir or worktree_path,
            branch_name=branch_name,
        )
        agent = self._launch_train_agent(train, "conductor", prompt, cwd=worktree_path)
        if agent is None:
            self._remove_worktree(train.repo_dir, worktree_path)
            train.reset_pipeline()
            return
        train.edits_tallied = False

    def _train_phase_inspector(self, train: Train):
        """If Conductor finished on this train and branch has changes, launch Inspector."""
        if self._is_train_agent_active(train, "inspector"):
            return
        if self._is_train_agent_active(train, "conductor"):
            return
        if time.time() < train.inspector_cooldown_until:
            return

        branch = train.branch
        cwd = train.working_dir
        if not branch or not cwd:
            return
        # Don't relaunch if feedback already exists — let service_recovery/rework handle it
        feedback_path = self._feedback_path(branch)
        if os.path.exists(feedback_path):
            return
        if not self._git_has_branch(branch, cwd=cwd):
            return

        diff = self._git_diff_trunk(branch, cwd=cwd)
        if not diff:
            activity(f"Inspector:{train.train_id} — no diff on branch {branch}, cleaning up empty branch")
            repo = train.repo_dir or cwd
            self._remove_worktree(train.repo_dir, train.working_dir)
            if self._git_has_branch(branch, cwd=repo):
                self._git("branch", "-D", branch, cwd=repo)
            if train.spec_path:
                in_progress = train.spec_path + ".in_progress"
                if os.path.exists(in_progress):
                    os.remove(in_progress)
            train.reset_pipeline()
            return

        # Dry-run merge check: detect conflicts before wasting an Inspector launch
        repo = train.repo_dir or cwd
        rc, _, merge_err = self._git_rc("merge", "--no-commit", "--no-ff", branch, cwd=repo)
        # Always abort the trial merge (even if clean) to restore working state
        self._git("merge", "--abort", cwd=repo)
        if rc != 0:
            # Track conflict retries to prevent infinite loops
            conflict_count = 0
            spec_title = ""
            if train.spec_path:
                ip_path = train.spec_path + ".in_progress"
                spec_read_path = ip_path if os.path.exists(ip_path) else train.spec_path
                try:
                    with open(spec_read_path) as f:
                        spec_data = json.load(f)
                    conflict_count = spec_data.get("conflict_count", 0) + 1
                    spec_title = spec_data.get("title", "unknown")
                except (OSError, json.JSONDecodeError):
                    conflict_count = 1

            self._remove_worktree(train.repo_dir, train.working_dir)
            if self._git_has_branch(branch, cwd=repo):
                self._git("branch", "-D", branch, cwd=repo)
            feedback_path = self._feedback_path(branch)
            if os.path.exists(feedback_path):
                os.remove(feedback_path)

            # Reject after max retries, otherwise re-queue with incremented count
            if conflict_count > config.MAX_CONFLICT_RETRIES:
                activity(f"REJECTED [{train.train_id}] spec '{spec_title}' after {config.MAX_CONFLICT_RETRIES} conflict retries")
                project = train.repo_dir or train.working_dir or ""
                record_rejection(spec_title, f"persistent merge conflict with {config.TRUNK_BRANCH} after {config.MAX_CONFLICT_RETRIES} attempts", project=project)
                METRICS.specs_total.inc({"outcome": "conflict"})
                if train.spec_path:
                    ip_path = train.spec_path + ".in_progress"
                    if os.path.exists(ip_path):
                        os.remove(ip_path)
                    if os.path.exists(train.spec_path):
                        os.remove(train.spec_path)
            else:
                activity(f"CONFLICT [{train.train_id}] — branch {branch} conflicts with {config.TRUNK_BRANCH}, re-queuing (attempt {conflict_count}/{config.MAX_CONFLICT_RETRIES})")
                if train.spec_path:
                    ip_path = train.spec_path + ".in_progress"
                    spec_read_path = ip_path if os.path.exists(ip_path) else train.spec_path
                    try:
                        with open(spec_read_path) as f:
                            spec_data = json.load(f)
                        spec_data["conflict_count"] = conflict_count
                        with open(train.spec_path, "w") as f:
                            json.dump(spec_data, f, indent=2)
                        if os.path.exists(ip_path) and ip_path != train.spec_path:
                            os.remove(ip_path)
                    except (OSError, json.JSONDecodeError):
                        # Fall back to simple rename if JSON operations fail
                        if os.path.exists(ip_path):
                            os.rename(ip_path, train.spec_path)

            train.reset_pipeline()
            return

        feedback_path = os.path.join(
            config.REVIEW_DIR,
            f"{branch.replace('/', '_')}_feedback.md",
        )

        # Read spec JSON to give Inspector the acceptance criteria
        spec_json_str = "(spec not available)"
        if train.spec_path:
            ip_path = train.spec_path + ".in_progress"
            spec_read_path = ip_path if os.path.exists(ip_path) else train.spec_path
            try:
                with open(spec_read_path) as f:
                    spec_json_str = f.read().strip()
            except OSError:
                pass

        activity(f"Inspector:{train.train_id} — reviewing branch {branch} in {cwd}")
        prompt = config.INSPECTOR_PROMPT.format(
            branch_name=branch,
            diff=diff[:config.INSPECTOR_DIFF_MAX_CHARS],
            working_dir=cwd,
            review_dir=config.REVIEW_DIR,
            feedback_path=feedback_path,
            spec_json=spec_json_str,
        )
        self._launch_train_agent(train, "inspector", prompt, cwd=cwd)

    def _train_phase_rework(self, train: Train):
        """If inspector requested changes on this train, re-launch Conductor."""
        if train.conductor is not None or train.inspector is not None:
            # Still active — check them
            self._is_train_agent_active(train, "conductor")
            self._is_train_agent_active(train, "inspector")
            if train.conductor is not None or train.inspector is not None:
                return

        branch = train.branch
        spec_path = train.spec_path
        cwd = train.working_dir
        if not branch or not spec_path:
            return

        feedback_path = self._feedback_path(branch)
        if not os.path.exists(feedback_path):
            return

        try:
            with open(feedback_path) as f:
                first_line = f.readline().strip()
                if first_line != "CHANGES_REQUESTED":
                    return
                reviewer_feedback = first_line + "\n" + f.read()
        except OSError:
            return

        train.rework_count += 1
        if train.rework_count > config.MAX_REWORK_ATTEMPTS:
            activity(f"CANCELLED [{train.train_id}] spec after {config.MAX_REWORK_ATTEMPTS} rework attempts — branch {branch}")
            repo = train.repo_dir or cwd
            self._remove_worktree(train.repo_dir, train.working_dir)
            if self._git_has_branch(branch, cwd=repo):
                self._git("branch", "-D", branch, cwd=repo)
            in_progress = spec_path + ".in_progress"
            if os.path.exists(in_progress):
                os.remove(in_progress)
            if os.path.exists(feedback_path):
                os.remove(feedback_path)
            train.reset_pipeline()
            return

        in_progress_path = spec_path + ".in_progress"
        spec_read_path = in_progress_path if os.path.exists(in_progress_path) else spec_path
        try:
            with open(spec_read_path) as f:
                spec_data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Cannot read spec for rework %s: %s", spec_read_path, e)
            return

        activity(
            f"RETURN [{train.rework_count}/{config.MAX_REWORK_ATTEMPTS}] "
            f"— Conductor:{train.train_id} re-addressing feedback on {branch}"
        )
        prompt = config.CONDUCTOR_REWORK_PROMPT.format(
            spec_json=json.dumps(spec_data, indent=2),
            spec_title=spec_data.get("title", "untitled"),
            branch_name=branch,
            reviewer_feedback=reviewer_feedback,
            working_dir=cwd,
            repo_dir=train.repo_dir or cwd,
        )
        self._launch_train_agent(train, "conductor", prompt, cwd=cwd)
        train.edits_tallied = False
        os.remove(feedback_path)

    def _signal_project_dir(self) -> str:
        for train in self.trains:
            if train.repo_dir:
                return train.repo_dir
        return os.path.join(config.DEVELOPMENT_DIR, config.DEFAULT_PROJECT)

    def _log_watcher_tick(self, project_dir: str):
        """Per-tick grep for ERROR/WARNING lines across all registered log sources.

        Each source tracks its own read position. New lines are aggregated, then
        filtered for error patterns. Signal is triggered on-demand when matches
        are found, subject to dedup and open-bug throttling.
        """
        all_new_lines: list[str] = []
        for source in self._log_sources:
            try:
                all_new_lines.extend(source.read_new_lines(project_dir))
            except Exception as e:
                log.warning("Log source %s error: %s", source.name, e)

        if not all_new_lines:
            return

        matching = [l for l in all_new_lines if _WATCHER_PATTERN.search(l)]
        if not matching:
            return

        METRICS.log_errors_detected_total.inc(amount=len(matching))

        # Prune stale dedup entries (older than 1 hour)
        now = time.time()
        self.watcher_recent_specs = {
            k: v for k, v in self.watcher_recent_specs.items() if now - v < 3600
        }

        # Build a signature from the first matching line (normalised)
        sig = re.sub(r'\d+', 'N', matching[0])[:120]
        if sig in self.watcher_recent_specs:
            return

        # Skip if an open Signal bug already covers this
        open_bugs = self._get_cached_open_bugs()
        if len(open_bugs) >= config.MAX_SRE_OPEN_BUGS:
            return

        # Compute summary slug early so we can use it for cross-checks
        summary = re.sub(r'[^a-z0-9]+', '_', matching[0][:40].lower()).strip('_')

        # Cross-check: skip if a backlog spec or active train branch already covers this
        # issue. Match on keywords longer than 4 chars to avoid generic words like "error".
        kw_parts = [w for w in summary.split('_') if len(w) > 4]
        if kw_parts:
            try:
                for fname in os.listdir(config.BACKLOG_DIR):
                    if any(w in fname for w in kw_parts):
                        return
            except OSError:
                pass
            for train in self.trains:
                if train.branch and any(w in train.branch for w in kw_parts):
                    return

        self.watcher_recent_specs[sig] = now
        # Trigger Signal to analyze the matching lines with LLM (reactive, not polling)
        self._trigger_signal_reactive(project_dir, matching)

    def _trigger_signal_reactive(self, project_dir: str, matching_lines: list[str]):
        """Launch Signal on-demand to analyze error lines found by the log watcher."""
        if self._is_agent_active("signal"):
            return

        # Skip Signal for 2 minutes after a merge to let deployment propagate
        if self.last_merge_time > 0 and time.time() - self.last_merge_time < 120:
            return

        open_bugs = self._get_cached_open_bugs()
        if open_bugs:
            existing_bugs_text = "\n".join(
                f"- {bug.get('title', '(untitled)')}" for bug in open_bugs
            )
        else:
            existing_bugs_text = "(none)"

        ts = time.strftime("%Y%m%d_%H%M%S")
        log_lines = "\n".join(matching_lines[:50])
        METRICS.signal_triggers_total.inc()
        activity(f"SIGNAL [watcher] triggered — {len(matching_lines)} error lines detected")
        prompt = config.SIGNAL_PROMPT.format(
            log_lines=log_lines,
            timestamp=ts,
            working_dir=project_dir,
            backlog_dir=config.BACKLOG_DIR,
            existing_bugs=existing_bugs_text,
        )
        self._launch_agent("signal", prompt, cwd=project_dir)

    # Signal is now reactive — triggered by _log_watcher_tick via _trigger_signal_reactive.
    # No polling (_phase_signal) or health check (_check_signal_health) needed.

    def _train_phase_entropy_check(self, train: Train):
        """If branch has too many fix/update commits, fire Conductor and restart."""
        branch = train.branch
        cwd = train.working_dir
        if not branch or not cwd:
            return
        if not self._git_has_branch(branch, cwd=cwd):
            return

        fix_count = self._count_fix_commits_on_branch(branch, cwd=cwd)
        if fix_count >= config.ENTROPY_FIX_COMMIT_THRESHOLD:
            self._fire_conductor_entropy(train, branch, cwd=cwd)

    def _train_phase_station_manager_check(self, train: Train):
        """If Conductor edited same files >= 3 times without merge, reset branch and re-queue."""
        if train.inspector is not None:
            return
        branch = train.branch
        cwd = train.working_dir
        if not branch:
            return

        # Don't fire if the branch was already merged — let service_recovery handle it
        feedback_path = self._feedback_path(branch)
        if os.path.exists(feedback_path):
            try:
                with open(feedback_path) as f:
                    if f.readline().strip() == "APPROVED":
                        return
            except OSError:
                pass

        max_edits = max(train.file_edits.values()) if train.file_edits else 0
        if max_edits < config.MAX_ENG_EDITS_BEFORE_RESET:
            return

        activity(f"SIGNAL CHANGE [{train.train_id}] — {max_edits} edits without merge on {branch}")

        if train.spec_path and os.path.exists(train.spec_path + ".in_progress"):
            os.rename(train.spec_path + ".in_progress", train.spec_path)
            activity(f"RE-ROUTED spec: {os.path.basename(train.spec_path)}")

        fb = self._feedback_path(branch)
        if os.path.exists(fb):
            os.remove(fb)

        repo = train.repo_dir or cwd
        self._remove_worktree(train.repo_dir, train.working_dir)
        if self._git_has_branch(branch, cwd=repo):
            self._git("branch", "-D", branch, cwd=repo)

        train.reset_pipeline()

    def _deploy_to_railway(self, cwd: str | None = None):
        """Deploy to Railway via git push: staging branch first, then main if healthy."""
        crash_indicators = ("Traceback", "FATAL", "ModuleNotFoundError", "SyntaxError", "ImportError", "panic:")

        # 1. Push to staging branch → triggers Railway staging deploy
        activity(f"RAILWAY pushing to staging branch...")
        result = subprocess.run(
            ["git", "push", "origin", f"{config.TRUNK_BRANCH}:staging"],
            capture_output=True, text=True, timeout=60, cwd=cwd,
        )
        if result.returncode != 0:
            activity(f"RAILWAY staging push failed (rc={result.returncode}): {result.stderr[:200]}")
            return

        # 2. Wait for Railway to build and start the service
        activity("RAILWAY staging pushed, waiting 60s for build + startup...")
        time.sleep(60)

        # 3. Check staging health
        staging_logs = self._fetch_railway_logs(config.RAILWAY_STAGING_ENV)
        unhealthy = [ind for ind in crash_indicators if ind in staging_logs]
        if unhealthy:
            activity(f"RAILWAY staging UNHEALTHY — found: {', '.join(unhealthy)}. Skipping production deploy.")
            return

        # 4. Push to main branch → triggers Railway production deploy
        activity(f"RAILWAY staging healthy, pushing to main branch...")
        result = subprocess.run(
            ["git", "push", "origin", config.TRUNK_BRANCH],
            capture_output=True, text=True, timeout=60, cwd=cwd,
        )
        if result.returncode != 0:
            activity(f"RAILWAY production push failed (rc={result.returncode}): {result.stderr[:200]}")
            return

        activity("RAILWAY production deploy triggered")

    def _train_phase_service_recovery(self, train: Train):
        """If Inspector approved on this train, merge in main repo and restart the service."""
        if train.inspector is not None:
            # Check if inspector just finished
            self._is_train_agent_active(train, "inspector")
        if train.inspector is not None:
            return
        if not train.branch:
            return

        feedback_path = self._feedback_path(train.branch)
        if not os.path.exists(feedback_path):
            return

        try:
            with open(feedback_path) as f:
                first_line = f.readline().strip()
        except OSError:
            return

        if first_line != "APPROVED":
            return

        repo_dir = train.repo_dir or train.working_dir
        activity(f"TERMINUS [{train.train_id}] — branch {train.branch} approved, merging to trunk.")

        # Merge in the main repo (not the worktree — main is checked out there)
        rc, merge_stdout, merge_stderr = self._git_rc("merge", "--no-ff", train.branch, cwd=repo_dir)

        if rc != 0:
            # Merge failed (conflicts or other error) — abort and re-queue
            activity(f"MERGE FAILED [{train.train_id}] — {merge_stderr[:200] or merge_stdout[:200]}")
            self._git("merge", "--abort", cwd=repo_dir)

            # Re-queue the spec so it can be retried on a clean base
            if train.spec_path:
                ip_path = train.spec_path + ".in_progress"
                if os.path.exists(ip_path):
                    os.rename(ip_path, train.spec_path)
                    activity(f"REQUEUE [{train.train_id}] — spec re-queued to backlog after merge failure")

            # Clean up the worktree and branch
            self._remove_worktree(train.repo_dir, train.working_dir)
            if self._git_has_branch(train.branch, cwd=repo_dir):
                self._git("branch", "-D", train.branch, cwd=repo_dir)

            if os.path.exists(feedback_path):
                os.remove(feedback_path)

            train.reset_pipeline()
            return

        activity(f"MERGE [{train.train_id}] — {merge_stdout or 'ok'}")

        # Verify no conflict markers in tracked files
        conflict_check = self._git("diff", "--check", "HEAD~1..HEAD", cwd=repo_dir)
        if conflict_check:
            activity(f"MERGE WARNING [{train.train_id}] — conflict markers detected, reverting merge")
            self._git("revert", "--no-edit", "HEAD", cwd=repo_dir)

            if train.spec_path:
                ip_path = train.spec_path + ".in_progress"
                if os.path.exists(ip_path):
                    os.rename(ip_path, train.spec_path)
                    activity(f"REQUEUE [{train.train_id}] — spec re-queued after conflict marker detection")

            self._remove_worktree(train.repo_dir, train.working_dir)
            if self._git_has_branch(train.branch, cwd=repo_dir):
                self._git("branch", "-D", train.branch, cwd=repo_dir)

            if os.path.exists(feedback_path):
                os.remove(feedback_path)

            train.reset_pipeline()
            return

        current_head = self._git_last_commit(cwd=repo_dir)
        self.last_merge_commit = current_head
        self.last_merge_time = time.time()
        METRICS.specs_total.inc({"outcome": "merged"})
        # Successful merge clears the stall state for this project
        project = train.repo_dir or ""
        if project:
            self._project_rejection_counts.pop(project, None)
            self._stalled_projects.pop(project, None)

        # Remove the worktree, then delete the feature branch from the main repo
        self._remove_worktree(train.repo_dir, train.working_dir)
        if self._git_has_branch(train.branch, cwd=repo_dir):
            self._git("branch", "-D", train.branch, cwd=repo_dir)

        if config.RAILWAY_PROJECT:
            self._deploy_to_railway(cwd=repo_dir)
        elif config.SERVICE_RESTART_CMD:
            try:
                cmd_parts = shlex.split(config.SERVICE_RESTART_CMD)
            except ValueError as e:
                activity(f"SERVICE restart skipped — invalid command string: {e}")
                cmd_parts = []
            if cmd_parts:
                try:
                    result = subprocess.run(
                        cmd_parts,
                        timeout=config.SERVICE_RESTART_TIMEOUT,
                        capture_output=True, text=True,
                    )
                    if result.returncode == 0:
                        activity("SERVICE restarted successfully")
                    else:
                        activity(f"SERVICE restart failed (rc={result.returncode}): {result.stderr[:200]}")
                except subprocess.TimeoutExpired:
                    activity(f"SERVICE restart timed out after {config.SERVICE_RESTART_TIMEOUT}s")
        else:
            activity("SERVICE restart skipped (no deployment method configured)")

        if train.spec_path and os.path.exists(train.spec_path + ".in_progress"):
            os.remove(train.spec_path + ".in_progress")

        if os.path.exists(feedback_path):
            os.remove(feedback_path)

        train.reset_pipeline()

    def _log_ops_summary(self, output: str):
        """Extract and log the ops agent's activity summary with visual breakers."""
        if not output or not output.strip():
            return
        # Use the full output as the summary — ops is instructed to lead with it
        lines = output.strip().splitlines()
        # Cap at 15 lines to keep the log readable
        summary_lines = lines[:15]
        activity("*" * 60)
        activity("OPS REPORT — last hour")
        for line in summary_lines:
            activity(f"  {line}")
        activity("*" * 60)

    def _phase_ops(self):
        """Periodically analyze orchestrator activity and implement small improvements."""
        ops_agent = self.active_agents.get("ops")
        if self._is_agent_active("ops"):
            return

        # If ops just completed, log summary and check for new commits → restart
        if self._ops_head_before is not None:
            if ops_agent is not None:
                self._log_ops_summary(ops_agent.get_output())
            current_head = self._git_last_commit(cwd=config.BASE_DIR)
            if current_head != self._ops_head_before:
                self._ops_head_before = None
                conducting = any(
                    t.conductor is not None and t.conductor.proc is not None
                    and t.conductor.proc.poll() is None
                    for t in self.trains
                )
                if conducting:
                    activity("OPS RESTART deferred — conductor is mid-run, will restart when it finishes")
                    self.restart_pending = True
                else:
                    self._request_self_restart()
            self._ops_head_before = None
            return

        # Skip if in cooldown
        if "ops" in self.agent_cooldowns and time.time() < self.agent_cooldowns["ops"]:
            return

        activity_tail, git_log = self._gather_ops_context()
        prompt = config.OPS_PROMPT.format(
            base_dir=config.BASE_DIR,
            activity_tail=activity_tail or "(no activity log)",
            git_log=git_log or "(no commits)",
        )
        agent = self._launch_agent("ops", prompt, cwd=config.BASE_DIR)
        if agent is not None:
            self._ops_head_before = self._git_last_commit(cwd=config.BASE_DIR)

    # ─── SLA enforcement ─────────────────────────────────────────────────

    def _check_spec_sla(self):
        """Drop specs that exceed the wall-clock SLA across all pipeline phases."""
        now = time.time()
        for train in self.trains:
            if not train.spec_path or train.spec_started_at <= 0:
                continue
            elapsed = now - train.spec_started_at
            # Add 60s grace period to reduce false positives from timing jitter (tick interval is 10s)
            if elapsed < config.SPEC_SLA_SECONDS + 60:
                continue

            spec_name = os.path.basename(train.spec_path) if train.spec_path else "unknown"
            activity(
                f"SLA BREACH [{train.train_id}] — spec '{spec_name}' exceeded "
                f"{config.SPEC_SLA_SECONDS}s wall-clock ({int(elapsed)}s elapsed). Dropping."
            )

            # Kill any active agents on this train
            for role in ("conductor", "inspector"):
                agent = getattr(train, role)
                if agent and agent.proc and agent.proc.poll() is None:
                    agent.proc.terminate()
                    try:
                        agent.proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        agent.proc.kill()
                        agent.proc.wait()
                    agent.save_log(marker="[SLA BREACH]")
                setattr(train, role, None)

            # Remove the in-progress spec and record failure
            branch_name = train.branch.removeprefix("feature/") if train.branch else spec_name
            record_failure(branch_name, f"SLA breach ({int(elapsed)}s elapsed)")
            METRICS.specs_total.inc({"outcome": "sla_breach"})
            if train.spec_path:
                ip = train.spec_path + ".in_progress"
                if os.path.exists(ip):
                    os.remove(ip)

            # Clean up worktree and branch
            repo = train.repo_dir or train.working_dir
            branch = train.branch
            self._remove_worktree(train.repo_dir, train.working_dir)
            if repo and branch and self._git_has_branch(branch, cwd=repo):
                self._git("branch", "-D", branch, cwd=repo)

            fb = self._feedback_path(branch) if branch else None
            if fb and os.path.exists(fb):
                os.remove(fb)

            train.reset_pipeline()

    def _check_checkpoint_sla(self):
        """Detect trains stuck at checkpoint (feedback exists, no agents active)."""
        now = time.time()
        for train in self.trains:
            if not train.branch or not train.spec_path:
                continue
            # Only applies when no agents are active on this train
            if train.conductor is not None or train.inspector is not None:
                train.checkpoint_idle_since = 0.0
                continue

            feedback_path = self._feedback_path(train.branch)
            if not os.path.exists(feedback_path):
                train.checkpoint_idle_since = 0.0
                continue

            # Train has a spec, feedback exists, but no agents running = stuck at checkpoint
            if train.checkpoint_idle_since <= 0:
                train.checkpoint_idle_since = now
                continue

            idle_for = now - train.checkpoint_idle_since
            # Add 60s grace period to prevent timing jitter from removing approved feedback prematurely
            if idle_for < config.CHECKPOINT_SLA_SECONDS + 60:
                continue

            activity(
                f"SLA CHECKPOINT [{train.train_id}] — idle at checkpoint for "
                f"{int(idle_for)}s (limit {config.CHECKPOINT_SLA_SECONDS}s). "
                f"Removing stale feedback to retry."
            )
            # Remove the feedback file so the pipeline re-evaluates
            os.remove(feedback_path)
            train.checkpoint_idle_since = 0.0

    def _check_idle_sla(self):
        """If all trains idle and backlog empty for too long, prompt dispatcher."""
        any_busy = any(t.branch for t in self.trains)
        backlog_count = len(self._backlog_specs())
        has_backlog = backlog_count > 0

        if any_busy or has_backlog:
            self.all_idle_since = 0.0
            return

        now = time.time()
        if self.all_idle_since <= 0:
            self.all_idle_since = now
            return

        idle_for = now - self.all_idle_since
        if idle_for < config.IDLE_SLA_SECONDS:
            return

        # Don't fire if service is suspended (rate-limited)
        if time.time() < self.sleep_until:
            return
        # Don't fire if dispatcher is active or in cooldown
        if self._is_agent_active("dispatcher"):
            return
        if "dispatcher" in self.agent_cooldowns and now < self.agent_cooldowns["dispatcher"]:
            return

        activity(f"SLA IDLE — all trains idle for {int(idle_for)}s, triggering dispatcher")
        self.all_idle_since = 0.0
        # Clear the dispatcher's last launch time so it fires immediately
        self.last_launch_times.pop("dispatcher", None)

    # ─── Main loop ───────────────────────────────────────────────────────

    def run(self):
        activity("=" * 60)
        activity("YAMANOTE LINE OPEN")
        activity(f"  Tick interval: {config.TICK_INTERVAL}s")
        activity(f"  Backlog: {config.BACKLOG_DIR}")
        activity(f"  Activity log: {config.ACTIVITY_LOG}")
        activity(f"  Agent timeout: {config.AGENT_TIMEOUT_SECONDS}s")
        activity(f"  Fare limit: {config.MAX_AGENT_LAUNCHES_PER_HOUR} launches/hr")
        activity(f"  Entropy threshold: {config.ENTROPY_FIX_COMMIT_THRESHOLD} fix commits")
        activity(f"  Trains: {len(self.trains)}")
        for train in self.trains:
            activity(f"    {train.train_id}: type={train.train_type} complexity={train.complexity} conductor={train.conductor_model} inspector={train.inspector_model}")
        for agent_name, model in config.AGENT_MODELS.items():
            interval = config.AGENT_MIN_INTERVALS.get(agent_name, 0)
            activity(f"  {agent_name}: model={model}  min_interval={interval}s")
        activity("=" * 60)

        last_tick_time = time.time()
        try:
            while True:
                # Sleep/wake detection — if the gap since the last tick is much larger
                # than TICK_INTERVAL, the machine likely slept. Advance all timestamps
                # so rate limits stay meaningful and agents don't burst-launch on wake.
                now = time.time()
                gap = now - last_tick_time
                if gap > config.TICK_INTERVAL * 6:  # ~60s threshold
                    drift = gap - config.TICK_INTERVAL
                    # Only log wake events every 5 minutes to reduce noise
                    if now - self.last_wake_log_time >= 300:
                        activity(f"WAKE DETECTED — {int(gap)}s gap, advancing timestamps by {int(drift)}s")
                        self.last_wake_log_time = now
                    for k in list(self.last_launch_times):
                        self.last_launch_times[k] += drift
                    for k in list(self.agent_cooldowns):
                        self.agent_cooldowns[k] += drift
                    if self.sleep_until > 0:
                        self.sleep_until += drift
                    for train in self.trains:
                        if train.conductor_cooldown_until > 0:
                            train.conductor_cooldown_until += drift
                        if train.inspector_cooldown_until > 0:
                            train.inspector_cooldown_until += drift
                        if train.triage_cooldown_until > 0:
                            train.triage_cooldown_until += drift
                last_tick_time = now

                # Sleep mode check
                if time.time() < self.sleep_until:
                    remaining = int(self.sleep_until - time.time())
                    if remaining % 300 < config.TICK_INTERVAL:  # log every ~5 min
                        activity(f"SERVICE SUSPENDED — {remaining}s remaining (fare limit)")
                    time.sleep(config.TICK_INTERVAL)
                    continue

                # Pause check — touch agents/pause to pause, rm to resume
                if os.path.exists(config.PAUSE_FILE):
                    if not getattr(self, '_pause_logged', False):
                        activity("PAUSED — agents/pause file detected, skipping all launches")
                        self._pause_logged = True
                    time.sleep(config.TICK_INTERVAL)
                    continue
                elif getattr(self, '_pause_logged', False):
                    activity("RESUMED — agents/pause file removed")
                    self._pause_logged = False

                # Rotate logs if needed (cheap size check every tick; prune hourly)
                self._maybe_rotate_logs()
                # Periodic maintenance: recycle stale HOLD specs, GC orphaned worktrees
                self._recycle_stale_drafts()
                self._gc_orphaned_worktrees()

                # Advance tick — invalidates all per-tick caches
                self._advance_tick()

                # Per-train phases
                for train in self.trains:
                    self._train_phase_service_recovery(train)
                    self._train_phase_rework(train)
                    self._train_phase_conductor(train)
                    self._train_phase_triage(train)
                    self._train_phase_conductor_launch(train)
                    self._train_phase_inspector(train)
                    self._train_phase_entropy_check(train)
                    self._train_phase_station_manager_check(train)

                # SLA checks
                self._check_spec_sla()
                self._check_checkpoint_sla()
                self._check_idle_sla()

                # Deferred ops restart — fire once all conductors have finished
                if self.restart_pending:
                    conducting = any(
                        t.conductor is not None and t.conductor.proc is not None
                        and t.conductor.proc.poll() is None
                        for t in self.trains
                    )
                    if not conducting:
                        self._request_self_restart()

                # Global phases
                project_dir = self._signal_project_dir()
                self._log_watcher_tick(project_dir)     # fast grep every tick; triggers Signal on errors
                self._phase_dispatcher()
                self._phase_ops()

                # Tick summary
                active = [n for n, a in self.active_agents.items() if a is not None]
                for train in self.trains:
                    if train.conductor is not None:
                        active.append(f"conductor:{train.train_id}")
                    if train.inspector is not None:
                        active.append(f"inspector:{train.train_id}")
                    if train.triage is not None:
                        active.append(f"triage:{train.train_id}")
                backlog_count = self._get_cached_backlog_count()
                if active or backlog_count:
                    log.info(
                        "Tick: active=[%s] backlog=%d",
                        ", ".join(active) if active else "none",
                        backlog_count,
                    )

                time.sleep(config.TICK_INTERVAL)
        except KeyboardInterrupt:
            activity("LAST TRAIN — terminating active agents...")
            for name, agent in self.active_agents.items():
                if agent and agent.proc and agent.proc.poll() is None:
                    activity(f"Terminating {name} (PID {agent.proc.pid})")
                    try:
                        os.killpg(agent.proc.pid, signal.SIGTERM)
                    except (ProcessLookupError, PermissionError, OSError):
                        agent.proc.terminate()
                    try:
                        agent.proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(agent.proc.pid, signal.SIGKILL)
                        except (ProcessLookupError, PermissionError, OSError):
                            agent.proc.kill()
                    agent.save_log()
            for train in self.trains:
                for role, agent in [("conductor", train.conductor), ("inspector", train.inspector), ("triage", train.triage)]:
                    if agent and agent.proc and agent.proc.poll() is None:
                        activity(f"Terminating {role}:{train.train_id} (PID {agent.proc.pid})")
                        try:
                            os.killpg(agent.proc.pid, signal.SIGTERM)
                        except (ProcessLookupError, PermissionError, OSError):
                            agent.proc.terminate()
                        try:
                            agent.proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            try:
                                os.killpg(agent.proc.pid, signal.SIGKILL)
                            except (ProcessLookupError, PermissionError, OSError):
                                agent.proc.kill()
                        agent.save_log()
            activity("All agents stopped. Goodbye.")


if __name__ == "__main__":
    import argparse
    import atexit

    parser = argparse.ArgumentParser(description="Yamanote — multi-agent orchestrator")
    parser.add_argument("--dashboard", action="store_true",
                        help="Enable web dashboard on port 8080")
    parser.add_argument("--dashboard-port", type=int, default=0, metavar="PORT",
                        help="Enable web dashboard on a specific port")
    args = parser.parse_args()

    # Check for existing instance via PID file
    if os.path.exists(config.PID_FILE):
        try:
            with open(config.PID_FILE) as f:
                old_pid = int(f.read().strip())
            # Check if process is still running
            try:
                os.kill(old_pid, 0)  # Signal 0 just checks if process exists
                log.error("Orchestrator already running (PID %d). Exiting.", old_pid)
                sys.exit(1)
            except OSError:
                # Process doesn't exist, stale PID file
                log.warning("Removing stale PID file (process %d no longer exists)", old_pid)
                os.remove(config.PID_FILE)
        except (ValueError, OSError):
            # Malformed or unreadable PID file, remove it
            os.remove(config.PID_FILE)

    # Write our PID and ensure cleanup on exit
    with open(config.PID_FILE, "w") as f:
        f.write(str(os.getpid()))
    atexit.register(lambda: os.path.exists(config.PID_FILE) and os.remove(config.PID_FILE))

    # Priority: --dashboard-port > --dashboard (8080) > env var > disabled
    dash_port = args.dashboard_port or (8080 if args.dashboard else config.DASHBOARD_PORT)

    # Route SIGTERM (sent by systemd stop / kill) through the same cleanup path as Ctrl-C
    def _sigterm_handler(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _sigterm_handler)

    station_manager = StationManager()

    if dash_port:
        from dashboard import start_dashboard
        start_dashboard(station_manager, dash_port)

    station_manager.run()
