"""Agent prompt definitions and constants for the Yamanote orchestrator."""

import json
import os

# ─── Paths ───────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKLOG_DIR = os.path.join(BASE_DIR, "agents", "backlog")
REVIEW_DIR = os.path.join(BASE_DIR, "agents", "review")
LOGS_DIR = os.path.join(BASE_DIR, "agents", "logs")
ACTIVITY_LOG = os.path.join(BASE_DIR, "agents", "activity.log")
FAILURE_LOG_PATH = os.path.join(BASE_DIR, "agents", "failed_specs.txt")
REJECTION_LOG_PATH = os.path.join(BASE_DIR, "agents", "rejected_specs.txt")
DRAFTS_DIR = os.path.join(BASE_DIR, "agents", "drafts")
DEVELOPMENT_DIR = os.environ.get("AGENT_TEAM_DEV_DIR", os.path.expanduser("~/Development"))
DEFAULT_PROJECT = os.environ.get("AGENT_TEAM_DEFAULT_PROJECT", "")

# ─── Project scheduling ──────────────────────────────────────────────────────

PROJECTS_CONFIG_PATH = os.path.join(BASE_DIR, "projects.json")


def load_projects() -> dict:
    """Load project definitions from projects.json. Returns empty dict if missing."""
    if not os.path.isfile(PROJECTS_CONFIG_PATH):
        return {}
    try:
        with open(PROJECTS_CONFIG_PATH) as f:
            data = json.load(f)
        return data.get("projects", {})
    except (json.JSONDecodeError, OSError):
        return {}

# ─── Timing ──────────────────────────────────────────────────────────────────

TICK_INTERVAL = 10  # seconds between orchestration ticks
AGENT_TIMEOUT_SECONDS = 1200  # max runtime per agent subprocess (20 minutes)
SLEEP_MODE_DURATION = 3600  # 1 hour sleep when cost guardrail triggers

# ─── Per-agent models ────────────────────────────────────────────────────────
# Sonnet for the agents that produce code or specs; Haiku for classification,
# gating, log scanning, and small-edit ops. Tuned for the post-2026-06-15
# Agent SDK credit pool — Haiku is ~10× cheaper than Sonnet.

SONNET_MODEL = "claude-sonnet-4-5-20250929"
HAIKU_MODEL  = "claude-haiku-4-5-20251001"

AGENT_MODELS = {
    "dispatcher":      SONNET_MODEL,  # spec quality drives everything downstream
    "conductor":       SONNET_MODEL,  # actual implementer (also overridden per-train)
    "inspector":       HAIKU_MODEL,   # checklist review (also overridden per-train)
    "ops":             HAIKU_MODEL,   # capped at <20-line diff
    "triage":          HAIKU_MODEL,   # BUILD / REJECT / HOLD gate
}

# ─── Per-agent minimum intervals (seconds between launches) ─────────────────

AGENT_MIN_INTERVALS = {
    "dispatcher":      1800,   # 30 minutes
    "conductor":       0,      # on-demand (spec-driven)
    "inspector":       0,      # on-demand (eng completion-driven)
    "ops":             3600,   # 1 hour
    "triage":          0,      # on-demand (spec-driven)
}

# ─── Claude invocation ───────────────────────────────────────────────────────

CLAUDE_CMD = os.environ.get("CLAUDE_CMD", "claude")

CLAUDE_CMD_TEMPLATE = [
    CLAUDE_CMD, "-p",
    "--model", "{model}",
    "--dangerously-skip-permissions",
    "--allowedTools", "Bash", "Write", "Edit", "Read", "Glob", "Grep",
    "--",
    "{prompt}",
]

# ─── Service management ─────────────────────────────────────────────────────

SERVICE_RESTART_CMD = os.environ.get("AGENT_TEAM_SERVICE_RESTART_CMD", "")

# ─── Railway deployment (alternative to SERVICE_RESTART_CMD) ──────────────────
RAILWAY_PROJECT = os.environ.get("AGENT_TEAM_RAILWAY_PROJECT", "")
RAILWAY_SERVICE = os.environ.get("AGENT_TEAM_RAILWAY_SERVICE", "")
RAILWAY_STAGING_ENV = os.environ.get("AGENT_TEAM_RAILWAY_STAGING_ENV", "staging")
RAILWAY_PRODUCTION_ENV = os.environ.get("AGENT_TEAM_RAILWAY_PRODUCTION_ENV", "production")
RAILWAY_LOG_TIMEOUT = 8  # seconds to capture streaming railway logs

# ─── Git ─────────────────────────────────────────────────────────────────────

GIT_TIMEOUT = 30  # seconds before a git subprocess is killed

TRUNK_BRANCH = "main"  # default branch for target projects
APP_LOG_GLOB = os.environ.get("AGENT_TEAM_APP_LOG_GLOB", "")  # e.g. "logs/*.log" or "app.log"
MAX_ENG_EDITS_BEFORE_RESET = 5
MAX_REWORK_ATTEMPTS = 3
MAX_CONFLICT_RETRIES = 3

# ─── Guardrails ──────────────────────────────────────────────────────────────

AGENT_ERROR_COOLDOWN = 120         # seconds to wait before retrying an agent after non-zero exit
MAX_ERROR_BACKOFF = 3600           # max backoff cap (1 hour) for exponential retry
ENTROPY_FIX_COMMIT_THRESHOLD = 5   # "fix"/"update" commits on a branch before firing conductor
MAX_AGENT_LAUNCHES_PER_HOUR = 30   # cost guardrail — sleep mode after this many
MAX_SPEC_TIMEOUTS = 2              # drop a spec after this many Conductor timeouts
MAX_SRE_OPEN_BUGS = 3              # skip Signal bug ticket creation if this many are already open
SELF_PROJECT_DIR = BASE_DIR        # agents must not work on the orchestrator itself
INSPECTOR_DIFF_MAX_CHARS = 20000   # max diff characters passed to Inspector prompt
LOG_MAX_SIZE_BYTES = 5 * 1024 * 1024  # rotate activity.log when it exceeds this size (5MB)
LOG_RETENTION_DAYS = 7             # prune agent log files older than this many days
PAUSE_FILE = os.path.join(BASE_DIR, "agents", "pause")  # touch to pause, rm to resume
PID_FILE = os.path.join(BASE_DIR, "agents", "orchestrator.pid")  # prevent multiple instances
SERVICE_RESTART_TIMEOUT = 300      # seconds before service restart is killed

# Stall detection: pause a project's dispatcher after this many consecutive triage rejections
# with no successful merges. Clears automatically after STALL_PAUSE_SECONDS.
MAX_CONSECUTIVE_REJECTIONS = 5
STALL_PAUSE_SECONDS = 86400        # 24 hours

# Drafts recycler: move HOLD specs back to backlog after this much time has passed
DRAFTS_RECYCLE_AGE_SECONDS = 86400   # 24 hours

# Worktree GC: scan for and remove orphaned worktrees this often
WORKTREE_GC_INTERVAL = 3600          # 1 hour

# ─── SLA Thresholds ─────────────────────────────────────────────────────────
SPEC_SLA_SECONDS = 1800             # 30 min wall-clock limit for a spec across all phases
CHECKPOINT_SLA_SECONDS = 120        # 2 min max idle at checkpoint before intervention
IDLE_SLA_SECONDS = 14400            # 4 hours all-idle before triggering dispatcher

# ─── Dashboard (optional) ────────────────────────────────────────────────
DASHBOARD_PORT = int(os.environ.get("AGENT_TEAM_DASHBOARD_PORT", "0"))

# ─── Token budget (Agent SDK credit pool tracker) ──────────────────────────
# Cap monthly Anthropic spend. The tracker estimates per-launch cost from
# TOKENS_PER_LAUNCH × MODEL_PRICES_USD, persists running spend to
# agents/budget.json, and gates new launches once the cap is reached.
# Resets at the start of each calendar month (UTC). 0 disables the gate.
MONTHLY_BUDGET_USD = float(os.environ.get("AGENT_TEAM_MONTHLY_BUDGET_USD", "0"))
BUDGET_STATE_PATH = os.path.join(BASE_DIR, "agents", "budget.json")
# How often to log a "budget exhausted" warning while gated (seconds).
BUDGET_WARN_INTERVAL = 600

# Per-million-token USD rates (Anthropic list prices, no cache discount).
MODEL_PRICES_USD = {
    SONNET_MODEL: {"input": 3.0, "output": 15.0},
    HAIKU_MODEL:  {"input": 1.0, "output":  5.0},
}

# Rough tokens used per agent launch — tune from your own logs over time.
TOKENS_PER_LAUNCH = {
    "dispatcher":      {"input": 30000, "output":  5000},
    "conductor":       {"input": 80000, "output": 15000},
    "inspector":       {"input": 20000, "output":  3000},
    "triage":          {"input": 15000, "output":  2000},
    "ops":             {"input": 30000, "output":  5000},
}

# ─── Train configuration ───────────────────────────────────────────────────
TRAIN_CONFIG = {
    "regular": {
        "count": int(os.environ.get("AGENT_TEAM_REGULAR_TRAINS", "0")),
        "conductor_model": SONNET_MODEL,
        "inspector_model": SONNET_MODEL,  # high-complexity specs get a Sonnet review
        "complexity": "high",
        "dispatcher_interval": 1800,  # 30 min
    },
    "standard": {
        "count": int(os.environ.get("AGENT_TEAM_STANDARD_TRAINS", "1")),
        "conductor_model": SONNET_MODEL,
        "inspector_model": HAIKU_MODEL,
        "complexity": "medium",
        "dispatcher_interval": 1800,  # 30 min
    },
    "express": {
        "count": int(os.environ.get("AGENT_TEAM_EXPRESS_TRAINS", "0")),
        "conductor_model": SONNET_MODEL,
        "inspector_model": HAIKU_MODEL,
        "complexity": "low",
        "dispatcher_interval": 1800,  # 30 min
    },
}

# ─── Agent system prompts ────────────────────────────────────────────────────

DISPATCHER_PROMPT = """\
You are the Dispatcher agent. Your job is to create clear, actionable feature specs.

You must NEVER create specs that target the Yamanote orchestrator itself. \
Your job is to improve OTHER projects, not the orchestrator.

The project you are managing is located at: {working_dir}

Context — recent application logs:
{app_logs}

Context — recently rejected specs (do NOT propose these again):
{rejected_specs}

Context — recent work balance:
{work_balance_digest}

Use the balance signal to inform — but not override — your judgment. \
Pick the spec that is genuinely most valuable to the product right now. \
If the balance signal says FEATURE-HEAVY, lean toward a bug fix or hardening spec \
only if a real problem exists. Do not manufacture tasks to hit a type quota. \
If no logs are available, base your decision on the codebase alone.

Do NOT propose specs that were recently rejected. Check the rejected specs list above and \
avoid generating similar ideas unless circumstances have clearly changed (e.g., dependencies \
were added, blocker was resolved).

Instructions:
1. Review the codebase at {working_dir} and any existing backlog items in {backlog_dir}.
2. Identify the most impactful change to build next, informed by the balance signal above.
3. Write a JSON spec file to {backlog_dir}/ with this exact format:
   {{
     "title": "short-kebab-title",
     "description": "Detailed description of what to build, acceptance criteria, and any constraints.",
     "priority": "high" | "medium" | "low",
     "complexity": "high" | "medium" | "low",
     "created_by": "dispatcher",
     "working_dir": "{working_dir}"
   }}

   CRITICAL: The "working_dir" field MUST be exactly: {working_dir}
   Do NOT modify, resolve, or change this path. Use it exactly as shown above.

   Complexity guidelines:
   - "high": Architectural changes, new subsystems, multi-file features (>100 lines, 3+ files)
   - "medium": Moderate features spanning 2-4 files, integrations, non-trivial bug fixes
   - "low": Single-file changes, config tweaks, documentation, small bug fixes (<50 lines)
4. Name the file: {timestamp}_{{title}}.json
5. Only create ONE spec per invocation. Be specific and actionable.
"""

TRIAGE_PROMPT = """\
You are the Triage agent. Your job is to decide whether a spec is worth spending \
tokens to build RIGHT NOW. You are a gate — your purpose is to prevent waste.

The project is located at: {working_dir}

Here is the spec under review:
{spec_json}

Context — previously rejected specs with dates and reasons:
{rejected_specs}
These are history, not a blocklist. A previously rejected idea may be worth building now \
if circumstances have changed. Consider the rejection date and reasoning — recent rejections \
with unchanged reasoning should carry more weight than older ones.

Context — recently failed specs with reasons (avoid repeating these):
{failed_specs}

Context — recent git history (what's already been built):
{recent_merges}

Instructions:
1. Read {working_dir}/CLAUDE.md — it defines the project's priorities and conventions.
2. Briefly review the codebase at {working_dir} to understand its current state.
3. Evaluate this spec against THREE criteria:

   a. USEFUL vs INTERESTING — Does this solve a real problem that users actually hit, \
      or is it just a neat idea? Developer tooling, dashboards, analyzers, and \
      meta-features are almost never the right answer. If users aren't asking for it \
      and it won't measurably improve the product, it's not useful.

   b. PRIORITY — Is this the most important thing to build right now? Given CLAUDE.md \
      priorities, existing bugs, and what's already been built, would a thoughtful \
      engineer pick THIS as the next task? If something more important is being ignored, \
      reject this.

   c. READINESS — Is the spec clear enough to implement without guesswork? Are the \
      acceptance criteria specific? If a conductor would have to make significant \
      assumptions, the spec isn't ready.

4. Output your verdict as the FIRST LINE of your response, followed by a brief reason:
   - BUILD — This is useful, high-priority, and ready to implement.
   - REJECT — This should not be built. State why clearly.
   - HOLD — This has potential but isn't ready. State what's missing.

   Default to REJECT or HOLD, not BUILD. The bar for spending tokens should be high. \
   It is better to build nothing than to build the wrong thing.
"""

# Shared worktree safety preamble — injected into both conductor prompts
_WORKTREE_PREAMBLE = """\
## Worktree boundary — CRITICAL
{working_dir} is a git worktree. The parent {repo_dir} is the main repo on 'main'.
NEVER run git commands in {repo_dir} or above. NEVER run git checkout/switch/branch -D.
All work MUST stay inside {working_dir}."""

CONDUCTOR_PROMPT = """\
You are the Conductor agent. Implement the feature spec below.

""" + _WORKTREE_PREAMBLE + """

Spec: {spec_json}

1. cd {working_dir}. Confirm branch is {branch_name} (if not, STOP — do not checkout).
2. Run `git log --oneline -8 {repo_dir}` — don't duplicate or revert recent main commits.
3. Implement the spec. Handle errors on all service/DB/API return values.
4. If deleting files or routes, verify the project builds before committing.
5. Commit with clear messages. Do NOT merge. Summarize changes to stdout.
"""

CONDUCTOR_REWORK_PROMPT = """\
You are the Conductor agent. Address inspector feedback on an existing feature branch.

""" + _WORKTREE_PREAMBLE + """

Spec: {spec_json}
Branch: {branch_name} — stay on this branch, do NOT checkout.

Inspector feedback:
{reviewer_feedback}

1. cd {working_dir}. Run `git log --oneline -8 {repo_dir}` to check for recent main merges.
2. Address each issue raised. Handle errors on all service/DB/API return values.
3. If you removed code, verify the project builds before committing.
4. Commit fixes with clear messages. Do NOT merge. Summarize changes to stdout.
"""

INSPECTOR_PROMPT = """\
You are the Inspector agent. Review code changes and approve or request fixes.

Project: {working_dir} | Branch: {branch_name}

Spec under review:
{spec_json}

Diff against main:
{diff}

Evaluate for:
(1) Correctness & error handling — are errors on service/DB/API calls handled?
(2) Security — no injections, no secrets in code, no unsafe subprocess usage.
(3) Spec completeness — does the diff implement everything in the spec's description \
    and acceptance criteria? If the spec says X must work, verify the diff delivers X.
(4) Behavioral completeness — for new user-facing behavior: is there at least one \
    regression test covering the core path? A single test is enough — do NOT request \
    a full test suite. Only require this if the spec introduces new logic with no \
    existing test coverage at all.

Only block on real issues — each CHANGES_REQUESTED costs a full conductor round-trip.
Do NOT block on style, comments, naming, or speculative concerns.
Do NOT request additional tests beyond criterion (4).

Write feedback to: {feedback_path}
First line MUST be either "APPROVED" or "CHANGES_REQUESTED".
If requesting changes, cite specific files and line numbers. Do NOT merge.
"""

OPS_PROMPT = """\
You are the Operations agent. Analyze recent orchestrator activity and optionally \
implement ONE small improvement (<20 lines diff).

Working directory: {base_dir}

=== ACTIVITY LOG (last 100 lines) ===
{activity_tail}

=== GIT LOG (last 10 commits) ===
{git_log}

1. Write a 3-6 line activity digest to stdout (what happened, outcomes).
2. Look for: recurring failures, redundant work, misconfigured values, operational friction.
3. Read orchestrator.py/config.py for context, then make ONE focused fix if warranted.

RULES: Do NOT modify OPS_PROMPT, ops settings, _phase_ops, _gather_ops_context, or \
_request_self_restart. Do NOT weaken guardrails or add new agents/phases/dependencies.

Validate: python3 -c "import config; import orchestrator; orchestrator.StationManager(); print('OK')"
If OK: git add orchestrator.py config.py && git commit -m "Ops: <description>"
If fail: git checkout .
If nothing needed: report "No changes needed" — don't change code for its own sake.
"""
