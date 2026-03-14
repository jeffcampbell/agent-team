"""Agent prompt definitions and constants for the Yamanote orchestrator."""

import os

# ─── Paths ───────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKLOG_DIR = os.path.join(BASE_DIR, "agents", "backlog")
REVIEW_DIR = os.path.join(BASE_DIR, "agents", "review")
LOGS_DIR = os.path.join(BASE_DIR, "agents", "logs")
ACTIVITY_LOG = os.path.join(BASE_DIR, "agents", "activity.log")
DEVELOPMENT_DIR = os.environ.get("AGENT_TEAM_DEV_DIR", os.path.expanduser("~/Development"))
DEFAULT_PROJECT = os.environ.get("AGENT_TEAM_DEFAULT_PROJECT", "")

# ─── Timing ──────────────────────────────────────────────────────────────────

TICK_INTERVAL = 10  # seconds between orchestration ticks
AGENT_TIMEOUT_SECONDS = 1200  # max runtime per agent subprocess (20 minutes)
SLEEP_MODE_DURATION = 3600  # 1 hour sleep when cost guardrail triggers

# ─── Per-agent models ────────────────────────────────────────────────────────
# Sonnet for everything — good balance of capability and cost.

AGENT_MODELS = {
    "dispatcher":      "claude-sonnet-4-5-20250929",
    "conductor":       "claude-sonnet-4-5-20250929",
    "inspector":       "claude-sonnet-4-5-20250929",
    "signal":          "claude-sonnet-4-5-20250929",
    "station_manager": "claude-sonnet-4-5-20250929",
    "ops":             "claude-sonnet-4-5-20250929",
}

# ─── Per-agent minimum intervals (seconds between launches) ─────────────────

AGENT_MIN_INTERVALS = {
    "dispatcher":      1800,   # 30 minutes
    "conductor":       0,      # on-demand (spec-driven)
    "inspector":       0,      # on-demand (eng completion-driven)
    "signal":          0,      # on-demand (triggered by log watcher, not polling)
    "station_manager": 0,      # on-demand
    "ops":             3600,   # 1 hour
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

TRUNK_BRANCH = "main"  # default branch for target projects
APP_LOG_GLOB = os.environ.get("AGENT_TEAM_APP_LOG_GLOB", "")  # e.g. "logs/*.log" or "app.log"
MAX_ENG_EDITS_BEFORE_RESET = 5
MAX_REWORK_ATTEMPTS = 3

# ─── Guardrails ──────────────────────────────────────────────────────────────

AGENT_ERROR_COOLDOWN = 120         # seconds to wait before retrying an agent after non-zero exit
MAX_ERROR_BACKOFF = 3600           # max backoff cap (1 hour) for exponential retry
SIGNAL_MAX_BACKOFF = 300           # cap Signal failure backoff at 5 min
ENTROPY_FIX_COMMIT_THRESHOLD = 5   # "fix"/"update" commits on a branch before firing conductor
MAX_AGENT_LAUNCHES_PER_HOUR = 30   # cost guardrail — sleep mode after this many
MAX_SPEC_TIMEOUTS = 2              # drop a spec after this many Conductor timeouts
MAX_SRE_OPEN_BUGS = 3              # skip Signal launch if this many Signal bugs are already open
SELF_PROJECT_DIR = BASE_DIR        # agents must not work on the orchestrator itself

# ─── Dashboard (optional) ────────────────────────────────────────────────
DASHBOARD_PORT = int(os.environ.get("AGENT_TEAM_DASHBOARD_PORT", "0"))

# ─── Train configuration ───────────────────────────────────────────────────
TRAIN_CONFIG = {
    "regular": {
        "count": int(os.environ.get("AGENT_TEAM_REGULAR_TRAINS", "0")),
        "conductor_model": "claude-sonnet-4-5-20250929",
        "inspector_model": "claude-sonnet-4-5-20250929",
        "complexity": "high",
        "dispatcher_interval": 1800,  # 30 min
    },
    "standard": {
        "count": int(os.environ.get("AGENT_TEAM_STANDARD_TRAINS", "1")),
        "conductor_model": "claude-sonnet-4-5-20250929",
        "inspector_model": "claude-sonnet-4-5-20250929",
        "complexity": "medium",
        "dispatcher_interval": 1800,  # 30 min
    },
    "express": {
        "count": int(os.environ.get("AGENT_TEAM_EXPRESS_TRAINS", "0")),
        "conductor_model": "claude-sonnet-4-5-20250929",
        "inspector_model": "claude-sonnet-4-5-20250929",
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

Use these logs to inform your decision. Look for:
- Gaps in functionality — what could the app do that it doesn't yet?
- Recurring errors or friction points users hit

IMPORTANT: Prefer proposing new features and capabilities over refactoring,
cleanup, or incremental polish of existing functionality. Bias towards what would make
users say "oh cool, it can do THAT now?" rather than small quality-of-life tweaks.
This is not an absolute rule. If the product seems feature complete, start working on polish.
If no logs are available, base your decision on the codebase alone.

Instructions:
1. Review the codebase at {working_dir} and any existing backlog items in {backlog_dir}.
2. Identify the most impactful change to build next.
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
Diff against main:
{diff}

Evaluate for: (1) correctness & error handling, (2) security, (3) spec completeness.
Only block on real issues — each CHANGES_REQUESTED costs a full conductor round-trip.
Do NOT block on style, comments, or speculative concerns.

Write feedback to: {feedback_path}
First line MUST be either "APPROVED" or "CHANGES_REQUESTED".
If requesting changes, cite specific files and line numbers. Do NOT merge.
"""

SIGNAL_PROMPT = """\
You are the Signal agent. Analyze log lines and file bug reports for new issues.

Project: {working_dir}
Open bugs (do NOT duplicate): {existing_bugs}

Log lines:
{log_lines}

If you find a NEW issue, write a JSON bug ticket to {backlog_dir}/:
  {{"title": "bug-summary", "description": "...", "priority": "high", "created_by": "signal", "working_dir": "{working_dir}"}}
  Name: {timestamp}_bug_{{summary}}.json
If already tracked or healthy, report to stdout only.
"""

STATION_MANAGER_PROMPT = """\
You are the Station Manager. Assess workflow status and report bottlenecks.

Active agents: {active_agents} | Backlog: {backlog_count}
Recent merges: {recent_merges} | Edit counts: {eng_edits}
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
