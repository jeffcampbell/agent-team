"""Optional web dashboard for the Yamanote orchestrator.

Serves a dark-themed status page and a JSON API endpoint.
Started as a daemon thread — does not block orchestrator shutdown.
"""

import base64
import glob
import json
import logging
import os
import subprocess
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

import config

_log = logging.getLogger("orchestrator")

# ── Game preview ─────────────────────────────────────────────────────────────

def _today_game_dir():
    """Return the path to today's game directory."""
    project_dir = os.path.join(config.DEVELOPMENT_DIR, config.DEFAULT_PROJECT)
    today = time.strftime("%Y-%m-%d")
    return os.path.join(project_dir, "games", today), today


def _ensure_game_export(game_dir):
    """Export game.p8 to HTML/JS if the export is missing or stale."""
    p8_path = os.path.join(game_dir, "game.p8")
    html_path = os.path.join(game_dir, "game.html")

    if not os.path.isfile(p8_path):
        return False

    # Skip if HTML is already up-to-date
    if os.path.isfile(html_path):
        if os.path.getmtime(html_path) >= os.path.getmtime(p8_path):
            return True

    # Export
    try:
        result = subprocess.run(
            ["pico8", p8_path, "-export", html_path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and os.path.isfile(html_path):
            return True
        _log.warning("PICO-8 export failed: %s", result.stderr.strip()[:200])
    except (OSError, subprocess.TimeoutExpired) as e:
        _log.warning("PICO-8 export error: %s", e)
    return False


_CLICK_TO_PLAY = b"""<style>
#p8_overlay{position:fixed;top:0;left:0;width:100%;height:100%;
background:rgba(0,0,0,.75);z-index:999;display:flex;align-items:center;
justify-content:center;cursor:pointer;flex-direction:column;gap:12px}
#p8_overlay svg{width:64px;height:64px;filter:drop-shadow(0 0 12px rgba(154,205,50,.4))}
#p8_overlay span{color:#999;font:600 11px/1 Helvetica,Arial,sans-serif;
letter-spacing:3px;text-transform:uppercase}
</style>
<div id="p8_overlay" onclick="
  this.remove();
  p8_create_audio_context();
  p8_run_cart();
"><svg viewBox="0 0 64 64"><circle cx="32" cy="32" r="30" fill="none"
stroke="#9acd32" stroke-width="2"/><polygon points="26,20 26,44 46,32"
fill="#9acd32"/></svg><span>Click to play</span></div>
</body>"""


def _patch_autoplay(data):
    """Patch PICO-8 exported HTML for click-to-play overlay in iframe embeds.

    Injects a play overlay that guarantees a user gesture before starting,
    so the AudioContext is created with full permission (audio works immediately).
    """
    data = data.replace(b"</body>", _CLICK_TO_PLAY)
    return data


def _game_dir_for_date(date_str):
    """Return the game directory path for a given YYYY-MM-DD date string."""
    project_dir = os.path.join(config.DEVELOPMENT_DIR, config.DEFAULT_PROJECT)
    return os.path.join(project_dir, "games", date_str)


def _fetch_game_preview():
    """Return metadata about today's game."""
    game_dir, today = _today_game_dir()
    p8_path = os.path.join(game_dir, "game.p8")
    html_path = os.path.join(game_dir, "game.html")
    assessment_path = os.path.join(game_dir, "assessment.md")

    result = {"date": today, "playable": False, "assessment": None, "error": None}

    if not os.path.isfile(p8_path):
        result["error"] = f"No game yet for {today}"
        return result

    # Try to ensure export exists
    result["playable"] = _ensure_game_export(game_dir)

    try:
        with open(assessment_path, "r") as f:
            result["assessment"] = f.read()
    except (OSError, FileNotFoundError):
        pass

    return result


def _list_all_games():
    """Return a list of all game dates with metadata, newest first."""
    project_dir = os.path.join(config.DEVELOPMENT_DIR, config.DEFAULT_PROJECT)
    games_root = os.path.join(project_dir, "games")
    results = []
    try:
        entries = sorted(os.listdir(games_root), reverse=True)
    except OSError:
        return results
    for entry in entries:
        # Validate YYYY-MM-DD format
        if len(entry) != 10 or entry[4] != '-' or entry[7] != '-':
            continue
        game_dir = os.path.join(games_root, entry)
        if not os.path.isdir(game_dir):
            continue
        p8_path = os.path.join(game_dir, "game.p8")
        if not os.path.isfile(p8_path):
            continue
        html_path = os.path.join(game_dir, "game.html")
        assessment_path = os.path.join(game_dir, "assessment.md")
        snippet = None
        try:
            with open(assessment_path, "r") as f:
                text = f.read().strip()
                snippet = text[:200] if text else None
        except (OSError, FileNotFoundError):
            pass
        results.append({
            "date": entry,
            "has_html": os.path.isfile(html_path),
            "assessment_snippet": snippet,
        })
    return results

# Cache HTML files at import time (zero disk I/O per request)
_DIR = os.path.dirname(os.path.abspath(__file__))
_HTML_PATH = os.path.join(_DIR, "dashboard.html")
try:
    with open(_HTML_PATH, "rb") as _f:
        _HTML_BYTES = _f.read()
except FileNotFoundError:
    _HTML_BYTES = b"<h1>dashboard.html not found</h1>"

_GAMES_HTML_PATH = os.path.join(_DIR, "games.html")
try:
    with open(_GAMES_HTML_PATH, "rb") as _f:
        _GAMES_HTML_BYTES = _f.read()
except FileNotFoundError:
    _GAMES_HTML_BYTES = b"<h1>games.html not found</h1>"


def _agent_status_dict(agent, cooldown_until, failures, now):
    """Build status dict for a single agent (train-local or global)."""
    in_cooldown = now < cooldown_until

    if agent is not None and agent.proc is not None and agent.proc.poll() is None:
        status = "running"
        pid = agent.proc.pid
        running_for = now - (agent.start_time or now)
        cooldown_rem = None
    elif in_cooldown:
        status = "cooldown"
        pid = None
        running_for = None
        cooldown_rem = cooldown_until - now
    else:
        status = "idle"
        pid = None
        running_for = None
        cooldown_rem = None

    return {
        "status": status,
        "pid": pid,
        "running_for_seconds": round(running_for, 1) if running_for is not None else None,
        "cooldown_remaining_seconds": round(cooldown_rem, 1) if cooldown_rem is not None else None,
        "consecutive_failures": failures,
    }


def _read_live_log(agent_name: str, lines: int = 150) -> list:
    """Return the last N lines from an agent's live stdout log file."""
    safe_name = agent_name.replace(":", "-").replace("/", "-")
    path = f"/tmp/yamanote-{safe_name}-live.log"
    try:
        with open(path, "r") as f:
            all_lines = f.readlines()
        return [l.rstrip("\n") for l in all_lines[-lines:]]
    except (OSError, FileNotFoundError):
        return []


def _build_status_payload(station_manager, verbose: bool = False) -> dict:
    """Snapshot mutable StationManager state into a JSON-safe dict.

    Thread safety: we copy all mutable containers at the top so the rest
    of the function operates on local, immutable snapshots.
    """
    now = time.time()

    # ── Snapshot mutable containers (GIL-safe dict()/list() copies) ──
    active_agents = dict(station_manager.active_agents)
    launch_times = list(station_manager.launch_times)
    agent_cooldowns = dict(station_manager.agent_cooldowns)
    consecutive_failures = dict(station_manager.consecutive_failures)
    last_launch_times = dict(station_manager.last_launch_times)
    trains = list(station_manager.trains)

    # ── Scalars (GIL-safe direct reads) ──
    start_time = getattr(station_manager, "start_time", now)
    sleep_until = station_manager.sleep_until

    # ── Global agents (dispatcher, signal, station_manager, ops) ──
    agents_out = {}
    for name in ("dispatcher", "signal", "station_manager", "ops"):
        agent = active_agents.get(name)
        cooldown_until = agent_cooldowns.get(name, 0)
        in_cooldown = now < cooldown_until

        if agent is not None and agent.proc is not None and agent.proc.poll() is None:
            status = "running"
            pid = agent.proc.pid
            running_for = now - (agent.start_time or now)
            cooldown_rem = None
        elif in_cooldown:
            status = "cooldown"
            pid = None
            running_for = None
            cooldown_rem = cooldown_until - now
        else:
            status = "idle"
            pid = None
            running_for = None
            cooldown_rem = None

        last_launch = last_launch_times.get(name)
        min_interval = config.AGENT_MIN_INTERVALS.get(name, 0)
        next_run = None
        if min_interval > 0 and last_launch is not None and status == "idle":
            remaining = (last_launch + min_interval) - now
            if remaining > 0:
                next_run = remaining

        agents_out[name] = {
            "status": status,
            "pid": pid,
            "running_for_seconds": round(running_for, 1) if running_for is not None else None,
            "cooldown_remaining_seconds": round(cooldown_rem, 1) if cooldown_rem is not None else None,
            "next_run_seconds": round(next_run, 1) if next_run is not None else None,
            "last_launch": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_launch)) if last_launch else None,
            "consecutive_failures": consecutive_failures.get(name, 0),
            "model": config.AGENT_MODELS.get(name, "unknown"),
        }

    # ── Trains ──
    trains_out = []
    for train in trains:
        # Determine stage
        if train.conductor is not None and train.conductor.proc is not None and train.conductor.proc.poll() is None:
            stage = "transit"
        elif train.inspector is not None and train.inspector.proc is not None and train.inspector.proc.poll() is None:
            stage = "checkpoint"
        elif train.branch and train.rework_count > 0:
            stage = "reroute"
        elif train.branch:
            stage = "checkpoint"
        else:
            stage = "idle"

        # Read spec title
        spec_title = None
        if train.spec_path:
            try:
                spec_read = train.spec_path + ".in_progress"
                if not os.path.exists(spec_read):
                    spec_read = train.spec_path
                with open(spec_read) as f:
                    spec_title = json.load(f).get("title")
            except (json.JSONDecodeError, OSError):
                spec_title = os.path.basename(train.spec_path)

        trains_out.append({
            "train_id": train.train_id,
            "train_type": train.train_type,
            "complexity": train.complexity,
            "stage": stage,
            "current_spec": spec_title,
            "current_branch": train.branch,
            "working_dir": train.working_dir,
            "rework_count": train.rework_count,
            "max_rework": config.MAX_REWORK_ATTEMPTS,
            "conductor": _agent_status_dict(train.conductor, train.conductor_cooldown_until, train.conductor_failures, now),
            "inspector": _agent_status_dict(train.inspector, train.inspector_cooldown_until, train.inspector_failures, now),
            "conductor_model": train.conductor_model,
            "inspector_model": train.inspector_model,
        })

    # ── Backward-compat: synthesize a single pipeline from the first active train ──
    pipeline_out = {"current_spec": None, "current_branch": None, "working_dir": None, "rework_count": 0, "max_rework": config.MAX_REWORK_ATTEMPTS, "stage": "idle"}
    for t in trains_out:
        if t["stage"] != "idle":
            pipeline_out = {
                "current_spec": t["current_spec"],
                "current_branch": t["current_branch"],
                "working_dir": t["working_dir"],
                "rework_count": t["rework_count"],
                "max_rework": t["max_rework"],
                "stage": t["stage"],
            }
            break

    # ── Backlog (filesystem read — read-only, safe) ──
    specs_out = []
    in_progress_count = 0
    for pattern in ("*.json", "*.json.in_progress"):
        for path in sorted(glob.glob(os.path.join(config.BACKLOG_DIR, pattern))):
            try:
                with open(path) as f:
                    data = json.load(f)
                is_ip = path.endswith(".in_progress")
                if is_ip:
                    in_progress_count += 1
                specs_out.append({
                    "filename": os.path.basename(path),
                    "title": data.get("title", "(untitled)"),
                    "description": data.get("description", ""),
                    "priority": data.get("priority", "medium"),
                    "complexity": data.get("complexity", "high"),
                    "created_by": data.get("created_by", "?"),
                })
            except (json.JSONDecodeError, OSError):
                continue
    json_only = len(sorted(glob.glob(os.path.join(config.BACKLOG_DIR, "*.json"))))

    backlog_out = {
        "count": json_only,
        "in_progress_count": in_progress_count,
        "specs": specs_out,
    }

    # ── Stats ──
    recent = [t for t in launch_times if t > now - 3600]
    sleep_active = now < sleep_until
    stats_out = {
        "launches_last_hour": len(recent),
        "max_launches_per_hour": config.MAX_AGENT_LAUNCHES_PER_HOUR,
        "sleep_mode_active": sleep_active,
        "sleep_remaining_seconds": round(sleep_until - now, 1) if sleep_active else 0,
    }

    # ── Activity log tail (filesystem read — read-only) ──
    activity_lines = []
    all_lines = []
    try:
        with open(config.ACTIVITY_LOG, "r") as f:
            all_lines = f.readlines()
            activity_lines = [l.rstrip("\n") for l in all_lines[-80:]]
    except (OSError, FileNotFoundError):
        pass

    # ── Recently completed (derived from TERMINUS/MERGED entries in activity log) ──
    completed_out = []
    for raw_line in reversed(all_lines):
        if len(completed_out) >= 20:
            break
        line = raw_line.strip()
        # Support both new "TERMINUS" and old "MERGED" keywords
        if ("TERMINUS" not in line and "MERGED" not in line) or "branch feature/" not in line:
            continue
        # Format varies:
        #   old: [TS]  TERMINUS — branch feature/title merged to trunk.
        #   new: [TS]  TERMINUS [train-id] — branch feature/title approved, merging to trunk.
        try:
            ts = line[1:20]  # "YYYY-MM-DD HH:MM:SS"
            branch_start = line.index("branch feature/") + len("branch ")
            # Try both old and new suffixes
            branch_end = -1
            for suffix in (" merged to trunk", " approved, merging to trunk"):
                try:
                    branch_end = line.index(suffix)
                    break
                except ValueError:
                    continue
            if branch_end < 0:
                continue
            branch = line[branch_start:branch_end]
            title = branch.replace("feature/", "")
            completed_out.append({"title": title, "merged_at": ts})
        except (ValueError, IndexError):
            continue

    # ── Config summary ──
    config_out = {
        "tick_interval": config.TICK_INTERVAL,
        "agent_timeout": config.AGENT_TIMEOUT_SECONDS,
        "sleep_mode_duration": config.SLEEP_MODE_DURATION,
        "max_launches_per_hour": config.MAX_AGENT_LAUNCHES_PER_HOUR,
        "entropy_threshold": config.ENTROPY_FIX_COMMIT_THRESHOLD,
        "max_rework": config.MAX_REWORK_ATTEMPTS,
        "models": dict(config.AGENT_MODELS),
        "intervals": dict(config.AGENT_MIN_INTERVALS),
        "trains": {k: v for k, v in config.TRAIN_CONFIG.items()},
    }

    # ── Verbose agent logs (only when requested) ──
    verbose_logs = {}
    if verbose:
        for name, agent_info in agents_out.items():
            if agent_info["status"] == "running":
                verbose_logs[name] = _read_live_log(name)
        for train in trains:
            if train.conductor is not None and train.conductor.proc is not None and train.conductor.proc.poll() is None:
                key = f"conductor:{train.train_id}"
                verbose_logs[key] = _read_live_log(key)
            if train.inspector is not None and train.inspector.proc is not None and train.inspector.proc.poll() is None:
                key = f"inspector:{train.train_id}"
                verbose_logs[key] = _read_live_log(key)

    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "uptime_seconds": round(now - start_time, 1),
        "agents": agents_out,
        "trains": trains_out,
        "pipeline": pipeline_out,
        "backlog": backlog_out,
        "completed": completed_out,
        "stats": stats_out,
        "activity": activity_lines,
        "config": config_out,
        "verbose_logs": verbose_logs,
    }


def _make_handler(station_manager):
    """Factory returning a request handler class bound to the given StationManager."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/api/status", "/api/status?verbose=1"):
                verbose = self.path.endswith("?verbose=1")
                payload = json.dumps(_build_status_payload(station_manager, verbose=verbose)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            elif self.path == "/api/game-preview":
                preview = _fetch_game_preview()
                payload = json.dumps(preview).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            elif self.path == "/api/games":
                games = _list_all_games()
                payload = json.dumps(games).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            elif self.path in ("/game/game.html", "/game/game.js"):
                game_dir, _ = _today_game_dir()
                filename = "game.html" if self.path.endswith(".html") else "game.js"
                filepath = os.path.join(game_dir, filename)
                _ensure_game_export(game_dir)
                if os.path.isfile(filepath):
                    with open(filepath, "rb") as f:
                        data = f.read()
                    if filename == "game.html":
                        data = _patch_autoplay(data)
                    ctype = "text/html; charset=utf-8" if filename == "game.html" else "application/javascript"
                    self.send_response(200)
                    self.send_header("Content-Type", ctype)
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self.send_response(404)
                    self.end_headers()
                    self.wfile.write(b"Game not exported yet")
            elif self.path.startswith("/game/") and self.path.count("/") == 3:
                # /game/<date>/game.html or /game/<date>/game.js
                parts = self.path.split("/")  # ['', 'game', '<date>', 'game.html|game.js']
                date_str = parts[2]
                filename = parts[3]
                # Validate date format and filename
                if (len(date_str) == 10 and date_str[4] == '-' and date_str[7] == '-'
                        and filename in ("game.html", "game.js", "game.p8")):
                    game_dir = _game_dir_for_date(date_str)
                    if filename != "game.p8":
                        _ensure_game_export(game_dir)
                    filepath = os.path.join(game_dir, filename)
                    if os.path.isfile(filepath):
                        with open(filepath, "rb") as f:
                            data = f.read()
                        if filename == "game.html":
                            data = _patch_autoplay(data)
                            ctype = "text/html; charset=utf-8"
                        elif filename == "game.p8":
                            ctype = "application/octet-stream"
                        else:
                            ctype = "application/javascript"
                        self.send_response(200)
                        self.send_header("Content-Type", ctype)
                        self.send_header("Content-Length", str(len(data)))
                        self.send_header("Cache-Control", "no-cache")
                        if filename == "game.p8":
                            self.send_header("Content-Disposition",
                                             'attachment; filename="game-' + date_str + '.p8"')
                        self.end_headers()
                        self.wfile.write(data)
                    else:
                        self.send_response(404)
                        self.end_headers()
                        self.wfile.write(b"Game not exported yet")
                else:
                    self.send_response(404)
                    self.end_headers()
            elif self.path == "/games":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(_GAMES_HTML_BYTES)))
                self.end_headers()
                self.wfile.write(_GAMES_HTML_BYTES)
            elif self.path == "/" or self.path == "/index.html":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(_HTML_BYTES)))
                self.end_headers()
                self.wfile.write(_HTML_BYTES)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            # Suppress per-request stderr logging
            pass

    return Handler


def start_dashboard(station_manager, port: int):
    """Start the dashboard HTTP server on a daemon thread.

    Logs an error and returns (without crashing) if the port is in use.
    """
    import logging
    log = logging.getLogger("orchestrator")

    try:
        handler = _make_handler(station_manager)
        server = HTTPServer(("0.0.0.0", port), handler)
    except OSError as exc:
        log.error("Dashboard failed to start on port %d: %s", port, exc)
        return

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    log.info("Dashboard running at http://0.0.0.0:%d/", port)
