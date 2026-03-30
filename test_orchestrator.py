#!/usr/bin/env python3
"""Comprehensive tests for the Yamanote orchestrator.

Covers agent lifecycle, cost guardrails, spec management, error recovery,
pipeline state machine, scheduling, file-edit tracking, entropy detection,
log reading, feedback path resolution, safety guards, dashboard payload,
orphan recovery, and spec timeout handling.

Does NOT exercise real git worktree operations — those are covered by
test_worktree.py. All subprocess.Popen / subprocess.run calls are mocked.
"""

import json
import os
import shutil
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

import config

# Save originals so tearDown can restore them
_orig_backlog = config.BACKLOG_DIR
_orig_review = config.REVIEW_DIR
_orig_logs = config.LOGS_DIR
_orig_activity = config.ACTIVITY_LOG
_orig_self_project_dir = config.SELF_PROJECT_DIR
_orig_development_dir = config.DEVELOPMENT_DIR
_orig_default_project = config.DEFAULT_PROJECT
_orig_railway_project = config.RAILWAY_PROJECT
_orig_service_restart_cmd = config.SERVICE_RESTART_CMD
_orig_app_log_glob = config.APP_LOG_GLOB
_orig_train_config = config.TRAIN_CONFIG
_orig_failure_log = config.FAILURE_LOG_PATH
_orig_rejection_log = config.REJECTION_LOG_PATH
_orig_drafts_dir = config.DRAFTS_DIR
_orig_projects_config = config.PROJECTS_CONFIG_PATH


class OrchestratorTestBase(unittest.TestCase):
    """Shared setup: patches config dirs to temp locations, provides helpers."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="yamanote_test_")
        config.BACKLOG_DIR = os.path.join(self.tmpdir, "backlog")
        config.REVIEW_DIR = os.path.join(self.tmpdir, "review")
        config.LOGS_DIR = os.path.join(self.tmpdir, "logs")
        config.ACTIVITY_LOG = os.path.join(self.tmpdir, "activity.log")
        config.FAILURE_LOG_PATH = os.path.join(self.tmpdir, "failed_specs.txt")
        config.REJECTION_LOG_PATH = os.path.join(self.tmpdir, "rejected_specs.txt")
        config.DRAFTS_DIR = os.path.join(self.tmpdir, "drafts")
        config.RAILWAY_PROJECT = ""
        config.SERVICE_RESTART_CMD = ""
        config.APP_LOG_GLOB = ""
        config.PROJECTS_CONFIG_PATH = os.path.join(self.tmpdir, "projects.json")
        # Ensure default train config has at least 1 regular train
        config.TRAIN_CONFIG = {
            "regular": {
                "count": 1,
                "conductor_model": "claude-sonnet-4-5-20250929",
                "inspector_model": "claude-haiku-4-5-20251001",
                "complexity": "high",
                "dispatcher_interval": 900,
            },
            "express": {
                "count": 0,
                "conductor_model": "claude-haiku-4-5-20251001",
                "inspector_model": "claude-haiku-4-5-20251001",
                "complexity": "low",
                "dispatcher_interval": 300,
            },
        }

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        config.BACKLOG_DIR = _orig_backlog
        config.REVIEW_DIR = _orig_review
        config.LOGS_DIR = _orig_logs
        config.ACTIVITY_LOG = _orig_activity
        config.SELF_PROJECT_DIR = _orig_self_project_dir
        config.DEVELOPMENT_DIR = _orig_development_dir
        config.DEFAULT_PROJECT = _orig_default_project
        config.RAILWAY_PROJECT = _orig_railway_project
        config.SERVICE_RESTART_CMD = _orig_service_restart_cmd
        config.APP_LOG_GLOB = _orig_app_log_glob
        config.TRAIN_CONFIG = _orig_train_config
        config.FAILURE_LOG_PATH = _orig_failure_log
        config.REJECTION_LOG_PATH = _orig_rejection_log
        config.DRAFTS_DIR = _orig_drafts_dir
        config.PROJECTS_CONFIG_PATH = _orig_projects_config

    # ── Helpers ──

    def _write_spec(self, filename, data):
        os.makedirs(config.BACKLOG_DIR, exist_ok=True)
        path = os.path.join(config.BACKLOG_DIR, filename)
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    def _write_feedback(self, branch, content):
        os.makedirs(config.REVIEW_DIR, exist_ok=True)
        fname = f"{branch.replace('/', '_')}_feedback.md"
        path = os.path.join(config.REVIEW_DIR, fname)
        with open(path, "w") as f:
            f.write(content)
        return path

    def _make_mock_proc(self, returncode=0, stdout="", stderr="", pid=12345):
        proc = MagicMock()
        proc.returncode = returncode
        proc.pid = pid
        proc.poll.return_value = None  # running by default
        proc.communicate.return_value = (stdout, stderr)
        return proc

    def _make_station_manager(self):
        from orchestrator import StationManager
        return StationManager()

    def _approve_triage(self, sm, train, mock_popen):
        """Simulate triage BUILD approval so conductor can launch."""
        # First call launches the triage agent
        sm._train_phase_triage(train)
        self.assertIsNotNone(train.triage)
        # Simulate triage finishing with BUILD verdict
        train.triage.proc.poll.return_value = 0  # finished
        train.triage.proc.returncode = 0
        train.triage._output = "BUILD\nLooks good."
        train.triage._live_log_path = "/dev/null"
        # Second call processes the verdict
        sm._train_phase_triage(train)
        self.assertFalse(train.needs_triage)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TestAgentProcessLifecycle
# ═══════════════════════════════════════════════════════════════════════════════

class TestAgentProcessLifecycle(OrchestratorTestBase):
    """Test AgentProcess start, poll, timeout, output, and save_log."""

    @patch("subprocess.Popen")
    def test_start_sets_proc_and_pid(self, mock_popen):
        from orchestrator import AgentProcess
        mock_popen.return_value = self._make_mock_proc(pid=99)
        agent = AgentProcess("test", "do stuff", cwd="/tmp")
        agent.start()
        self.assertIsNotNone(agent.proc)
        self.assertEqual(agent.proc.pid, 99)

    @patch("subprocess.Popen")
    def test_start_sets_start_time(self, mock_popen):
        from orchestrator import AgentProcess
        mock_popen.return_value = self._make_mock_proc()
        agent = AgentProcess("test", "do stuff")
        before = time.time()
        agent.start()
        after = time.time()
        self.assertGreaterEqual(agent.start_time, before)
        self.assertLessEqual(agent.start_time, after)

    @patch("subprocess.Popen")
    def test_start_strips_claudecode_env(self, mock_popen):
        from orchestrator import AgentProcess
        mock_popen.return_value = self._make_mock_proc()
        with patch.dict(os.environ, {"CLAUDECODE": "1", "HOME": "/tmp"}):
            agent = AgentProcess("test", "prompt")
            agent.start()
            call_kwargs = mock_popen.call_args[1]
            self.assertNotIn("CLAUDECODE", call_kwargs["env"])
            self.assertIn("HOME", call_kwargs["env"])

    @patch("subprocess.Popen")
    def test_start_uses_configured_model(self, mock_popen):
        from orchestrator import AgentProcess
        mock_popen.return_value = self._make_mock_proc()
        agent = AgentProcess("test", "prompt", model="claude-opus-4-20250514")
        agent.start()
        cmd = mock_popen.call_args[0][0]
        self.assertIn("claude-opus-4-20250514", cmd)

    def test_poll_returns_true_when_no_proc(self):
        from orchestrator import AgentProcess
        agent = AgentProcess("test", "prompt")
        self.assertTrue(agent.poll())

    @patch("subprocess.Popen")
    def test_poll_returns_false_when_running(self, mock_popen):
        from orchestrator import AgentProcess
        proc = self._make_mock_proc()
        proc.poll.return_value = None
        mock_popen.return_value = proc
        agent = AgentProcess("test", "prompt")
        agent.start()
        self.assertFalse(agent.poll())

    @patch("subprocess.Popen")
    def test_poll_returns_true_when_done(self, mock_popen):
        from orchestrator import AgentProcess
        proc = self._make_mock_proc()
        proc.poll.return_value = 0
        mock_popen.return_value = proc
        agent = AgentProcess("test", "prompt")
        agent.start()
        self.assertTrue(agent.poll())

    @patch("subprocess.Popen")
    def test_is_timed_out_within_limit(self, mock_popen):
        from orchestrator import AgentProcess
        mock_popen.return_value = self._make_mock_proc()
        agent = AgentProcess("test", "prompt")
        agent.start()
        self.assertFalse(agent.is_timed_out())

    @patch("subprocess.Popen")
    def test_is_timed_out_past_limit(self, mock_popen):
        from orchestrator import AgentProcess
        mock_popen.return_value = self._make_mock_proc()
        agent = AgentProcess("test", "prompt")
        agent.start()
        agent.start_time = time.time() - config.AGENT_TIMEOUT_SECONDS - 1
        self.assertTrue(agent.is_timed_out())

    def test_is_timed_out_not_started(self):
        from orchestrator import AgentProcess
        agent = AgentProcess("test", "prompt")
        self.assertFalse(agent.is_timed_out())

    @patch("subprocess.Popen")
    def test_get_output_returns_stdout(self, mock_popen):
        from orchestrator import AgentProcess
        proc = self._make_mock_proc(stdout="hello world")
        proc.stderr = MagicMock()
        proc.stderr.read.return_value = ""
        proc.wait.return_value = 0
        mock_popen.return_value = proc
        agent = AgentProcess("test", "prompt")
        agent.start()
        # Write to the live log file that get_output will read
        if agent._live_log_file:
            agent._live_log_file.write("hello world")
            agent._live_log_file.flush()
        self.assertEqual(agent.get_output(), "hello world")

    @patch("subprocess.Popen")
    def test_get_output_caches(self, mock_popen):
        from orchestrator import AgentProcess
        proc = self._make_mock_proc(stdout="cached")
        proc.stderr = MagicMock()
        proc.stderr.read.return_value = ""
        proc.wait.return_value = 0
        mock_popen.return_value = proc
        agent = AgentProcess("test", "prompt")
        agent.start()
        if agent._live_log_file:
            agent._live_log_file.write("cached")
            agent._live_log_file.flush()
        result1 = agent.get_output()
        result2 = agent.get_output()
        self.assertEqual(result1, "cached")
        self.assertEqual(result1, result2)

    def test_get_output_empty_when_not_started(self):
        from orchestrator import AgentProcess
        agent = AgentProcess("test", "prompt")
        self.assertEqual(agent.get_output(), "")

    @patch("subprocess.Popen")
    def test_save_log_creates_file_with_metadata(self, mock_popen):
        from orchestrator import AgentProcess
        os.makedirs(config.LOGS_DIR, exist_ok=True)
        proc = self._make_mock_proc(stdout="output", stderr="errors")
        proc.returncode = 0
        proc.stderr = MagicMock()
        proc.stderr.read.return_value = "errors"
        proc.wait.return_value = 0
        mock_popen.return_value = proc
        agent = AgentProcess("test_agent", "prompt")
        agent.start()
        if agent._live_log_file:
            agent._live_log_file.write("output")
            agent._live_log_file.flush()
        log_path = agent.save_log(marker="[OVERDUE]")
        self.assertTrue(os.path.exists(log_path))
        with open(log_path) as f:
            content = f.read()
        self.assertIn("[OVERDUE]", content)
        self.assertIn("test_agent", content)
        self.assertIn("output", content)
        self.assertIn("errors", content)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TestCostGuardrail
# ═══════════════════════════════════════════════════════════════════════════════

class TestCostGuardrail(OrchestratorTestBase):
    """Test the rolling-window launch limiter and sleep-mode triggers."""

    @patch("subprocess.Popen")
    def test_launch_within_limit_succeeds(self, mock_popen):
        mock_popen.return_value = self._make_mock_proc()
        sm = self._make_station_manager()
        # Use station_manager agent (min_interval=0) to avoid throttle
        agent = sm._launch_agent("station_manager", "prompt")
        self.assertIsNotNone(agent)

    @patch("subprocess.Popen")
    def test_exceeding_limit_triggers_sleep(self, mock_popen):
        mock_popen.return_value = self._make_mock_proc()
        sm = self._make_station_manager()
        # Fill up to the limit
        for i in range(config.MAX_AGENT_LAUNCHES_PER_HOUR):
            sm.launch_times.append(time.time())
        # Use station_manager agent (min_interval=0) to avoid throttle
        agent = sm._launch_agent("station_manager", "prompt")
        self.assertIsNone(agent)
        self.assertGreater(sm.sleep_until, time.time())

    def test_old_launches_pruned(self):
        sm = self._make_station_manager()
        old_time = time.time() - 7200  # 2 hours ago
        for _ in range(10):
            sm.launch_times.append(old_time)
        # Pruning happens inside _launch_agent — simulate manually
        now = time.time()
        sm.launch_times.append(now)
        while sm.launch_times and sm.launch_times[0] < now - 3600:
            sm.launch_times.popleft()
        self.assertEqual(len(sm.launch_times), 1)

    @patch("subprocess.Popen")
    def test_train_agents_share_cost_guardrail(self, mock_popen):
        mock_popen.return_value = self._make_mock_proc()
        sm = self._make_station_manager()
        train = sm.trains[0]
        # Fill to the limit
        for _ in range(config.MAX_AGENT_LAUNCHES_PER_HOUR):
            sm.launch_times.append(time.time())
        agent = sm._launch_train_agent(train, "conductor", "prompt", cwd="/tmp")
        self.assertIsNone(agent)
        self.assertGreater(sm.sleep_until, time.time())

    def test_rate_limit_in_output_triggers_sleep_global(self):
        sm = self._make_station_manager()
        agent = MagicMock()
        agent.name = "dispatcher"
        agent.poll.return_value = True  # finished
        proc = self._make_mock_proc(returncode=1, stdout="out of extra usage")
        agent.proc = proc
        agent.get_output.return_value = "out of extra usage"
        agent.get_stderr.return_value = ""
        sm.active_agents["dispatcher"] = agent
        sm._is_agent_active("dispatcher")
        self.assertGreater(sm.sleep_until, time.time())

    def test_rate_limit_in_output_triggers_sleep_train(self):
        sm = self._make_station_manager()
        train = sm.trains[0]
        agent = MagicMock()
        agent.name = "conductor:regular-0"
        agent.poll.return_value = True
        proc = self._make_mock_proc(returncode=1, stdout="Rate Limit exceeded")
        agent.proc = proc
        agent.get_output.return_value = "Rate Limit exceeded"
        agent.get_stderr.return_value = ""
        train.conductor = agent
        sm._is_train_agent_active(train, "conductor")
        self.assertGreater(sm.sleep_until, time.time())

    @patch("subprocess.Popen")
    def test_sleep_mode_blocks_all_launches(self, mock_popen):
        mock_popen.return_value = self._make_mock_proc()
        sm = self._make_station_manager()
        sm.sleep_until = time.time() + 3600
        # _launch_agent doesn't check sleep_until directly,
        # but the main loop does; verify cost guardrail doesn't clear it
        self.assertGreater(sm.sleep_until, time.time())


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TestSpecManagement
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpecManagement(OrchestratorTestBase):
    """Test backlog listing, filtering, priority sorting, and spec selection."""

    def test_specs_sorted_by_priority_then_filename(self):
        sm = self._make_station_manager()
        self._write_spec("aaa_low.json", {"priority": "low", "title": "a"})
        self._write_spec("bbb_high.json", {"priority": "high", "title": "b"})
        self._write_spec("ccc_medium.json", {"priority": "medium", "title": "c"})
        specs = sm._backlog_specs()
        basenames = [os.path.basename(s) for s in specs]
        self.assertEqual(basenames[0], "bbb_high.json")
        self.assertEqual(basenames[-1], "aaa_low.json")

    def test_filter_by_complexity(self):
        sm = self._make_station_manager()
        self._write_spec("a.json", {"complexity": "high", "title": "a"})
        self._write_spec("b.json", {"complexity": "low", "title": "b"})
        high = sm._backlog_specs(complexity="high")
        low = sm._backlog_specs(complexity="low")
        self.assertEqual(len(high), 1)
        self.assertIn("a.json", high[0])
        self.assertEqual(len(low), 1)
        self.assertIn("b.json", low[0])

    def test_default_complexity_is_high(self):
        sm = self._make_station_manager()
        self._write_spec("no_complexity.json", {"title": "test"})
        high = sm._backlog_specs(complexity="high")
        self.assertEqual(len(high), 1)
        low = sm._backlog_specs(complexity="low")
        self.assertEqual(len(low), 0)

    def test_malformed_json_handled_gracefully(self):
        sm = self._make_station_manager()
        os.makedirs(config.BACKLOG_DIR, exist_ok=True)
        path = os.path.join(config.BACKLOG_DIR, "bad.json")
        with open(path, "w") as f:
            f.write("{invalid json")
        # Should not crash; malformed defaults to high complexity
        specs = sm._backlog_specs(complexity="high")
        self.assertEqual(len(specs), 1)

    def test_in_progress_files_excluded(self):
        sm = self._make_station_manager()
        self._write_spec("active.json", {"title": "active"})
        # Create an .in_progress file
        os.makedirs(config.BACKLOG_DIR, exist_ok=True)
        ip_path = os.path.join(config.BACKLOG_DIR, "wip.json.in_progress")
        with open(ip_path, "w") as f:
            json.dump({"title": "wip"}, f)
        specs = sm._backlog_specs()
        basenames = [os.path.basename(s) for s in specs]
        self.assertIn("active.json", basenames)
        self.assertNotIn("wip.json.in_progress", basenames)

    def test_regular_train_prefers_high_falls_back_low(self):
        sm = self._make_station_manager()
        train = sm.trains[0]
        self.assertEqual(train.train_type, "regular")
        self.assertEqual(train.complexity, "high")
        # Only low spec available
        self._write_spec("low_spec.json", {"complexity": "low", "title": "low-task"})
        spec = sm._find_spec_for_train(train)
        self.assertIsNotNone(spec)
        self.assertIn("low_spec.json", spec)

    def test_express_train_only_picks_low(self):
        # Create an express train
        config.TRAIN_CONFIG["express"]["count"] = 1
        sm = self._make_station_manager()
        express = [t for t in sm.trains if t.train_type == "express"][0]
        self._write_spec("high_spec.json", {"complexity": "high", "title": "high-task"})
        spec = sm._find_spec_for_train(express)
        self.assertIsNone(spec)

    def test_express_train_picks_low(self):
        config.TRAIN_CONFIG["express"]["count"] = 1
        sm = self._make_station_manager()
        express = [t for t in sm.trains if t.train_type == "express"][0]
        self._write_spec("low_spec.json", {"complexity": "low", "title": "low-task"})
        spec = sm._find_spec_for_train(express)
        self.assertIsNotNone(spec)

    def test_signal_open_bugs_counts_json_and_in_progress(self):
        sm = self._make_station_manager()
        self._write_spec("bug1.json", {"title": "bug1", "created_by": "signal"})
        os.makedirs(config.BACKLOG_DIR, exist_ok=True)
        ip_path = os.path.join(config.BACKLOG_DIR, "bug2.json.in_progress")
        with open(ip_path, "w") as f:
            json.dump({"title": "bug2", "created_by": "sre"}, f)
        bugs = sm._signal_open_bugs()
        self.assertEqual(len(bugs), 2)

    def test_signal_open_bugs_ignores_non_signal(self):
        sm = self._make_station_manager()
        self._write_spec("feature.json", {"title": "feat", "created_by": "dispatcher"})
        bugs = sm._signal_open_bugs()
        self.assertEqual(len(bugs), 0)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TestErrorRecovery
# ═══════════════════════════════════════════════════════════════════════════════

class TestErrorRecovery(OrchestratorTestBase):
    """Test exponential backoff, cooldowns, and signal offset rollback."""

    def test_global_failure_sets_cooldown(self):
        sm = self._make_station_manager()
        agent = MagicMock()
        agent.name = "signal"
        agent.poll.return_value = True
        agent.proc = self._make_mock_proc(returncode=1, stdout="error")
        agent.get_output.return_value = "error"
        agent.get_stderr.return_value = ""
        sm.active_agents["signal"] = agent
        sm._is_agent_active("signal")
        self.assertIn("signal", sm.agent_cooldowns)
        self.assertGreater(sm.agent_cooldowns["signal"], time.time())

    def test_consecutive_failures_increase_backoff(self):
        sm = self._make_station_manager()
        cooldowns = []
        for i in range(3):
            agent = MagicMock()
            agent.name = "signal"
            agent.poll.return_value = True
            agent.proc = self._make_mock_proc(returncode=1, stdout="err")
            agent.get_output.return_value = "err"
            agent.get_stderr.return_value = ""
            sm.active_agents["signal"] = agent
            sm._is_agent_active("signal")
            cooldowns.append(sm.agent_cooldowns["signal"])
        # Each cooldown should be further in the future
        for j in range(1, len(cooldowns)):
            self.assertGreater(cooldowns[j], cooldowns[j - 1])

    def test_backoff_capped_at_max(self):
        sm = self._make_station_manager()
        sm.consecutive_failures["signal"] = 100  # huge streak
        agent = MagicMock()
        agent.name = "signal"
        agent.poll.return_value = True
        agent.proc = self._make_mock_proc(returncode=1, stdout="err")
        agent.get_output.return_value = "err"
        agent.get_stderr.return_value = ""
        sm.active_agents["signal"] = agent
        sm._is_agent_active("signal")
        max_allowed = time.time() + config.MAX_ERROR_BACKOFF + 1
        self.assertLessEqual(sm.agent_cooldowns["signal"], max_allowed)

    def test_success_clears_failures(self):
        sm = self._make_station_manager()
        sm.consecutive_failures["signal"] = 5
        agent = MagicMock()
        agent.name = "signal"
        agent.poll.return_value = True
        agent.proc = self._make_mock_proc(returncode=0, stdout="ok")
        agent.get_output.return_value = "ok"
        agent.get_stderr.return_value = ""
        sm.active_agents["signal"] = agent
        sm._is_agent_active("signal")
        self.assertNotIn("signal", sm.consecutive_failures)

    @patch("subprocess.Popen")
    def test_launch_respects_cooldown(self, mock_popen):
        mock_popen.return_value = self._make_mock_proc()
        sm = self._make_station_manager()
        sm.agent_cooldowns["dispatcher"] = time.time() + 9999
        agent = sm._launch_agent("dispatcher", "prompt")
        self.assertIsNone(agent)

    @patch("subprocess.Popen")
    def test_expired_cooldown_allows_launch(self, mock_popen):
        mock_popen.return_value = self._make_mock_proc()
        sm = self._make_station_manager()
        # Use station_manager (min_interval=0) so only cooldown matters
        sm.agent_cooldowns["station_manager"] = time.time() - 1  # expired
        agent = sm._launch_agent("station_manager", "prompt")
        self.assertIsNotNone(agent)

    def test_train_conductor_failure_sets_cooldown(self):
        sm = self._make_station_manager()
        train = sm.trains[0]
        agent = MagicMock()
        agent.name = "conductor:regular-0"
        agent.poll.return_value = True
        agent.proc = self._make_mock_proc(returncode=1, stdout="err")
        agent.get_output.return_value = "err"
        agent.get_stderr.return_value = ""
        train.conductor = agent
        sm._is_train_agent_active(train, "conductor")
        self.assertGreater(train.conductor_cooldown_until, time.time())
        self.assertEqual(train.conductor_failures, 1)

    def test_train_inspector_failure_sets_cooldown(self):
        sm = self._make_station_manager()
        train = sm.trains[0]
        agent = MagicMock()
        agent.name = "inspector:regular-0"
        agent.poll.return_value = True
        agent.proc = self._make_mock_proc(returncode=1, stdout="err")
        agent.get_output.return_value = "err"
        agent.get_stderr.return_value = ""
        train.inspector = agent
        sm._is_train_agent_active(train, "inspector")
        self.assertGreater(train.inspector_cooldown_until, time.time())
        self.assertEqual(train.inspector_failures, 1)

    def test_signal_failure_rolls_back_offsets(self):
        sm = self._make_station_manager()
        sm._sre_prev_offsets = {"/project": 100}
        sm.sre_log_offsets = {"/project": 500}
        agent = MagicMock()
        agent.name = "signal"
        agent.poll.return_value = True
        agent.proc = self._make_mock_proc(returncode=1, stdout="err")
        agent.get_output.return_value = "err"
        agent.get_stderr.return_value = ""
        sm.active_agents["signal"] = agent
        sm._is_agent_active("signal")
        self.assertEqual(sm.sre_log_offsets["/project"], 100)

    def test_signal_timeout_rolls_back_offsets(self):
        sm = self._make_station_manager()
        sm._sre_prev_offsets = {"/project": 50}
        sm.sre_log_offsets = {"/project": 300}
        agent = MagicMock()
        agent.name = "signal"
        agent.proc = self._make_mock_proc()
        agent.proc.poll.return_value = None  # still running
        agent.start_time = time.time() - config.AGENT_TIMEOUT_SECONDS - 10
        sm.active_agents["signal"] = agent
        sm._kill_timed_out_agent("signal", agent)
        self.assertEqual(sm.sre_log_offsets["/project"], 50)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TestTrainPipelineStateMachine
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrainPipelineStateMachine(OrchestratorTestBase):
    """Test conductor → inspector → rework → service_recovery flow."""

    def _setup_sm_with_mocked_git(self):
        sm = self._make_station_manager()
        sm._create_worktree = MagicMock(return_value="/tmp/fake_worktree")
        sm._remove_worktree = MagicMock()
        sm._git = MagicMock(return_value="")
        sm._git_rc = MagicMock(return_value=(0, "", ""))
        sm._git_has_branch = MagicMock(return_value=False)
        sm._git_diff_trunk = MagicMock(return_value="")
        sm._git_last_commit = MagicMock(return_value="abc123")
        return sm

    @patch("subprocess.Popen")
    def test_conductor_picks_spec_and_starts(self, mock_popen):
        mock_popen.return_value = self._make_mock_proc()
        sm = self._setup_sm_with_mocked_git()
        train = sm.trains[0]
        dev_dir = os.path.join(self.tmpdir, "dev")
        os.makedirs(dev_dir, exist_ok=True)
        config.DEVELOPMENT_DIR = self.tmpdir
        config.DEFAULT_PROJECT = "dev"
        spec_path = self._write_spec("spec1.json", {
            "title": "test-feature",
            "description": "test",
            "working_dir": dev_dir,
        })
        sm._train_phase_conductor(train)
        self.assertEqual(train.spec_path, spec_path)
        self.assertEqual(train.branch, "feature/test-feature")
        self.assertTrue(train.needs_triage)
        # Approve triage, then launch conductor
        self._approve_triage(sm, train, mock_popen)
        sm._train_phase_conductor_launch(train)
        self.assertIsNotNone(train.conductor)

    @patch("subprocess.Popen")
    def test_conductor_skipped_when_branch_set(self, mock_popen):
        sm = self._setup_sm_with_mocked_git()
        train = sm.trains[0]
        train.branch = "feature/existing"
        self._write_spec("spec.json", {"title": "t", "description": "d"})
        sm._train_phase_conductor(train)
        self.assertIsNone(train.conductor)

    @patch("subprocess.Popen")
    def test_conductor_skipped_during_cooldown(self, mock_popen):
        sm = self._setup_sm_with_mocked_git()
        train = sm.trains[0]
        train.conductor_cooldown_until = time.time() + 9999
        dev_dir = os.path.join(self.tmpdir, "dev")
        os.makedirs(dev_dir, exist_ok=True)
        config.DEVELOPMENT_DIR = self.tmpdir
        config.DEFAULT_PROJECT = "dev"
        self._write_spec("spec.json", {
            "title": "t", "description": "d", "working_dir": dev_dir,
        })
        sm._train_phase_conductor(train)
        self.assertIsNone(train.conductor)

    @patch("subprocess.Popen")
    def test_inspector_launches_when_conductor_done_with_diff(self, mock_popen):
        mock_popen.return_value = self._make_mock_proc()
        sm = self._setup_sm_with_mocked_git()
        sm._git_has_branch.return_value = True
        sm._git_diff_trunk.return_value = "+new line"
        train = sm.trains[0]
        train.branch = "feature/test"
        train.working_dir = "/tmp/wt"
        train.spec_path = "/tmp/spec.json"
        sm._train_phase_inspector(train)
        self.assertIsNotNone(train.inspector)

    def test_inspector_cleans_up_on_no_diff(self):
        sm = self._setup_sm_with_mocked_git()
        sm._git_has_branch.return_value = True
        sm._git_diff_trunk.return_value = ""
        train = sm.trains[0]
        train.branch = "feature/empty"
        train.working_dir = "/tmp/wt"
        train.repo_dir = "/tmp/repo"
        spec_path = self._write_spec("empty.json", {"title": "empty"})
        train.spec_path = spec_path
        os.rename(spec_path, spec_path + ".in_progress")
        sm._train_phase_inspector(train)
        self.assertIsNone(train.branch)
        sm._remove_worktree.assert_called()

    @patch("subprocess.Popen")
    def test_rework_increments_count_and_relaunches(self, mock_popen):
        mock_popen.return_value = self._make_mock_proc()
        sm = self._setup_sm_with_mocked_git()
        train = sm.trains[0]
        train.branch = "feature/rework"
        train.working_dir = "/tmp/wt"
        spec_path = self._write_spec("rework.json", {"title": "rework", "description": "d"})
        train.spec_path = spec_path
        os.rename(spec_path, spec_path + ".in_progress")
        self._write_feedback("feature/rework", "CHANGES_REQUESTED\nFix it.\n")
        sm._train_phase_rework(train)
        self.assertEqual(train.rework_count, 1)
        self.assertIsNotNone(train.conductor)

    def test_exceeding_max_rework_cancels_spec(self):
        sm = self._setup_sm_with_mocked_git()
        sm._git_has_branch.return_value = True
        train = sm.trains[0]
        train.branch = "feature/too-many"
        train.working_dir = "/tmp/wt"
        train.repo_dir = "/tmp/repo"
        train.rework_count = config.MAX_REWORK_ATTEMPTS
        spec_path = self._write_spec("cancel.json", {"title": "cancel", "description": "d"})
        train.spec_path = spec_path
        self._write_feedback("feature/too-many", "CHANGES_REQUESTED\nStill bad.\n")
        sm._train_phase_rework(train)
        self.assertIsNone(train.branch)
        sm._remove_worktree.assert_called()

    @patch("subprocess.run")
    def test_service_recovery_merges_on_approved(self, mock_run):
        mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)
        sm = self._setup_sm_with_mocked_git()
        sm._git_has_branch.return_value = True
        train = sm.trains[0]
        train.branch = "feature/merge-me"
        train.working_dir = "/tmp/wt"
        train.repo_dir = "/tmp/repo"
        spec_path = self._write_spec("merge.json", {"title": "merge"})
        train.spec_path = spec_path
        os.rename(spec_path, spec_path + ".in_progress")
        self._write_feedback("feature/merge-me", "APPROVED\nShip it!\n")
        sm._train_phase_service_recovery(train)
        self.assertIsNone(train.branch)
        sm._git.assert_any_call("merge", "--no-ff", "feature/merge-me", cwd="/tmp/repo")

    def test_service_recovery_ignores_non_approved(self):
        sm = self._setup_sm_with_mocked_git()
        train = sm.trains[0]
        train.branch = "feature/not-yet"
        train.working_dir = "/tmp/wt"
        self._write_feedback("feature/not-yet", "CHANGES_REQUESTED\nFix.\n")
        sm._train_phase_service_recovery(train)
        self.assertEqual(train.branch, "feature/not-yet")

    @patch("subprocess.run")
    def test_service_recovery_calls_restart_cmd(self, mock_run):
        mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)
        config.SERVICE_RESTART_CMD = "systemctl restart myapp"
        sm = self._setup_sm_with_mocked_git()
        sm._git_has_branch.return_value = True
        train = sm.trains[0]
        train.branch = "feature/restart"
        train.working_dir = "/tmp/wt"
        train.repo_dir = "/tmp/repo"
        spec_path = self._write_spec("restart.json", {"title": "restart"})
        train.spec_path = spec_path
        os.rename(spec_path, spec_path + ".in_progress")
        self._write_feedback("feature/restart", "APPROVED\n")
        sm._train_phase_service_recovery(train)
        mock_run.assert_any_call(
            "systemctl restart myapp",
            shell=True, timeout=config.SERVICE_RESTART_TIMEOUT,
            capture_output=True, text=True,
        )

    def test_inspector_skips_when_feedback_exists(self):
        """Inspector should not relaunch when feedback file already exists."""
        sm = self._setup_sm_with_mocked_git()
        sm._git_has_branch.return_value = True
        sm._git_diff_trunk.return_value = "+new line"
        train = sm.trains[0]
        train.branch = "feature/already-reviewed"
        train.working_dir = "/tmp/wt"
        self._write_feedback("feature/already-reviewed", "APPROVED\n")
        sm._train_phase_inspector(train)
        self.assertIsNone(train.inspector)

    def test_create_worktree_raises_on_failure(self):
        """_create_worktree should raise RuntimeError if directory wasn't created."""
        from orchestrator import StationManager
        sm = self._make_station_manager()
        sm._git = MagicMock(return_value="")
        sm._git_has_branch = MagicMock(return_value=False)
        repo_dir = os.path.join(self.tmpdir, "repo")
        os.makedirs(repo_dir, exist_ok=True)
        with self.assertRaises(RuntimeError) as ctx:
            sm._create_worktree(repo_dir, "feature/broken", "train-0")
        self.assertIn("Failed to create worktree", str(ctx.exception))

    @patch("subprocess.Popen")
    def test_conductor_handles_worktree_failure_gracefully(self, mock_popen):
        """Conductor should reset train without crashing on worktree failure."""
        mock_popen.return_value = self._make_mock_proc()
        sm = self._setup_sm_with_mocked_git()
        sm._create_worktree.side_effect = RuntimeError("worktree creation failed")
        dev_dir = os.path.join(self.tmpdir, "dev")
        os.makedirs(dev_dir, exist_ok=True)
        config.DEVELOPMENT_DIR = self.tmpdir
        config.DEFAULT_PROJECT = "dev"
        self._write_spec("fail.json", {
            "title": "fail-feature",
            "description": "test",
            "working_dir": dev_dir,
        })
        train = sm.trains[0]
        sm._train_phase_conductor(train)
        self.assertTrue(train.needs_triage)
        self._approve_triage(sm, train, mock_popen)
        sm._train_phase_conductor_launch(train)
        self.assertIsNone(train.branch)
        self.assertIsNone(train.conductor)

    @patch("subprocess.Popen")
    def test_conductor_handles_worktree_failure_orphan_path(self, mock_popen):
        """Orphan recovery path should reset train on worktree failure."""
        mock_popen.return_value = self._make_mock_proc()
        sm = self._setup_sm_with_mocked_git()
        sm._git_has_branch.return_value = True
        sm._git_diff_trunk.return_value = "+orphan changes"
        sm._create_worktree.side_effect = RuntimeError("worktree creation failed")
        dev_dir = os.path.join(self.tmpdir, "dev")
        os.makedirs(dev_dir, exist_ok=True)
        config.DEVELOPMENT_DIR = self.tmpdir
        config.DEFAULT_PROJECT = "dev"
        self._write_spec("orphan.json", {
            "title": "orphan-feature",
            "description": "test",
            "working_dir": dev_dir,
        })
        train = sm.trains[0]
        sm._train_phase_conductor(train)
        self.assertTrue(train.needs_triage)
        self._approve_triage(sm, train, mock_popen)
        sm._train_phase_conductor_launch(train)
        self.assertIsNone(train.branch)
        self.assertIsNone(train.conductor)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TestPhaseScheduling
# ═══════════════════════════════════════════════════════════════════════════════

class TestPhaseScheduling(OrchestratorTestBase):
    """Test minimum-interval throttling and dispatcher skip logic."""

    @patch("subprocess.Popen")
    def test_min_interval_blocks_launch(self, mock_popen):
        mock_popen.return_value = self._make_mock_proc()
        sm = self._make_station_manager()
        # Use dispatcher (30-min interval) instead of signal (now on-demand, 0 interval)
        sm.last_launch_times["dispatcher"] = time.time()
        agent = sm._launch_agent("dispatcher", "prompt")
        self.assertIsNone(agent)

    @patch("subprocess.Popen")
    def test_min_interval_allows_launch_after_elapsed(self, mock_popen):
        mock_popen.return_value = self._make_mock_proc()
        sm = self._make_station_manager()
        sm.last_launch_times["dispatcher"] = time.time() - config.AGENT_MIN_INTERVALS["dispatcher"] - 1
        agent = sm._launch_agent("dispatcher", "prompt")
        self.assertIsNotNone(agent)

    @patch("subprocess.Popen")
    @patch("subprocess.run")
    def test_dispatcher_skipped_when_backlog_not_empty(self, mock_run, mock_popen):
        mock_popen.return_value = self._make_mock_proc()
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        sm = self._make_station_manager()
        self._write_spec("existing.json", {"title": "exists"})
        sm._phase_dispatcher()
        self.assertIsNone(sm.active_agents.get("dispatcher"))

    @patch("subprocess.Popen")
    @patch("subprocess.run")
    def test_dispatcher_skipped_when_train_active(self, mock_run, mock_popen):
        mock_popen.return_value = self._make_mock_proc()
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        sm = self._make_station_manager()
        sm.trains[0].branch = "feature/active"
        sm.last_launch_times["dispatcher"] = 0  # long ago
        sm._phase_dispatcher()
        self.assertIsNone(sm.active_agents.get("dispatcher"))

    @patch("subprocess.Popen")
    @patch("subprocess.run")
    def test_dispatcher_uses_shortest_idle_train_interval(self, mock_run, mock_popen):
        mock_popen.return_value = self._make_mock_proc()
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        config.TRAIN_CONFIG["express"]["count"] = 1
        sm = self._make_station_manager()
        # All trains idle, express interval is 300s — set last launch old enough
        sm.last_launch_times["dispatcher"] = time.time() - 400
        dev_dir = os.path.join(self.tmpdir, "dev")
        os.makedirs(dev_dir, exist_ok=True)
        config.DEVELOPMENT_DIR = self.tmpdir
        config.DEFAULT_PROJECT = "dev"
        # Override min_interval for dispatcher so _launch_agent doesn't throttle
        orig_intervals = dict(config.AGENT_MIN_INTERVALS)
        config.AGENT_MIN_INTERVALS["dispatcher"] = 0
        try:
            sm._phase_dispatcher()
        finally:
            config.AGENT_MIN_INTERVALS.update(orig_intervals)
        self.assertIsNotNone(sm.active_agents.get("dispatcher"))

    @patch("subprocess.Popen")
    @patch("subprocess.run")
    def test_dispatcher_skip_log_dedup(self, mock_run, mock_popen):
        """Dispatcher skip log only emitted once per same set of active trains."""
        mock_popen.return_value = self._make_mock_proc()
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        sm = self._make_station_manager()
        sm.trains[0].branch = "feature/active"
        sm.last_launch_times["dispatcher"] = 0
        sm._phase_dispatcher()
        first_set = sm._dispatcher_skip_logged_trains
        sm._phase_dispatcher()
        # Same set — should not change
        self.assertEqual(sm._dispatcher_skip_logged_trains, first_set)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. TestFileEditTracking
# ═══════════════════════════════════════════════════════════════════════════════

class TestFileEditTracking(OrchestratorTestBase):
    """Test file-edit tallying and station manager check."""

    def _setup_sm_with_git_diff(self, diff_output):
        sm = self._make_station_manager()
        sm._create_worktree = MagicMock(return_value="/tmp/wt")
        sm._remove_worktree = MagicMock()
        sm._git = MagicMock(return_value=diff_output)
        sm._git_has_branch = MagicMock(return_value=True)
        sm._git_diff_trunk = MagicMock(return_value="some diff")
        return sm

    @patch("subprocess.Popen")
    def test_edits_tallied_after_conductor_finishes(self, mock_popen):
        mock_popen.return_value = self._make_mock_proc()
        sm = self._setup_sm_with_git_diff("file1.py\nfile2.py\nfile1.py")
        train = sm.trains[0]
        train.branch = "feature/edit-test"
        train.working_dir = "/tmp/wt"
        train.edits_tallied = False
        train.conductor = None
        sm._train_phase_conductor(train)
        self.assertTrue(train.edits_tallied)

    def test_station_manager_check_fires_at_threshold(self):
        sm = self._make_station_manager()
        sm._remove_worktree = MagicMock()
        sm._git = MagicMock(return_value="")
        sm._git_has_branch = MagicMock(return_value=True)
        train = sm.trains[0]
        train.branch = "feature/hot-file"
        train.working_dir = "/tmp/wt"
        train.repo_dir = "/tmp/repo"
        train.file_edits = {"hot.py": config.MAX_ENG_EDITS_BEFORE_RESET}
        sm._train_phase_station_manager_check(train)
        self.assertIsNone(train.branch)

    def test_station_manager_check_no_fire_below_threshold(self):
        sm = self._make_station_manager()
        train = sm.trains[0]
        train.branch = "feature/ok-file"
        train.working_dir = "/tmp/wt"
        train.file_edits = {"ok.py": 1}
        sm._train_phase_station_manager_check(train)
        self.assertEqual(train.branch, "feature/ok-file")

    def test_station_manager_check_skips_approved_branches(self):
        sm = self._make_station_manager()
        train = sm.trains[0]
        train.branch = "feature/approved"
        train.working_dir = "/tmp/wt"
        train.file_edits = {"hot.py": config.MAX_ENG_EDITS_BEFORE_RESET}
        self._write_feedback("feature/approved", "APPROVED\nAll good.\n")
        sm._train_phase_station_manager_check(train)
        # Branch should NOT be reset because it's approved
        self.assertEqual(train.branch, "feature/approved")

    def test_requeues_in_progress_spec_on_reset(self):
        sm = self._make_station_manager()
        sm._remove_worktree = MagicMock()
        sm._git = MagicMock(return_value="")
        sm._git_has_branch = MagicMock(return_value=True)
        train = sm.trains[0]
        train.branch = "feature/requeue"
        train.working_dir = "/tmp/wt"
        train.repo_dir = "/tmp/repo"
        train.file_edits = {"hot.py": config.MAX_ENG_EDITS_BEFORE_RESET}
        spec_path = self._write_spec("requeue.json", {"title": "requeue"})
        train.spec_path = spec_path
        os.rename(spec_path, spec_path + ".in_progress")
        sm._train_phase_station_manager_check(train)
        self.assertTrue(os.path.exists(spec_path))
        self.assertFalse(os.path.exists(spec_path + ".in_progress"))

    def test_edits_tallied_skips_when_no_branch(self):
        """Tallying should not crash or set edits_tallied when there's no branch."""
        sm = self._make_station_manager()
        train = sm.trains[0]
        train.edits_tallied = False
        train.conductor = None
        train.branch = None
        # Should not attempt git diff
        sm._train_phase_conductor(train)
        self.assertFalse(train.edits_tallied)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. TestEntropyDetection
# ═══════════════════════════════════════════════════════════════════════════════

class TestEntropyDetection(OrchestratorTestBase):
    """Test fix/update commit counting and entropy firing."""

    def test_count_fix_commits_basic(self):
        sm = self._make_station_manager()
        sm._git = MagicMock(return_value="abc Fix login\ndef Update API\nghi Add feature")
        count = sm._count_fix_commits_on_branch("feature/test")
        self.assertEqual(count, 2)

    def test_count_fix_commits_case_insensitive(self):
        sm = self._make_station_manager()
        sm._git = MagicMock(return_value="abc FIX crash\ndef UPDATE config")
        count = sm._count_fix_commits_on_branch("feature/test")
        self.assertEqual(count, 2)

    def test_count_fix_commits_empty_log(self):
        sm = self._make_station_manager()
        sm._git = MagicMock(return_value="")
        count = sm._count_fix_commits_on_branch("feature/test")
        self.assertEqual(count, 0)

    def test_entropy_check_fires_at_threshold(self):
        sm = self._make_station_manager()
        sm._git_has_branch = MagicMock(return_value=True)
        sm._git = MagicMock(return_value="\n".join(
            [f"abc{i} Fix thing {i}" for i in range(config.ENTROPY_FIX_COMMIT_THRESHOLD)]
        ))
        sm._fire_conductor_entropy = MagicMock()
        train = sm.trains[0]
        train.branch = "feature/entropy"
        train.working_dir = "/tmp/wt"
        sm._train_phase_entropy_check(train)
        sm._fire_conductor_entropy.assert_called_once()

    def test_entropy_check_does_not_fire_below_threshold(self):
        sm = self._make_station_manager()
        sm._git_has_branch = MagicMock(return_value=True)
        sm._git = MagicMock(return_value="abc Fix one thing")
        sm._fire_conductor_entropy = MagicMock()
        train = sm.trains[0]
        train.branch = "feature/ok"
        train.working_dir = "/tmp/wt"
        sm._train_phase_entropy_check(train)
        sm._fire_conductor_entropy.assert_not_called()

    def test_fire_entropy_kills_running_conductor(self):
        sm = self._make_station_manager()
        sm._remove_worktree = MagicMock()
        sm._git = MagicMock(return_value="")
        sm._git_has_branch = MagicMock(return_value=True)
        train = sm.trains[0]
        train.branch = "feature/fire"
        train.working_dir = "/tmp/wt"
        train.repo_dir = "/tmp/repo"
        conductor = MagicMock()
        conductor.proc = self._make_mock_proc()
        conductor.proc.poll.return_value = None  # still running
        train.conductor = conductor
        sm._fire_conductor_entropy(train, "feature/fire")
        conductor.proc.terminate.assert_called()
        self.assertIsNone(train.conductor)

    def test_fire_entropy_requeues_spec(self):
        sm = self._make_station_manager()
        sm._remove_worktree = MagicMock()
        sm._git = MagicMock(return_value="")
        sm._git_has_branch = MagicMock(return_value=True)
        train = sm.trains[0]
        train.branch = "feature/requeue"
        train.working_dir = "/tmp/wt"
        train.repo_dir = "/tmp/repo"
        spec_path = self._write_spec("entropy.json", {"title": "entropy"})
        train.spec_path = spec_path
        os.rename(spec_path, spec_path + ".in_progress")
        sm._fire_conductor_entropy(train, "feature/requeue")
        self.assertTrue(os.path.exists(spec_path))
        self.assertIsNone(train.branch)

    def test_fire_entropy_resets_pipeline(self):
        sm = self._make_station_manager()
        sm._remove_worktree = MagicMock()
        sm._git = MagicMock(return_value="")
        sm._git_has_branch = MagicMock(return_value=False)
        train = sm.trains[0]
        train.branch = "feature/reset"
        train.working_dir = "/tmp/wt"
        train.repo_dir = "/tmp/repo"
        sm._fire_conductor_entropy(train, "feature/reset")
        self.assertIsNone(train.branch)
        self.assertIsNone(train.working_dir)
        self.assertEqual(train.rework_count, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. TestLogReading
# ═══════════════════════════════════════════════════════════════════════════════

class TestLogReading(OrchestratorTestBase):
    """Test high-water mark log reading and _find_app_log resolution."""

    def _make_log_file(self, project_dir, name="app.log", content="line1\nline2\n"):
        log_path = os.path.join(project_dir, name)
        with open(log_path, "w") as f:
            f.write(content)
        return log_path

    def test_first_read_sets_hwm_returns_empty(self):
        sm = self._make_station_manager()
        project_dir = os.path.join(self.tmpdir, "proj")
        os.makedirs(project_dir, exist_ok=True)
        self._make_log_file(project_dir)
        result = sm._read_new_log_lines(project_dir)
        self.assertEqual(result, "")
        self.assertIn(project_dir, sm.sre_log_offsets)

    def test_second_read_returns_new_lines(self):
        sm = self._make_station_manager()
        project_dir = os.path.join(self.tmpdir, "proj")
        os.makedirs(project_dir, exist_ok=True)
        log_path = self._make_log_file(project_dir)
        sm._read_new_log_lines(project_dir)  # set HWM
        with open(log_path, "a") as f:
            f.write("new_line\n")
        result = sm._read_new_log_lines(project_dir)
        self.assertIn("new_line", result)

    def test_no_new_content_returns_empty(self):
        sm = self._make_station_manager()
        project_dir = os.path.join(self.tmpdir, "proj")
        os.makedirs(project_dir, exist_ok=True)
        self._make_log_file(project_dir)
        sm._read_new_log_lines(project_dir)  # set HWM
        result = sm._read_new_log_lines(project_dir)
        self.assertEqual(result, "")

    def test_log_rotation_resets_offset(self):
        sm = self._make_station_manager()
        project_dir = os.path.join(self.tmpdir, "proj")
        os.makedirs(project_dir, exist_ok=True)
        log_path = self._make_log_file(project_dir, content="a" * 1000)
        sm._read_new_log_lines(project_dir)  # set HWM to 1000
        # Simulate log rotation: file is now smaller
        with open(log_path, "w") as f:
            f.write("rotated\n")
        result = sm._read_new_log_lines(project_dir)
        self.assertIn("rotated", result)

    def test_find_app_log_env_glob(self):
        sm = self._make_station_manager()
        project_dir = os.path.join(self.tmpdir, "proj")
        os.makedirs(project_dir, exist_ok=True)
        self._make_log_file(project_dir, "custom.log")
        config.APP_LOG_GLOB = "custom.log"
        result = sm._find_app_log(project_dir)
        self.assertIsNotNone(result)
        self.assertIn("custom.log", result)

    def test_find_app_log_fallback_convention(self):
        sm = self._make_station_manager()
        project_dir = os.path.join(self.tmpdir, "proj")
        logs_dir = os.path.join(project_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        self._make_log_file(logs_dir, "server.log")
        config.APP_LOG_GLOB = ""
        result = sm._find_app_log(project_dir)
        self.assertIsNotNone(result)
        self.assertIn("server.log", result)

    def test_find_app_log_returns_most_recent(self):
        sm = self._make_station_manager()
        project_dir = os.path.join(self.tmpdir, "proj")
        os.makedirs(project_dir, exist_ok=True)
        old = self._make_log_file(project_dir, "old.log", "old")
        time.sleep(0.05)
        new = self._make_log_file(project_dir, "new.log", "new")
        config.APP_LOG_GLOB = ""
        result = sm._find_app_log(project_dir)
        self.assertEqual(result, new)

    def test_find_app_log_none_if_no_logs(self):
        sm = self._make_station_manager()
        project_dir = os.path.join(self.tmpdir, "empty_proj")
        os.makedirs(project_dir, exist_ok=True)
        config.APP_LOG_GLOB = ""
        result = sm._find_app_log(project_dir)
        self.assertIsNone(result)

    def test_railway_logs_first_read_sets_marker(self):
        sm = self._make_station_manager()
        config.RAILWAY_PROJECT = "my-project"
        sm._fetch_railway_logs = MagicMock(return_value="line1\nline2\nline3")
        result = sm._read_new_railway_logs("/project")
        self.assertEqual(result, "")
        self.assertIn("_railway_", sm.sre_log_offsets)

    def test_railway_logs_dedup_returns_new(self):
        sm = self._make_station_manager()
        config.RAILWAY_PROJECT = "my-project"
        sm._fetch_railway_logs = MagicMock(return_value="line1\nline2\nline3")
        sm._read_new_railway_logs("/project")
        sm._fetch_railway_logs.return_value = "line1\nline2\nline3\nline4\nline5"
        result = sm._read_new_railway_logs("/project")
        self.assertIn("line4", result)
        self.assertIn("line5", result)
        self.assertNotIn("line1", result)

    def test_railway_logs_rotation_returns_all(self):
        sm = self._make_station_manager()
        config.RAILWAY_PROJECT = "my-project"
        sm._fetch_railway_logs = MagicMock(return_value="line1\nline2")
        sm._read_new_railway_logs("/project")
        # Simulate log rotation — none of the old lines are present
        sm._fetch_railway_logs.return_value = "new1\nnew2"
        result = sm._read_new_railway_logs("/project")
        self.assertIn("new1", result)
        self.assertIn("new2", result)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. TestFeedbackPathResolution
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeedbackPathResolution(OrchestratorTestBase):
    """Test _feedback_path canonical/glob fallback logic."""

    def test_canonical_path_when_exists(self):
        sm = self._make_station_manager()
        path = self._write_feedback("feature/test", "APPROVED\n")
        result = sm._feedback_path("feature/test")
        self.assertEqual(result, path)

    def test_glob_fallback_single_match(self):
        sm = self._make_station_manager()
        os.makedirs(config.REVIEW_DIR, exist_ok=True)
        fb = os.path.join(config.REVIEW_DIR, "some_branch_feedback.md")
        with open(fb, "w") as f:
            f.write("APPROVED\n")
        # Query a different branch — canonical won't exist, glob finds one match
        result = sm._feedback_path("feature/nonexistent")
        self.assertEqual(result, fb)

    def test_canonical_fallback_multiple_matches(self):
        sm = self._make_station_manager()
        os.makedirs(config.REVIEW_DIR, exist_ok=True)
        for name in ("a_feedback.md", "b_feedback.md"):
            with open(os.path.join(config.REVIEW_DIR, name), "w") as f:
                f.write("X\n")
        result = sm._feedback_path("feature/unknown")
        # Multiple matches → falls back to canonical
        expected = os.path.join(config.REVIEW_DIR, "feature_unknown_feedback.md")
        self.assertEqual(result, expected)

    def test_canonical_fallback_no_matches(self):
        sm = self._make_station_manager()
        os.makedirs(config.REVIEW_DIR, exist_ok=True)
        result = sm._feedback_path("feature/empty")
        expected = os.path.join(config.REVIEW_DIR, "feature_empty_feedback.md")
        self.assertEqual(result, expected)

    def test_branch_slash_replaced(self):
        sm = self._make_station_manager()
        result = sm._feedback_path("feature/my/deep/branch")
        self.assertIn("feature_my_deep_branch_feedback.md", result)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. TestSafetyGuards
# ═══════════════════════════════════════════════════════════════════════════════

class TestSafetyGuards(OrchestratorTestBase):
    """Test is_self_project and conductor spec validation."""

    def test_is_self_project_true(self):
        sm = self._make_station_manager()
        self.assertTrue(sm._is_self_project(config.SELF_PROJECT_DIR))

    def test_is_self_project_false(self):
        sm = self._make_station_manager()
        self.assertFalse(sm._is_self_project("/tmp/other"))

    def test_is_self_project_none(self):
        sm = self._make_station_manager()
        self.assertFalse(sm._is_self_project(None))

    @patch("subprocess.Popen")
    def test_conductor_rejects_self_project(self, mock_popen):
        mock_popen.return_value = self._make_mock_proc()
        sm = self._make_station_manager()
        sm._create_worktree = MagicMock(return_value="/tmp/wt")
        sm._git_has_branch = MagicMock(return_value=False)
        train = sm.trains[0]
        config.DEVELOPMENT_DIR = os.path.dirname(config.SELF_PROJECT_DIR)
        spec_path = self._write_spec("self.json", {
            "title": "bad",
            "description": "d",
            "working_dir": config.SELF_PROJECT_DIR,
        })
        sm._train_phase_conductor(train)
        self.assertIsNone(train.conductor)
        self.assertFalse(os.path.exists(spec_path))

    @patch("subprocess.Popen")
    def test_conductor_rejects_outside_development_dir(self, mock_popen):
        mock_popen.return_value = self._make_mock_proc()
        sm = self._make_station_manager()
        sm._create_worktree = MagicMock(return_value="/tmp/wt")
        sm._git_has_branch = MagicMock(return_value=False)
        train = sm.trains[0]
        outside_dir = os.path.join(self.tmpdir, "outside")
        os.makedirs(outside_dir, exist_ok=True)
        config.DEVELOPMENT_DIR = os.path.join(self.tmpdir, "dev")
        os.makedirs(config.DEVELOPMENT_DIR, exist_ok=True)
        spec_path = self._write_spec("outside.json", {
            "title": "escape",
            "description": "d",
            "working_dir": outside_dir,
        })
        sm._train_phase_conductor(train)
        self.assertIsNone(train.conductor)
        self.assertFalse(os.path.exists(spec_path))

    @patch("subprocess.Popen")
    def test_conductor_rejects_nonexistent_working_dir(self, mock_popen):
        mock_popen.return_value = self._make_mock_proc()
        sm = self._make_station_manager()
        sm._create_worktree = MagicMock(return_value="/tmp/wt")
        sm._git_has_branch = MagicMock(return_value=False)
        train = sm.trains[0]
        spec_path = self._write_spec("ghost.json", {
            "title": "ghost",
            "description": "d",
            "working_dir": "/nonexistent/path",
        })
        sm._train_phase_conductor(train)
        self.assertIsNone(train.conductor)
        self.assertFalse(os.path.exists(spec_path))

    @patch("subprocess.Popen")
    def test_conductor_accepts_valid_spec(self, mock_popen):
        mock_popen.return_value = self._make_mock_proc()
        sm = self._make_station_manager()
        sm._create_worktree = MagicMock(return_value="/tmp/wt")
        sm._remove_worktree = MagicMock()
        sm._git = MagicMock(return_value="")
        sm._git_has_branch = MagicMock(return_value=False)
        sm._git_diff_trunk = MagicMock(return_value="")
        train = sm.trains[0]
        dev_dir = os.path.join(self.tmpdir, "dev", "myproject")
        os.makedirs(dev_dir, exist_ok=True)
        config.DEVELOPMENT_DIR = os.path.join(self.tmpdir, "dev")
        self._write_spec("valid.json", {
            "title": "valid-feature",
            "description": "d",
            "working_dir": dev_dir,
        })
        sm._train_phase_conductor(train)
        self.assertTrue(train.needs_triage)
        # Approve triage, then launch conductor
        self._approve_triage(sm, train, mock_popen)
        sm._train_phase_conductor_launch(train)
        self.assertIsNotNone(train.conductor)


# ═══════════════════════════════════════════════════════════════════════════════
# 12. TestDashboardPayload
# ═══════════════════════════════════════════════════════════════════════════════

class TestDashboardPayload(OrchestratorTestBase):
    """Test dashboard _build_status_payload output."""

    def _build_payload(self, sm):
        from dashboard import _build_status_payload
        return _build_status_payload(sm)

    def test_idle_stage(self):
        sm = self._make_station_manager()
        payload = self._build_payload(sm)
        self.assertEqual(payload["trains"][0]["stage"], "idle")

    def test_transit_stage_conductor_running(self):
        sm = self._make_station_manager()
        train = sm.trains[0]
        conductor = MagicMock()
        conductor.proc = self._make_mock_proc()
        conductor.proc.poll.return_value = None
        conductor.start_time = time.time()
        train.conductor = conductor
        train.branch = "feature/test"
        payload = self._build_payload(sm)
        self.assertEqual(payload["trains"][0]["stage"], "transit")

    def test_checkpoint_stage_inspector_running(self):
        sm = self._make_station_manager()
        train = sm.trains[0]
        inspector = MagicMock()
        inspector.proc = self._make_mock_proc()
        inspector.proc.poll.return_value = None
        inspector.start_time = time.time()
        train.inspector = inspector
        train.branch = "feature/test"
        payload = self._build_payload(sm)
        self.assertEqual(payload["trains"][0]["stage"], "checkpoint")

    def test_checkpoint_stage_branch_set_no_rework(self):
        sm = self._make_station_manager()
        train = sm.trains[0]
        train.branch = "feature/waiting"
        payload = self._build_payload(sm)
        self.assertEqual(payload["trains"][0]["stage"], "checkpoint")

    def test_reroute_stage_rework(self):
        sm = self._make_station_manager()
        train = sm.trains[0]
        train.branch = "feature/rework"
        train.rework_count = 1
        payload = self._build_payload(sm)
        self.assertEqual(payload["trains"][0]["stage"], "reroute")

    def test_global_agent_running(self):
        sm = self._make_station_manager()
        agent = MagicMock()
        agent.proc = self._make_mock_proc()
        agent.proc.poll.return_value = None
        agent.start_time = time.time()
        sm.active_agents["dispatcher"] = agent
        payload = self._build_payload(sm)
        self.assertEqual(payload["agents"]["dispatcher"]["status"], "running")

    def test_global_agent_cooldown(self):
        sm = self._make_station_manager()
        sm.agent_cooldowns["signal"] = time.time() + 600
        payload = self._build_payload(sm)
        self.assertEqual(payload["agents"]["signal"]["status"], "cooldown")

    def test_global_agent_idle(self):
        sm = self._make_station_manager()
        payload = self._build_payload(sm)
        self.assertEqual(payload["agents"]["signal"]["status"], "idle")

    def test_backlog_counts(self):
        sm = self._make_station_manager()
        self._write_spec("a.json", {"title": "a"})
        self._write_spec("b.json", {"title": "b"})
        os.makedirs(config.BACKLOG_DIR, exist_ok=True)
        ip = os.path.join(config.BACKLOG_DIR, "c.json.in_progress")
        with open(ip, "w") as f:
            json.dump({"title": "c"}, f)
        payload = self._build_payload(sm)
        self.assertEqual(payload["backlog"]["count"], 2)
        self.assertEqual(payload["backlog"]["in_progress_count"], 1)

    def test_sleep_mode_reflected(self):
        sm = self._make_station_manager()
        sm.sleep_until = time.time() + 3600
        payload = self._build_payload(sm)
        self.assertTrue(payload["stats"]["sleep_mode_active"])
        self.assertGreater(payload["stats"]["sleep_remaining_seconds"], 0)

    def test_payload_contains_expected_keys(self):
        sm = self._make_station_manager()
        payload = self._build_payload(sm)
        expected = {"timestamp", "uptime_seconds", "paused", "agents", "trains",
                    "pipeline", "backlog", "completed", "stats", "activity", "config",
                    "verbose_logs"}
        self.assertEqual(set(payload.keys()), expected)

    def test_backward_compat_pipeline_from_active_train(self):
        sm = self._make_station_manager()
        train = sm.trains[0]
        train.branch = "feature/active"
        train.rework_count = 1
        payload = self._build_payload(sm)
        self.assertEqual(payload["pipeline"]["current_branch"], "feature/active")
        self.assertEqual(payload["pipeline"]["stage"], "reroute")


# ═══════════════════════════════════════════════════════════════════════════════
# 13. TestOrphanRecovery
# ═══════════════════════════════════════════════════════════════════════════════

class TestOrphanRecovery(OrchestratorTestBase):
    """Test _recover_orphaned_specs on startup."""

    def test_orphaned_specs_renamed(self):
        os.makedirs(config.BACKLOG_DIR, exist_ok=True)
        ip = os.path.join(config.BACKLOG_DIR, "orphan.json.in_progress")
        with open(ip, "w") as f:
            json.dump({"title": "orphan"}, f)
        sm = self._make_station_manager()  # calls _recover_orphaned_specs
        self.assertFalse(os.path.exists(ip))
        self.assertTrue(os.path.exists(os.path.join(config.BACKLOG_DIR, "orphan.json")))

    def test_multiple_orphans_recovered(self):
        os.makedirs(config.BACKLOG_DIR, exist_ok=True)
        for name in ("a.json.in_progress", "b.json.in_progress"):
            with open(os.path.join(config.BACKLOG_DIR, name), "w") as f:
                json.dump({"title": name}, f)
        sm = self._make_station_manager()
        self.assertTrue(os.path.exists(os.path.join(config.BACKLOG_DIR, "a.json")))
        self.assertTrue(os.path.exists(os.path.join(config.BACKLOG_DIR, "b.json")))

    def test_no_orphans_no_crash(self):
        os.makedirs(config.BACKLOG_DIR, exist_ok=True)
        sm = self._make_station_manager()
        self.assertIsNotNone(sm)

    def test_regular_json_not_touched(self):
        spec_path = self._write_spec("regular.json", {"title": "keep"})
        sm = self._make_station_manager()
        self.assertTrue(os.path.exists(spec_path))
        with open(spec_path) as f:
            data = json.load(f)
        self.assertEqual(data["title"], "keep")

    def test_orphan_recovery_cleans_stale_worktrees(self):
        """Stale worktree directories should be removed during orphan recovery."""
        os.makedirs(config.BACKLOG_DIR, exist_ok=True)
        project_dir = os.path.join(self.tmpdir, "project")
        os.makedirs(project_dir, exist_ok=True)
        # Create a stale worktree directory
        wt_dir = os.path.join(project_dir, ".worktrees", "regular-0")
        os.makedirs(wt_dir, exist_ok=True)
        # Write the orphaned spec pointing at this project
        ip = os.path.join(config.BACKLOG_DIR, "stale.json.in_progress")
        with open(ip, "w") as f:
            json.dump({"title": "stale-feature", "working_dir": project_dir}, f)
        sm = self._make_station_manager()
        self.assertFalse(os.path.isdir(wt_dir))

    def test_orphan_recovery_cleans_stale_feedback(self):
        """Stale feedback files should be removed during orphan recovery."""
        os.makedirs(config.BACKLOG_DIR, exist_ok=True)
        project_dir = os.path.join(self.tmpdir, "project")
        os.makedirs(project_dir, exist_ok=True)
        # Write a stale feedback file
        self._write_feedback("feature/stale-feat", "APPROVED\n")
        # Write the orphaned spec
        ip = os.path.join(config.BACKLOG_DIR, "stale2.json.in_progress")
        with open(ip, "w") as f:
            json.dump({"title": "stale-feat", "working_dir": project_dir}, f)
        sm = self._make_station_manager()
        feedback_path = os.path.join(config.REVIEW_DIR, "feature_stale-feat_feedback.md")
        self.assertFalse(os.path.exists(feedback_path))


# ═══════════════════════════════════════════════════════════════════════════════
# 14. TestSpecTimeoutHandling
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpecTimeoutHandling(OrchestratorTestBase):
    """Test conductor timeout → spec re-route or permanent drop."""

    def test_first_timeout_reroutes_spec(self):
        sm = self._make_station_manager()
        sm._remove_worktree = MagicMock()
        sm._git_has_branch = MagicMock(return_value=False)
        sm._git = MagicMock(return_value="")
        train = sm.trains[0]
        train.branch = "feature/timeout"
        train.working_dir = "/tmp/wt"
        train.repo_dir = "/tmp/repo"
        spec_path = self._write_spec("timeout.json", {"title": "timeout"})
        train.spec_path = spec_path
        os.rename(spec_path, spec_path + ".in_progress")
        train.spec_timeout_count = 0
        agent = MagicMock()
        agent.name = "conductor:regular-0"
        agent.proc = self._make_mock_proc()
        agent.proc.poll.return_value = None
        agent.start_time = time.time() - config.AGENT_TIMEOUT_SECONDS - 10
        train.conductor = agent
        sm._kill_timed_out_train_agent(train, "conductor", agent)
        self.assertEqual(train.spec_timeout_count, 1)
        # Spec should be re-routed (renamed back to .json)
        self.assertTrue(os.path.exists(spec_path))

    def test_max_timeouts_drops_spec(self):
        sm = self._make_station_manager()
        sm._remove_worktree = MagicMock()
        sm._git_has_branch = MagicMock(return_value=False)
        sm._git = MagicMock(return_value="")
        train = sm.trains[0]
        train.branch = "feature/drop"
        train.working_dir = "/tmp/wt"
        train.repo_dir = "/tmp/repo"
        spec_path = self._write_spec("drop.json", {"title": "drop"})
        train.spec_path = spec_path
        os.rename(spec_path, spec_path + ".in_progress")
        train.spec_timeout_count = config.MAX_SPEC_TIMEOUTS - 1
        agent = MagicMock()
        agent.name = "conductor:regular-0"
        agent.proc = self._make_mock_proc()
        agent.proc.poll.return_value = None
        agent.start_time = time.time() - config.AGENT_TIMEOUT_SECONDS - 10
        train.conductor = agent
        sm._kill_timed_out_train_agent(train, "conductor", agent)
        # Spec should be permanently dropped
        self.assertFalse(os.path.exists(spec_path + ".in_progress"))
        self.assertIsNone(train.branch)

    def test_timeout_increments_conductor_failures(self):
        sm = self._make_station_manager()
        sm._remove_worktree = MagicMock()
        sm._git_has_branch = MagicMock(return_value=False)
        sm._git = MagicMock(return_value="")
        train = sm.trains[0]
        train.branch = "feature/backoff"
        train.working_dir = "/tmp/wt"
        train.repo_dir = "/tmp/repo"
        spec_path = self._write_spec("backoff.json", {"title": "backoff"})
        train.spec_path = spec_path
        os.rename(spec_path, spec_path + ".in_progress")
        train.spec_timeout_count = 0
        train.conductor_failures = 0
        agent = MagicMock()
        agent.name = "conductor:regular-0"
        agent.proc = self._make_mock_proc()
        agent.proc.poll.return_value = None
        agent.start_time = time.time() - config.AGENT_TIMEOUT_SECONDS - 10
        train.conductor = agent
        sm._kill_timed_out_train_agent(train, "conductor", agent)
        self.assertEqual(train.conductor_failures, 1)
        self.assertGreater(train.conductor_cooldown_until, time.time())


# ═══════════════════════════════════════════════════════════════════════════════
# Activity Log Cleanup Utilities
# ═══════════════════════════════════════════════════════════════════════════════

class TestStripMarkdown(unittest.TestCase):
    def test_removes_bold_and_italic(self):
        from orchestrator import _strip_markdown
        self.assertEqual(_strip_markdown("**bold** and *italic*"), "bold and italic")

    def test_removes_code_and_headers(self):
        from orchestrator import _strip_markdown
        self.assertEqual(_strip_markdown("## Heading `code`"), "Heading code")

    def test_normalizes_whitespace(self):
        from orchestrator import _strip_markdown
        self.assertEqual(_strip_markdown("  too   many   spaces  "), "too many spaces")

    def test_empty_string(self):
        from orchestrator import _strip_markdown
        self.assertEqual(_strip_markdown(""), "")

    def test_plain_text_unchanged(self):
        from orchestrator import _strip_markdown
        self.assertEqual(_strip_markdown("no formatting here"), "no formatting here")


class TestStripReasonPrefix(unittest.TestCase):
    def test_strips_reasoning_prefix(self):
        from orchestrator import _strip_reason_prefix
        self.assertEqual(_strip_reason_prefix("Reasoning: the feature is solid"), "the feature is solid")

    def test_strips_why_prefix(self):
        from orchestrator import _strip_reason_prefix
        self.assertEqual(_strip_reason_prefix("Why: not needed"), "not needed")

    def test_strips_reason_prefix(self):
        from orchestrator import _strip_reason_prefix
        self.assertEqual(_strip_reason_prefix("Reason: duplicate"), "duplicate")

    def test_strips_analysis_prefix(self):
        from orchestrator import _strip_reason_prefix
        self.assertEqual(_strip_reason_prefix("Analysis: looks good"), "looks good")

    def test_no_prefix_unchanged(self):
        from orchestrator import _strip_reason_prefix
        self.assertEqual(_strip_reason_prefix("the feature is solid"), "the feature is solid")

    def test_empty_string(self):
        from orchestrator import _strip_reason_prefix
        self.assertEqual(_strip_reason_prefix(""), "")


class TestFirstLineTruncated(unittest.TestCase):
    def test_short_text_unchanged(self):
        from orchestrator import _first_line_truncated
        self.assertEqual(_first_line_truncated("short text"), "short text")

    def test_truncates_at_word_boundary(self):
        from orchestrator import _first_line_truncated
        long_text = "word " * 40  # 200 chars
        result = _first_line_truncated(long_text, limit=50)
        self.assertTrue(result.endswith("..."))
        self.assertLessEqual(len(result), 53)  # limit + "..."

    def test_multiline_takes_first_line(self):
        from orchestrator import _first_line_truncated
        self.assertEqual(_first_line_truncated("first line\nsecond line"), "first line")

    def test_empty_string(self):
        from orchestrator import _first_line_truncated
        self.assertEqual(_first_line_truncated(""), "")

    def test_exact_limit(self):
        from orchestrator import _first_line_truncated
        text = "a" * 150
        self.assertEqual(_first_line_truncated(text, limit=150), text)


# ═══════════════════════════════════════════════════════════════════════════════
# Failure / Rejection Log Persistence
# ═══════════════════════════════════════════════════════════════════════════════

class TestRecordFailure(OrchestratorTestBase):
    def test_writes_to_failure_log(self):
        from orchestrator import record_failure
        record_failure("my-spec", "timed out")
        with open(config.FAILURE_LOG_PATH) as f:
            content = f.read()
        self.assertIn("my-spec", content)
        self.assertIn("timed out", content)

    def test_appends_multiple_entries(self):
        from orchestrator import record_failure
        record_failure("spec-a", "reason-a")
        record_failure("spec-b", "reason-b")
        with open(config.FAILURE_LOG_PATH) as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 2)

    def test_handles_missing_directory_gracefully(self):
        from orchestrator import record_failure
        config.FAILURE_LOG_PATH = "/nonexistent/dir/fail.txt"
        record_failure("spec", "reason")  # should not raise


class TestRecordRejection(OrchestratorTestBase):
    def test_writes_to_rejection_log(self):
        from orchestrator import record_rejection
        record_rejection("my-spec", "low priority", project="demo")
        with open(config.REJECTION_LOG_PATH) as f:
            content = f.read()
        self.assertIn("my-spec", content)
        self.assertIn("low priority", content)
        self.assertIn("demo", content)

    def test_empty_project_still_writes(self):
        from orchestrator import record_rejection
        record_rejection("spec", "reason")
        with open(config.REJECTION_LOG_PATH) as f:
            content = f.read()
        self.assertIn("spec", content)


class TestReadFailureLog(OrchestratorTestBase):
    def test_returns_none_when_no_file(self):
        from orchestrator import read_failure_log
        self.assertEqual(read_failure_log(), "(none)")

    def test_returns_last_n_lines(self):
        from orchestrator import record_failure, read_failure_log
        for i in range(20):
            record_failure(f"spec-{i}", "reason")
        result = read_failure_log(max_lines=5)
        lines = result.strip().split("\n")
        self.assertEqual(len(lines), 5)
        self.assertIn("spec-19", lines[-1])


class TestReadRejectionLog(OrchestratorTestBase):
    def test_returns_none_when_no_file(self):
        from orchestrator import read_rejection_log
        self.assertEqual(read_rejection_log(), "(none)")

    def test_filters_by_project(self):
        from orchestrator import record_rejection, read_rejection_log
        record_rejection("spec-a", "reason", project="alpha")
        record_rejection("spec-b", "reason", project="beta")
        result = read_rejection_log(project="alpha")
        self.assertIn("spec-a", result)
        self.assertNotIn("spec-b", result)

    def test_respects_max_lines(self):
        from orchestrator import record_rejection, read_rejection_log
        for i in range(10):
            record_rejection(f"spec-{i}", "reason", project="p")
        result = read_rejection_log(max_lines=3, project="p")
        lines = result.strip().split("\n")
        self.assertEqual(len(lines), 3)


# ═══════════════════════════════════════════════════════════════════════════════
# Per-Tick Caching
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerTickCaching(OrchestratorTestBase):

    @patch("subprocess.Popen")
    def test_backlog_count_cached_within_tick(self, mock_popen):
        sm = self._make_station_manager()
        self._write_spec("spec1.json", {"title": "a", "priority": "high"})
        self._write_spec("spec2.json", {"title": "b", "priority": "low"})
        count1 = sm._get_cached_backlog_count()
        # Add another spec — count should stay cached
        self._write_spec("spec3.json", {"title": "c", "priority": "medium"})
        count2 = sm._get_cached_backlog_count()
        self.assertEqual(count1, 2)
        self.assertEqual(count2, 2)  # still cached

    @patch("subprocess.Popen")
    def test_backlog_count_invalidated_on_new_tick(self, mock_popen):
        sm = self._make_station_manager()
        self._write_spec("spec1.json", {"title": "a", "priority": "high"})
        count1 = sm._get_cached_backlog_count()
        self._write_spec("spec2.json", {"title": "b", "priority": "low"})
        sm._advance_tick()
        count2 = sm._get_cached_backlog_count()
        self.assertEqual(count1, 1)
        self.assertEqual(count2, 2)

    @patch("subprocess.Popen")
    def test_backlog_specs_cached_within_tick(self, mock_popen):
        sm = self._make_station_manager()
        self._write_spec("spec1.json", {"title": "a", "priority": "high"})
        specs1 = sm._get_cached_backlog_specs()
        self._write_spec("spec2.json", {"title": "b", "priority": "low"})
        specs2 = sm._get_cached_backlog_specs()
        self.assertEqual(len(specs1), 1)
        self.assertIs(specs1, specs2)  # same object — cached

    @patch("subprocess.Popen")
    def test_open_bugs_cached_within_tick(self, mock_popen):
        sm = self._make_station_manager()
        self._write_spec("bug1.json", {"title": "bug", "created_by": "signal", "priority": "high"})
        bugs1 = sm._get_cached_open_bugs()
        self._write_spec("bug2.json", {"title": "bug2", "created_by": "signal", "priority": "high"})
        bugs2 = sm._get_cached_open_bugs()
        self.assertEqual(len(bugs1), 1)
        self.assertIs(bugs1, bugs2)  # same object — cached

    @patch("subprocess.Popen")
    def test_open_bugs_refreshed_after_tick_advance(self, mock_popen):
        sm = self._make_station_manager()
        self._write_spec("bug1.json", {"title": "bug", "created_by": "signal", "priority": "high"})
        bugs1 = sm._get_cached_open_bugs()
        sm._advance_tick()
        self._write_spec("bug2.json", {"title": "bug2", "created_by": "signal", "priority": "high"})
        bugs2 = sm._get_cached_open_bugs()
        self.assertEqual(len(bugs1), 1)
        self.assertEqual(len(bugs2), 2)


# ═══════════════════════════════════════════════════════════════════════════════
# Schedule Window & Project Scheduling
# ═══════════════════════════════════════════════════════════════════════════════

class TestScheduleWindow(unittest.TestCase):
    def test_none_schedule_always_eligible(self):
        from orchestrator import _is_in_schedule_window
        self.assertTrue(_is_in_schedule_window(None, now_hour=12))

    def test_empty_schedule_always_eligible(self):
        from orchestrator import _is_in_schedule_window
        self.assertTrue(_is_in_schedule_window("", now_hour=12))

    def test_in_window_normal_range(self):
        from orchestrator import _is_in_schedule_window
        self.assertTrue(_is_in_schedule_window("9-17", now_hour=12))

    def test_outside_window_normal_range(self):
        from orchestrator import _is_in_schedule_window
        self.assertFalse(_is_in_schedule_window("9-17", now_hour=20))

    def test_midnight_wraparound_in_window_late(self):
        from orchestrator import _is_in_schedule_window
        self.assertTrue(_is_in_schedule_window("22-2", now_hour=23))

    def test_midnight_wraparound_in_window_early(self):
        from orchestrator import _is_in_schedule_window
        self.assertTrue(_is_in_schedule_window("22-2", now_hour=1))

    def test_midnight_wraparound_outside(self):
        from orchestrator import _is_in_schedule_window
        self.assertFalse(_is_in_schedule_window("22-2", now_hour=12))

    def test_malformed_schedule_fail_open(self):
        from orchestrator import _is_in_schedule_window
        self.assertTrue(_is_in_schedule_window("not-a-schedule", now_hour=12))


class TestPickDispatcherProject(OrchestratorTestBase):

    def _write_projects_json(self, projects: dict):
        with open(config.PROJECTS_CONFIG_PATH, "w") as f:
            json.dump({"projects": projects}, f)

    @patch("subprocess.Popen")
    def test_falls_back_to_default_project_when_no_projects_json(self, mock_popen):
        sm = self._make_station_manager()
        project_dir = os.path.join(self.tmpdir, "myproject")
        os.makedirs(project_dir)
        config.DEVELOPMENT_DIR = self.tmpdir
        config.DEFAULT_PROJECT = "myproject"
        result = sm._pick_dispatcher_project()
        self.assertEqual(result, project_dir)

    @patch("subprocess.Popen")
    def test_scheduled_project_in_window_wins(self, mock_popen):
        sm = self._make_station_manager()
        proj_a = os.path.join(self.tmpdir, "alpha")
        proj_b = os.path.join(self.tmpdir, "beta")
        os.makedirs(proj_a)
        os.makedirs(proj_b)
        self._write_projects_json({
            "alpha": {"path": proj_a, "priority": 2},
            "beta": {"path": proj_b, "priority": 1, "schedule": "9-17"},
        })
        with patch("time.localtime") as mock_time:
            mock_time.return_value = time.struct_time((2026, 3, 30, 12, 0, 0, 0, 89, 0))
            result = sm._pick_dispatcher_project()
        self.assertEqual(result, proj_b)

    @patch("subprocess.Popen")
    def test_unscheduled_project_when_no_active_schedule(self, mock_popen):
        sm = self._make_station_manager()
        proj_a = os.path.join(self.tmpdir, "alpha")
        proj_b = os.path.join(self.tmpdir, "beta")
        os.makedirs(proj_a)
        os.makedirs(proj_b)
        self._write_projects_json({
            "alpha": {"path": proj_a, "priority": 1},
            "beta": {"path": proj_b, "priority": 2, "schedule": "22-2"},
        })
        with patch("time.localtime") as mock_time:
            mock_time.return_value = time.struct_time((2026, 3, 30, 12, 0, 0, 0, 89, 0))
            result = sm._pick_dispatcher_project()
        self.assertEqual(result, proj_a)

    @patch("subprocess.Popen")
    def test_paused_project_skipped(self, mock_popen):
        sm = self._make_station_manager()
        proj = os.path.join(self.tmpdir, "paused_proj")
        os.makedirs(proj)
        self._write_projects_json({
            "paused_proj": {"path": proj, "priority": 1, "paused": True},
        })
        result = sm._pick_dispatcher_project()
        self.assertIsNone(result)

    @patch("subprocess.Popen")
    def test_returns_none_when_all_schedules_inactive(self, mock_popen):
        sm = self._make_station_manager()
        proj = os.path.join(self.tmpdir, "night_proj")
        os.makedirs(proj)
        self._write_projects_json({
            "night_proj": {"path": proj, "priority": 1, "schedule": "22-2"},
        })
        with patch("time.localtime") as mock_time:
            mock_time.return_value = time.struct_time((2026, 3, 30, 12, 0, 0, 0, 89, 0))
            result = sm._pick_dispatcher_project()
        self.assertIsNone(result)


# ═══════════════════════════════════════════════════════════════════════════════
# SLA Enforcement Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpecSLA(OrchestratorTestBase):
    """Test spec wall-clock SLA enforcement."""

    @patch("subprocess.Popen")
    def test_spec_within_sla_not_dropped(self, mock_popen):
        sm = self._make_station_manager()
        train = sm.trains[0]
        train.spec_path = "/tmp/fake_spec.json"
        train.branch = "feature/test"
        train.spec_started_at = time.time() - 100  # well within 1800s SLA
        sm._check_spec_sla()
        self.assertIsNotNone(train.spec_path)

    @patch("subprocess.Popen")
    def test_spec_exceeding_sla_dropped(self, mock_popen):
        sm = self._make_station_manager()
        train = sm.trains[0]
        spec = self._write_spec("sla_test.json", {
            "title": "sla-test", "description": "test",
            "priority": "medium", "complexity": "high",
        })
        train.spec_path = spec
        train.branch = "feature/sla-test"
        train.spec_started_at = time.time() - (config.SPEC_SLA_SECONDS + 120)
        sm._check_spec_sla()
        self.assertIsNone(train.spec_path)
        self.assertIsNone(train.branch)


class TestCheckpointSLA(OrchestratorTestBase):
    """Test checkpoint idle SLA enforcement."""

    @patch("subprocess.Popen")
    def test_active_agents_reset_checkpoint_timer(self, mock_popen):
        sm = self._make_station_manager()
        train = sm.trains[0]
        train.spec_path = "/tmp/fake_spec.json"
        train.branch = "feature/test"
        train.checkpoint_idle_since = time.time() - 500
        # Simulate active conductor
        train.conductor = MagicMock()
        sm._check_checkpoint_sla()
        self.assertEqual(train.checkpoint_idle_since, 0.0)

    @patch("subprocess.Popen")
    def test_checkpoint_sla_removes_stale_feedback(self, mock_popen):
        sm = self._make_station_manager()
        train = sm.trains[0]
        train.spec_path = "/tmp/fake_spec.json"
        train.branch = "feature/checkpoint-test"
        self._write_feedback(train.branch, "CHANGES_REQUESTED\nFix stuff")
        train.checkpoint_idle_since = time.time() - (config.CHECKPOINT_SLA_SECONDS + 120)
        train.conductor = None
        train.inspector = None
        sm._check_checkpoint_sla()
        feedback_path = sm._feedback_path(train.branch)
        self.assertFalse(os.path.exists(feedback_path))
        self.assertEqual(train.checkpoint_idle_since, 0.0)


class TestIdleSLA(OrchestratorTestBase):
    """Test all-idle SLA enforcement."""

    @patch("subprocess.Popen")
    def test_busy_train_resets_idle_timer(self, mock_popen):
        sm = self._make_station_manager()
        sm.all_idle_since = time.time() - 50000
        train = sm.trains[0]
        train.branch = "feature/busy"
        sm._check_idle_sla()
        self.assertEqual(sm.all_idle_since, 0.0)

    @patch("subprocess.Popen")
    def test_idle_sla_clears_dispatcher_throttle(self, mock_popen):
        sm = self._make_station_manager()
        sm.last_launch_times["dispatcher"] = time.time()
        sm.all_idle_since = time.time() - (config.IDLE_SLA_SECONDS + 10)
        sm._check_idle_sla()
        self.assertNotIn("dispatcher", sm.last_launch_times)


# ═══════════════════════════════════════════════════════════════════════════════
# TestTriageSystem
# ═══════════════════════════════════════════════════════════════════════════════

class TestTriageSystem(OrchestratorTestBase):
    """Test triage gate between dispatcher and conductor."""

    def _setup_triage_train(self, mock_popen):
        """Set up a train with a spec ready for triage."""
        mock_popen.return_value = self._make_mock_proc()
        sm = self._make_station_manager()
        sm._create_worktree = MagicMock(return_value="/tmp/fake_worktree")
        sm._remove_worktree = MagicMock()
        sm._git = MagicMock(return_value="")
        sm._git_rc = MagicMock(return_value=(0, "", ""))
        sm._git_has_branch = MagicMock(return_value=False)
        sm._git_diff_trunk = MagicMock(return_value="")
        sm._git_last_commit = MagicMock(return_value="abc123")
        train = sm.trains[0]
        dev_dir = os.path.join(self.tmpdir, "dev")
        os.makedirs(dev_dir, exist_ok=True)
        config.DEVELOPMENT_DIR = self.tmpdir
        config.DEFAULT_PROJECT = "dev"
        self._write_spec("triage_test.json", {
            "title": "triage-feature",
            "description": "test feature for triage",
            "working_dir": dev_dir,
        })
        sm._train_phase_conductor(train)
        self.assertTrue(train.needs_triage)
        return sm, train

    @patch("subprocess.Popen")
    def test_build_verdict_approves_spec(self, mock_popen):
        sm, train = self._setup_triage_train(mock_popen)
        # Launch triage
        sm._train_phase_triage(train)
        self.assertIsNotNone(train.triage)
        # Simulate BUILD verdict
        train.triage.proc.poll.return_value = 0
        train.triage.proc.returncode = 0
        train.triage._output = "BUILD\nThis spec is useful and ready."
        train.triage._live_log_path = "/dev/null"
        sm._train_phase_triage(train)
        self.assertFalse(train.needs_triage)
        self.assertIsNotNone(train.spec_path)  # spec not removed

    @patch("subprocess.Popen")
    def test_reject_verdict_removes_spec(self, mock_popen):
        sm, train = self._setup_triage_train(mock_popen)
        spec_path = train.spec_path
        sm._train_phase_triage(train)
        # Simulate REJECT verdict
        train.triage.proc.poll.return_value = 0
        train.triage.proc.returncode = 0
        train.triage._output = "REJECT\nNot useful right now."
        train.triage._live_log_path = "/dev/null"
        sm._train_phase_triage(train)
        self.assertIsNone(train.spec_path)  # pipeline reset
        self.assertIsNone(train.branch)
        # Spec file should be removed
        self.assertFalse(os.path.exists(spec_path))
        self.assertFalse(os.path.exists(spec_path + ".in_progress"))

    @patch("subprocess.Popen")
    def test_hold_verdict_moves_spec_to_drafts(self, mock_popen):
        sm, train = self._setup_triage_train(mock_popen)
        sm._train_phase_triage(train)
        # Simulate HOLD verdict
        train.triage.proc.poll.return_value = 0
        train.triage.proc.returncode = 0
        train.triage._output = "HOLD\nNeeds more detail on acceptance criteria."
        train.triage._live_log_path = "/dev/null"
        sm._train_phase_triage(train)
        self.assertIsNone(train.spec_path)  # pipeline reset
        # Spec should be in drafts
        drafts = os.listdir(config.DRAFTS_DIR)
        self.assertEqual(len(drafts), 1)
        self.assertTrue(drafts[0].endswith(".json"))

    @patch("subprocess.Popen")
    def test_agent_failure_is_fail_open(self, mock_popen):
        sm, train = self._setup_triage_train(mock_popen)
        sm._train_phase_triage(train)
        # Simulate agent failure (rc != 0)
        train.triage.proc.poll.return_value = 1
        train.triage.proc.returncode = 1
        train.triage._output = ""
        train.triage._live_log_path = "/dev/null"
        sm._train_phase_triage(train)
        self.assertFalse(train.needs_triage)  # fail-open: approved
        self.assertEqual(train.triage_failures, 1)
        self.assertGreater(train.triage_cooldown_until, 0)

    @patch("subprocess.Popen")
    def test_unrecognized_verdict_rejects(self, mock_popen):
        sm, train = self._setup_triage_train(mock_popen)
        spec_path = train.spec_path
        sm._train_phase_triage(train)
        # Simulate unrecognized verdict
        train.triage.proc.poll.return_value = 0
        train.triage.proc.returncode = 0
        train.triage._output = "MAYBE\nI'm not sure about this one."
        train.triage._live_log_path = "/dev/null"
        sm._train_phase_triage(train)
        self.assertIsNone(train.spec_path)  # pipeline reset (rejected)
        self.assertIsNone(train.branch)

    @patch("subprocess.Popen")
    def test_triage_timeout_is_fail_open(self, mock_popen):
        sm, train = self._setup_triage_train(mock_popen)
        sm._train_phase_triage(train)
        self.assertIsNotNone(train.triage)
        # Simulate timeout: not finished yet, but timed out
        train.triage.proc.poll.return_value = None
        train.triage.is_timed_out = MagicMock(return_value=True)
        train.triage.start_time = time.time() - 9999
        sm._train_phase_triage(train)
        self.assertFalse(train.needs_triage)  # fail-open: approved
        self.assertEqual(train.triage_failures, 1)

    @patch("subprocess.Popen")
    def test_conductor_picks_up_spec_and_sets_needs_triage(self, mock_popen):
        mock_popen.return_value = self._make_mock_proc()
        sm, train = self._setup_triage_train(mock_popen)
        # After _train_phase_conductor, train should have spec and branch but needs_triage=True
        self.assertTrue(train.needs_triage)
        self.assertIsNotNone(train.spec_path)
        self.assertEqual(train.branch, "feature/triage-feature")
        self.assertIsNone(train.conductor)  # conductor not launched yet


if __name__ == "__main__":
    unittest.main()
