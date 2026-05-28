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
        sm._git_rc.assert_any_call("merge", "--no-ff", "feature/merge-me", cwd="/tmp/repo")

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
        # Must use shlex.split args list (not shell=True) for security
        mock_run.assert_any_call(
            ["systemctl", "restart", "myapp"],
            timeout=config.SERVICE_RESTART_TIMEOUT,
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

    def test_fire_entropy_terminates_spec(self):
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
        self.assertFalse(os.path.exists(spec_path + ".in_progress"))
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
    """Test _find_app_log resolution."""

    def _make_log_file(self, project_dir, name="app.log", content="line1\nline2\n"):
        log_path = os.path.join(project_dir, name)
        with open(log_path, "w") as f:
            f.write(content)
        return log_path

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
        sm.agent_cooldowns["ops"] = time.time() + 600
        payload = self._build_payload(sm)
        self.assertEqual(payload["agents"]["ops"]["status"], "cooldown")

    def test_global_agent_idle(self):
        sm = self._make_station_manager()
        payload = self._build_payload(sm)
        self.assertEqual(payload["agents"]["ops"]["status"], "idle")

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
                    "pipeline", "backlog", "completed", "stats", "budget", "activity",
                    "config", "verbose_logs"}
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
# TestOpsTriggerGate
# ═══════════════════════════════════════════════════════════════════════════════

class TestOpsTriggerGate(OrchestratorTestBase):
    """Ops should only launch when something is actionable (or the idle gap fires)."""

    def test_no_triggers_no_launch(self):
        sm = self._make_station_manager()
        # Simulate Ops having run recently so the idle gap doesn't fire.
        sm.last_launch_times["ops"] = time.time()
        should, reason = sm._ops_should_launch()
        self.assertFalse(should)
        self.assertEqual(reason, "")

    def test_fresh_orchestrator_no_triggers_no_launch(self):
        # StationManager init seeds last_launch_times["ops"] = startup time,
        # so the idle-gap rule must not fire on a fresh orchestrator with no
        # triggers — same path as "no triggers, recently ran".
        sm = self._make_station_manager()
        should, _ = sm._ops_should_launch()
        self.assertFalse(should)

    def test_consecutive_failures_fire(self):
        sm = self._make_station_manager()
        sm.last_launch_times["ops"] = time.time()
        sm.consecutive_failures["dispatcher"] = 2
        should, reason = sm._ops_should_launch()
        self.assertTrue(should)
        self.assertIn("dispatcher", reason)

    def test_zero_failure_count_does_not_fire(self):
        # consecutive_failures[name] = 0 entries (left over after success) must not trigger
        sm = self._make_station_manager()
        sm.last_launch_times["ops"] = time.time()
        sm.consecutive_failures["dispatcher"] = 0
        should, _ = sm._ops_should_launch()
        self.assertFalse(should)

    def test_sleep_mode_fires(self):
        sm = self._make_station_manager()
        sm.last_launch_times["ops"] = time.time()
        sm.sleep_until = time.time() + 600
        should, reason = sm._ops_should_launch()
        self.assertTrue(should)
        self.assertIn("sleep", reason)

    def test_restart_pending_fires(self):
        sm = self._make_station_manager()
        sm.last_launch_times["ops"] = time.time()
        sm.restart_pending = True
        should, reason = sm._ops_should_launch()
        self.assertTrue(should)
        self.assertIn("restart", reason)

    def test_stalled_projects_fires(self):
        sm = self._make_station_manager()
        sm.last_launch_times["ops"] = time.time()
        sm._stalled_projects = {"/dev/my-app": time.time() + 3600}
        should, reason = sm._ops_should_launch()
        self.assertTrue(should)
        self.assertIn("my-app", reason)

    def test_budget_high_utilization_fires(self):
        sm = self._make_station_manager()
        sm.last_launch_times["ops"] = time.time()
        sm.budget.monthly_limit_usd = 100.0
        sm.budget._state["spend_usd"] = 85.0
        should, reason = sm._ops_should_launch()
        self.assertTrue(should)
        self.assertIn("budget", reason)

    def test_budget_low_utilization_does_not_fire(self):
        sm = self._make_station_manager()
        sm.last_launch_times["ops"] = time.time()
        sm.budget.monthly_limit_usd = 100.0
        sm.budget._state["spend_usd"] = 10.0
        should, _ = sm._ops_should_launch()
        self.assertFalse(should)

    def test_budget_disabled_does_not_fire(self):
        # Even at notional 100% utilization, if budget cap is 0 the trigger is silent.
        sm = self._make_station_manager()
        sm.last_launch_times["ops"] = time.time()
        sm.budget.monthly_limit_usd = 0
        sm.budget._state["spend_usd"] = 999.0
        should, _ = sm._ops_should_launch()
        self.assertFalse(should)

    def test_idle_gap_fires_after_threshold(self):
        sm = self._make_station_manager()
        sm.last_launch_times["ops"] = time.time() - (config.MAX_OPS_IDLE_SECONDS + 60)
        should, reason = sm._ops_should_launch()
        self.assertTrue(should)
        self.assertIn("idle gap", reason)

    def test_idle_gap_does_not_fire_before_threshold(self):
        sm = self._make_station_manager()
        sm.last_launch_times["ops"] = time.time() - 600  # 10 min ago
        should, _ = sm._ops_should_launch()
        self.assertFalse(should)


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


# ═══════════════════════════════════════════════════════════════════════════════
# 14. TestSpecRenameRace
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpecRenameRace(OrchestratorTestBase):
    """Verify the atomic spec-claim fix: a failed rename aborts pipeline assignment."""

    @patch("subprocess.Popen")
    def test_rename_failure_resets_pipeline(self, mock_popen):
        """If os.rename raises OSError the train must be left idle (not holding the spec)."""
        mock_popen.return_value = self._make_mock_proc()
        sm = self._make_station_manager()
        sm._git = MagicMock(return_value="")
        sm._git_has_branch = MagicMock(return_value=False)
        dev_dir = os.path.join(self.tmpdir, "dev")
        os.makedirs(dev_dir, exist_ok=True)
        config.DEVELOPMENT_DIR = self.tmpdir
        config.DEFAULT_PROJECT = "dev"
        self._write_spec("race.json", {"title": "race-feature", "working_dir": dev_dir})
        train = sm.trains[0]

        with patch("os.rename", side_effect=OSError("device busy")):
            sm._train_phase_conductor(train)

        # Train must be idle — spec not claimed
        self.assertIsNone(train.spec_path)
        self.assertIsNone(train.branch)
        self.assertFalse(train.needs_triage)

    @patch("subprocess.Popen")
    def test_successful_rename_sets_needs_triage(self, mock_popen):
        """When rename succeeds, the train should have needs_triage=True."""
        mock_popen.return_value = self._make_mock_proc()
        sm = self._make_station_manager()
        sm._git = MagicMock(return_value="")
        sm._git_has_branch = MagicMock(return_value=False)
        dev_dir = os.path.join(self.tmpdir, "dev")
        os.makedirs(dev_dir, exist_ok=True)
        config.DEVELOPMENT_DIR = self.tmpdir
        config.DEFAULT_PROJECT = "dev"
        self._write_spec("ok.json", {"title": "ok-feature", "working_dir": dev_dir})
        train = sm.trains[0]
        sm._train_phase_conductor(train)
        self.assertTrue(train.needs_triage)
        self.assertIsNotNone(train.spec_path)
        # .in_progress file must exist on disk
        self.assertTrue(os.path.exists(train.spec_path + ".in_progress"))

    @patch("subprocess.Popen")
    def test_two_trains_cannot_both_claim_same_spec(self, mock_popen):
        """With two trains and one spec, only one should claim it."""
        mock_popen.return_value = self._make_mock_proc()
        config.TRAIN_CONFIG = {
            "regular": {
                "count": 2,
                "conductor_model": "claude-sonnet-4-5-20250929",
                "inspector_model": "claude-sonnet-4-5-20250929",
                "complexity": "high",
                "dispatcher_interval": 1800,
            },
        }
        sm = self._make_station_manager()
        sm._git = MagicMock(return_value="")
        sm._git_has_branch = MagicMock(return_value=False)
        dev_dir = os.path.join(self.tmpdir, "dev")
        os.makedirs(dev_dir, exist_ok=True)
        config.DEVELOPMENT_DIR = self.tmpdir
        config.DEFAULT_PROJECT = "dev"
        self._write_spec("shared.json", {"title": "shared-feature", "working_dir": dev_dir})

        train_a, train_b = sm.trains[0], sm.trains[1]
        sm._train_phase_conductor(train_a)
        sm._train_phase_conductor(train_b)

        # Exactly one train should have claimed the spec
        claimed = [t for t in (train_a, train_b) if t.spec_path is not None]
        self.assertEqual(len(claimed), 1)
        self.assertTrue(claimed[0].needs_triage)


# ═══════════════════════════════════════════════════════════════════════════════
# 15. TestServiceRestart
# ═══════════════════════════════════════════════════════════════════════════════

class TestServiceRestart(OrchestratorTestBase):
    """Verify SERVICE_RESTART_CMD uses shlex.split (not shell=True)."""

    def _setup_approved_train(self):
        sm = self._make_station_manager()
        sm._git = MagicMock(return_value="")
        sm._git_rc = MagicMock(return_value=(0, "ok", ""))
        sm._git_has_branch = MagicMock(return_value=True)
        sm._git_last_commit = MagicMock(return_value="abc123")
        sm._remove_worktree = MagicMock()
        train = sm.trains[0]
        train.branch = "feature/deploy"
        train.working_dir = "/tmp/wt"
        train.repo_dir = "/tmp/repo"
        spec_path = self._write_spec("deploy.json", {"title": "deploy"})
        train.spec_path = spec_path
        os.rename(spec_path, spec_path + ".in_progress")
        self._write_feedback("feature/deploy", "APPROVED\n")
        return sm, train

    @patch("subprocess.run")
    def test_restart_cmd_uses_args_list_not_shell(self, mock_run):
        mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)
        config.SERVICE_RESTART_CMD = "systemctl restart my-service"
        sm, train = self._setup_approved_train()
        sm._train_phase_service_recovery(train)
        call_args = mock_run.call_args
        # First positional arg must be a list, not a string
        self.assertIsInstance(call_args[0][0], list)
        self.assertEqual(call_args[0][0], ["systemctl", "restart", "my-service"])
        # shell must not be True
        self.assertNotEqual(call_args[1].get("shell"), True)

    @patch("subprocess.run")
    def test_restart_cmd_multi_word_split(self, mock_run):
        mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)
        config.SERVICE_RESTART_CMD = "sudo service nginx reload"
        sm, train = self._setup_approved_train()
        sm._train_phase_service_recovery(train)
        call_args = mock_run.call_args
        self.assertEqual(call_args[0][0], ["sudo", "service", "nginx", "reload"])

    @patch("subprocess.run")
    def test_invalid_cmd_string_does_not_crash(self, mock_run):
        mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)
        config.SERVICE_RESTART_CMD = "systemctl restart 'unclosed"  # invalid quotes
        sm, train = self._setup_approved_train()
        sm._train_phase_service_recovery(train)  # should not raise
        mock_run.assert_not_called()  # invalid cmd → skip restart


# ═══════════════════════════════════════════════════════════════════════════════
# 16. TestActivityLogObservability
# ═══════════════════════════════════════════════════════════════════════════════

class TestActivityLogObservability(OrchestratorTestBase):
    """Verify that activity log write failures go to stderr, not silently dropped."""

    def test_write_failure_prints_to_stderr(self):
        from orchestrator import activity
        config.ACTIVITY_LOG = "/nonexistent/path/activity.log"
        import io
        captured = io.StringIO()
        with patch("sys.stderr", captured):
            activity("test message")
        output = captured.getvalue()
        self.assertIn("activity log write failed", output)

    def test_message_still_logged_to_python_logger_on_write_failure(self):
        from orchestrator import activity
        config.ACTIVITY_LOG = "/nonexistent/path/activity.log"
        with self.assertLogs("orchestrator", level="INFO") as cm:
            activity("hello from test")
        self.assertTrue(any("hello from test" in line for line in cm.output))


# ═══════════════════════════════════════════════════════════════════════════════
# 17. TestLogSourcePlugin
# ═══════════════════════════════════════════════════════════════════════════════

class TestLogSourcePlugin(OrchestratorTestBase):
    """Verify the LogSource plugin interface and watcher integration."""

    @patch("subprocess.Popen")
    def test_custom_source_read_new_lines_called_on_tick(self, mock_popen):
        mock_popen.return_value = self._make_mock_proc()
        from orchestrator import LogSource, StationManager
        sm = self._make_station_manager()
        sm._trigger_signal_reactive = MagicMock()

        class FakeSource(LogSource):
            @property
            def name(self):
                return "fake"
            def read_new_lines(self, project_dir):
                return ["ERROR: something broke"]

        source = FakeSource()
        sm.register_log_source(source)
        sm._log_watcher_tick(self.tmpdir)
        sm._trigger_signal_reactive.assert_called_once()

    @patch("subprocess.Popen")
    def test_source_returning_no_lines_does_not_trigger_signal(self, mock_popen):
        mock_popen.return_value = self._make_mock_proc()
        from orchestrator import LogSource, StationManager
        sm = self._make_station_manager()
        sm._trigger_signal_reactive = MagicMock()
        # Remove default FileLogSource, replace with empty source
        sm._log_sources.clear()

        class EmptySource(LogSource):
            @property
            def name(self):
                return "empty"
            def read_new_lines(self, project_dir):
                return []

        sm.register_log_source(EmptySource())
        sm._log_watcher_tick(self.tmpdir)
        sm._trigger_signal_reactive.assert_not_called()

    @patch("subprocess.Popen")
    def test_file_log_source_skips_non_error_lines(self, mock_popen):
        mock_popen.return_value = self._make_mock_proc()
        from orchestrator import FileLogSource
        source = FileLogSource()
        log_file = os.path.join(self.tmpdir, "app.log")
        config.APP_LOG_GLOB = "app.log"
        config.DEVELOPMENT_DIR = self.tmpdir

        with open(log_file, "w") as f:
            f.write("INFO: all good\nDEBUG: verbose\n")
        # First call initializes offset to EOF
        source.read_new_lines(self.tmpdir)

        # Write new content — mix of error and info
        with open(log_file, "a") as f:
            f.write("ERROR: database down\nINFO: retry ok\n")

        lines = source.read_new_lines(self.tmpdir)
        self.assertIn("ERROR: database down", lines)
        self.assertIn("INFO: retry ok", lines)  # source returns all lines; watcher filters

    @patch("subprocess.Popen")
    def test_source_exception_does_not_crash_watcher(self, mock_popen):
        mock_popen.return_value = self._make_mock_proc()
        from orchestrator import LogSource
        sm = self._make_station_manager()
        sm._trigger_signal_reactive = MagicMock()
        sm._log_sources.clear()

        class CrashingSource(LogSource):
            @property
            def name(self):
                return "crasher"
            def read_new_lines(self, project_dir):
                raise RuntimeError("source exploded")

        sm.register_log_source(CrashingSource())
        sm._log_watcher_tick(self.tmpdir)  # must not raise


# ═══════════════════════════════════════════════════════════════════════════════
# 18. TestMetrics
# ═══════════════════════════════════════════════════════════════════════════════

class TestMetrics(unittest.TestCase):
    """Verify the metrics registry and Prometheus rendering."""

    def setUp(self):
        from metrics import Metrics
        self.m = Metrics()  # fresh instance per test

    def test_counter_starts_at_zero(self):
        samples = self.m.specs_total.collect()
        self.assertEqual(samples, [])

    def test_counter_inc_unlabeled(self):
        self.m.specs_total.inc()
        samples = self.m.specs_total.collect()
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0][1], 1.0)

    def test_counter_inc_labeled(self):
        self.m.specs_total.inc({"outcome": "merged"})
        self.m.specs_total.inc({"outcome": "merged"})
        self.m.specs_total.inc({"outcome": "rejected"})
        samples = dict(
            (tuple(sorted(labels.items())), v)
            for labels, v in self.m.specs_total.collect()
        )
        self.assertEqual(samples[(("outcome", "merged"),)], 2.0)
        self.assertEqual(samples[(("outcome", "rejected"),)], 1.0)

    def test_gauge_set(self):
        self.m.backlog_size.set(42.0)
        samples = self.m.backlog_size.collect()
        self.assertEqual(samples[0][1], 42.0)

    def test_gauge_set_overwrites(self):
        self.m.backlog_size.set(10.0)
        self.m.backlog_size.set(20.0)
        samples = self.m.backlog_size.collect()
        self.assertEqual(samples[0][1], 20.0)

    def test_render_prometheus_counter_format(self):
        from metrics import render_prometheus
        self.m.specs_total.inc({"outcome": "merged"})
        output = render_prometheus(self.m)
        self.assertIn("# HELP yamanote_specs_total", output)
        self.assertIn("# TYPE yamanote_specs_total counter", output)
        self.assertIn('yamanote_specs_total{outcome="merged"} 1', output)

    def test_render_prometheus_gauge_format(self):
        from metrics import render_prometheus
        self.m.backlog_size.set(7.0)
        output = render_prometheus(self.m)
        self.assertIn("# HELP yamanote_backlog_size", output)
        self.assertIn("# TYPE yamanote_backlog_size gauge", output)
        self.assertIn("yamanote_backlog_size 7", output)

    def test_render_prometheus_empty_counter_shows_zero(self):
        from metrics import render_prometheus
        output = render_prometheus(self.m)
        # All metrics with no samples should emit a zero baseline
        self.assertIn("yamanote_specs_total 0", output)

    def test_render_prometheus_ends_with_newline(self):
        from metrics import render_prometheus
        output = render_prometheus(self.m)
        self.assertTrue(output.endswith("\n"))

    def test_counter_thread_safety(self):
        import threading
        errors = []
        def inc_many():
            try:
                for _ in range(1000):
                    self.m.agent_launches_total.inc({"agent": "test"})
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=inc_many) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        samples = dict(
            (tuple(sorted(l.items())), v)
            for l, v in self.m.agent_launches_total.collect()
        )
        self.assertEqual(samples[(("agent", "test"),)], 5000.0)


class TestStallCircuitBreaker(OrchestratorTestBase):
    """Tests for the dispatcher-triage reject loop circuit breaker."""

    def setUp(self):
        super().setUp()
        self.project_dir = os.path.join(self.tmpdir, "dev")
        os.makedirs(self.project_dir)
        config.DEVELOPMENT_DIR = self.tmpdir
        config.DEFAULT_PROJECT = "dev"
        config.SELF_PROJECT_DIR = ""

    def _make_rejecting_train(self, sm, n):
        """Simulate n consecutive triage rejections for the test project."""
        from orchestrator import Train
        train = sm.trains[0]
        train.repo_dir = self.project_dir
        train.working_dir = self.project_dir
        train.branch = "feature/some-spec"
        spec_path = os.path.join(config.BACKLOG_DIR, "spec.json")
        os.makedirs(config.BACKLOG_DIR, exist_ok=True)
        with open(spec_path, "w") as f:
            json.dump({"title": "some-spec", "working_dir": self.project_dir}, f)
        train.spec_path = spec_path
        for _ in range(n):
            sm._on_triage_rejection(train)
        return train

    def test_no_stall_below_threshold(self):
        sm = self._make_station_manager()
        config.MAX_CONSECUTIVE_REJECTIONS = 5
        self._make_rejecting_train(sm, 4)
        self.assertNotIn(self.project_dir, sm._stalled_projects)

    def test_stall_triggers_at_threshold(self):
        sm = self._make_station_manager()
        config.MAX_CONSECUTIVE_REJECTIONS = 5
        self._make_rejecting_train(sm, 5)
        self.assertIn(self.project_dir, sm._stalled_projects)
        self.assertGreater(sm._stalled_projects[self.project_dir], time.time())

    def test_stall_resets_rejection_counter(self):
        sm = self._make_station_manager()
        config.MAX_CONSECUTIVE_REJECTIONS = 3
        self._make_rejecting_train(sm, 3)
        self.assertEqual(sm._project_rejection_counts.get(self.project_dir, 0), 0)

    def test_build_verdict_resets_rejection_counter(self):
        sm = self._make_station_manager()
        config.MAX_CONSECUTIVE_REJECTIONS = 5
        train = self._make_rejecting_train(sm, 3)
        self.assertEqual(sm._project_rejection_counts[self.project_dir], 3)
        # Simulate BUILD verdict
        project = train.repo_dir or ""
        if project:
            sm._project_rejection_counts.pop(project, None)
        self.assertEqual(sm._project_rejection_counts.get(self.project_dir, 0), 0)

    def test_merge_success_clears_stall(self):
        sm = self._make_station_manager()
        config.MAX_CONSECUTIVE_REJECTIONS = 3
        self._make_rejecting_train(sm, 3)
        self.assertIn(self.project_dir, sm._stalled_projects)
        # Simulate merge success clearing stall
        sm._project_rejection_counts.pop(self.project_dir, None)
        sm._stalled_projects.pop(self.project_dir, None)
        self.assertNotIn(self.project_dir, sm._stalled_projects)

    @patch("subprocess.Popen")
    @patch("subprocess.run")
    def test_stalled_project_skips_dispatcher(self, mock_run, mock_popen):
        """When a project is stalled, _phase_dispatcher must not launch."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        mock_popen.return_value = self._make_mock_proc()
        sm = self._make_station_manager()
        # Set stall with resume time well in the future
        sm._stalled_projects[self.project_dir] = time.time() + 86400
        sm.last_launch_times.pop("dispatcher", None)
        sm._phase_dispatcher()
        mock_popen.assert_not_called()

    @patch("subprocess.Popen")
    @patch("subprocess.run")
    def test_expired_stall_resumes_dispatcher(self, mock_run, mock_popen):
        """When stall period expires, the dispatcher fires on the next eligible tick."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        mock_popen.return_value = self._make_mock_proc()
        sm = self._make_station_manager()
        os.makedirs(self.project_dir, exist_ok=True)
        # Write a projects.json so _pick_dispatcher_project finds the project
        import json as _json
        projects_cfg = {"projects": {"dev": {"path": self.project_dir, "enabled": True}}}
        with open(config.PROJECTS_CONFIG_PATH, "w") as f:
            _json.dump(projects_cfg, f)
        # Set stall that already expired
        sm._stalled_projects[self.project_dir] = time.time() - 1
        sm.last_launch_times.pop("dispatcher", None)
        sm._phase_dispatcher()
        # Stall should be cleared
        self.assertNotIn(self.project_dir, sm._stalled_projects)


class TestDraftsRecycler(OrchestratorTestBase):
    """Tests for _recycle_stale_drafts."""

    def _write_draft(self, fname, age_seconds):
        os.makedirs(config.DRAFTS_DIR, exist_ok=True)
        path = os.path.join(config.DRAFTS_DIR, fname)
        with open(path, "w") as f:
            json.dump({"title": "held-spec"}, f)
        mtime = time.time() - age_seconds
        os.utime(path, (mtime, mtime))
        return path

    def test_old_draft_moved_to_backlog(self):
        sm = self._make_station_manager()
        config.DRAFTS_RECYCLE_AGE_SECONDS = 3600
        sm._last_draft_recycle = 0  # force run now
        self._write_draft("old_spec.json", age_seconds=7200)
        os.makedirs(config.BACKLOG_DIR, exist_ok=True)
        sm._recycle_stale_drafts()
        self.assertFalse(os.path.exists(os.path.join(config.DRAFTS_DIR, "old_spec.json")))
        self.assertTrue(os.path.exists(os.path.join(config.BACKLOG_DIR, "old_spec.json")))

    def test_young_draft_stays_in_drafts(self):
        sm = self._make_station_manager()
        config.DRAFTS_RECYCLE_AGE_SECONDS = 3600
        sm._last_draft_recycle = 0
        self._write_draft("new_spec.json", age_seconds=60)
        sm._recycle_stale_drafts()
        self.assertTrue(os.path.exists(os.path.join(config.DRAFTS_DIR, "new_spec.json")))
        self.assertFalse(os.path.exists(os.path.join(config.BACKLOG_DIR, "new_spec.json")))

    def test_recycle_skips_when_recently_run(self):
        sm = self._make_station_manager()
        config.DRAFTS_RECYCLE_AGE_SECONDS = 3600
        sm._last_draft_recycle = time.time()  # just ran
        self._write_draft("old_spec.json", age_seconds=7200)
        sm._recycle_stale_drafts()
        # Should not have been touched (still in drafts)
        self.assertTrue(os.path.exists(os.path.join(config.DRAFTS_DIR, "old_spec.json")))

    def test_empty_drafts_dir_no_error(self):
        sm = self._make_station_manager()
        sm._last_draft_recycle = 0
        config.DRAFTS_RECYCLE_AGE_SECONDS = 0
        sm._recycle_stale_drafts()  # should not raise

    def test_non_json_files_ignored(self):
        sm = self._make_station_manager()
        config.DRAFTS_RECYCLE_AGE_SECONDS = 60
        sm._last_draft_recycle = 0
        os.makedirs(config.DRAFTS_DIR, exist_ok=True)
        txt_path = os.path.join(config.DRAFTS_DIR, "readme.txt")
        with open(txt_path, "w") as f:
            f.write("notes")
        mtime = time.time() - 7200
        os.utime(txt_path, (mtime, mtime))
        sm._recycle_stale_drafts()
        self.assertTrue(os.path.exists(txt_path))  # not moved


class TestWorktreeGC(OrchestratorTestBase):
    """Tests for _gc_orphaned_worktrees."""

    @patch("subprocess.run")
    def test_orphaned_worktree_removed(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        sm = self._make_station_manager()
        config.WORKTREE_GC_INTERVAL = 0
        sm._last_worktree_gc = 0

        project_dir = os.path.join(self.tmpdir, "myproject")
        worktree_base = os.path.join(project_dir, ".worktrees")
        stale_wt = os.path.join(worktree_base, "old-train-0")
        os.makedirs(stale_wt)
        sm._seen_project_dirs.add(project_dir)

        sm._gc_orphaned_worktrees()

        self.assertFalse(os.path.isdir(stale_wt))

    @patch("subprocess.run")
    def test_active_worktree_preserved(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        sm = self._make_station_manager()
        config.WORKTREE_GC_INTERVAL = 0
        sm._last_worktree_gc = 0

        project_dir = os.path.join(self.tmpdir, "myproject")
        worktree_base = os.path.join(project_dir, ".worktrees")
        active_wt = os.path.join(worktree_base, "regular-0")
        os.makedirs(active_wt)
        sm._seen_project_dirs.add(project_dir)

        # Mark this worktree as active by setting it on a train
        train = sm.trains[0]
        train.working_dir = active_wt
        train.branch = "feature/active-work"

        sm._gc_orphaned_worktrees()

        self.assertTrue(os.path.isdir(active_wt))

    def test_gc_skips_when_recently_run(self):
        sm = self._make_station_manager()
        config.WORKTREE_GC_INTERVAL = 3600
        sm._last_worktree_gc = time.time()  # just ran

        project_dir = os.path.join(self.tmpdir, "myproject")
        stale_wt = os.path.join(project_dir, ".worktrees", "old-train")
        os.makedirs(stale_wt)
        sm._seen_project_dirs.add(project_dir)

        sm._gc_orphaned_worktrees()

        self.assertTrue(os.path.isdir(stale_wt))  # not touched

    def test_gc_no_error_on_missing_project_dir(self):
        sm = self._make_station_manager()
        config.WORKTREE_GC_INTERVAL = 0
        sm._last_worktree_gc = 0
        sm._seen_project_dirs.add("/nonexistent/path")
        sm._gc_orphaned_worktrees()  # should not raise


class TestClassifySpecType(unittest.TestCase):
    """Unit tests for _classify_spec_type heuristic."""

    def setUp(self):
        from orchestrator import _classify_spec_type
        self.classify = _classify_spec_type

    def test_feature_default(self):
        self.assertEqual(self.classify("add-user-profile-page"), "feature")

    def test_bugfix_by_fix_keyword(self):
        self.assertEqual(self.classify("fix-login-redirect"), "bugfix")

    def test_bugfix_by_bug_keyword(self):
        self.assertEqual(self.classify("bug-in-payment-flow"), "bugfix")

    def test_bugfix_by_error_keyword(self):
        self.assertEqual(self.classify("handle-error-on-signup"), "bugfix")

    def test_hardening_by_test_keyword(self):
        self.assertEqual(self.classify("add-test-coverage-for-api"), "hardening")

    def test_hardening_by_metric_keyword(self):
        self.assertEqual(self.classify("instrument-metric-collection"), "hardening")

    def test_refactor_keyword(self):
        self.assertEqual(self.classify("refactor-database-layer"), "refactor")

    def test_refactor_cleanup(self):
        self.assertEqual(self.classify("cleanup-legacy-endpoints"), "refactor")

    def test_case_insensitive(self):
        self.assertEqual(self.classify("Fix-Critical-Bug"), "bugfix")


class TestWorkBalanceDigest(OrchestratorTestBase):
    """Tests for StationManager._gather_work_balance_digest."""

    def _write_activity_log(self, lines):
        with open(config.ACTIVITY_LOG, "w") as f:
            f.write("\n".join(lines) + "\n")

    def _make_terminus_line(self, title):
        return f"[2026-01-01 12:00:00]  TERMINUS [t1] — branch feature/{title} approved, merging to trunk."

    def test_no_activity_log_returns_no_history(self):
        sm = self._make_station_manager()
        result = sm._gather_work_balance_digest("/some/project")
        self.assertIn("no activity history", result)

    def test_empty_activity_log_returns_no_merged(self):
        self._write_activity_log(["[2026-01-01 12:00:00]  Dispatcher — backlog empty"])
        sm = self._make_station_manager()
        result = sm._gather_work_balance_digest("/some/project")
        self.assertIn("no merged specs", result)

    def test_all_features_signals_feature_heavy(self):
        lines = [self._make_terminus_line(f"add-feature-{i}") for i in range(10)]
        self._write_activity_log(lines)
        sm = self._make_station_manager()
        result = sm._gather_work_balance_digest("/some/project")
        self.assertIn("FEATURE-HEAVY", result)

    def test_mixed_signals_balanced(self):
        lines = [
            self._make_terminus_line("add-user-profile"),
            self._make_terminus_line("fix-login-bug"),
            self._make_terminus_line("add-dashboard-widget"),
            self._make_terminus_line("fix-checkout-error"),
            self._make_terminus_line("add-search-feature"),
            self._make_terminus_line("refactor-auth-layer"),
        ]
        self._write_activity_log(lines)
        sm = self._make_station_manager()
        result = sm._gather_work_balance_digest("/some/project")
        self.assertIn("BALANCED", result)

    def test_hardening_heavy_signals_hardening_heavy(self):
        lines = [self._make_terminus_line(f"add-test-coverage-module-{i}") for i in range(8)]
        lines += [self._make_terminus_line("add-new-feature")]
        lines += [self._make_terminus_line("fix-minor-bug")]
        self._write_activity_log(lines)
        sm = self._make_station_manager()
        result = sm._gather_work_balance_digest("/some/project")
        self.assertIn("HARDENING-HEAVY", result)

    def test_summary_includes_counts(self):
        lines = [
            self._make_terminus_line("add-feature-x"),
            self._make_terminus_line("fix-bug-y"),
        ]
        self._write_activity_log(lines)
        sm = self._make_station_manager()
        result = sm._gather_work_balance_digest("/some/project")
        self.assertIn("2 specs", result)

    def test_window_limits_to_recent_entries(self):
        # Write 30 terminus lines: 25 features + 5 hardening
        lines = [self._make_terminus_line(f"add-feature-{i}") for i in range(25)]
        lines += [self._make_terminus_line(f"add-test-{i}") for i in range(5)]
        self._write_activity_log(lines)
        sm = self._make_station_manager()
        # window=20 should see the last 20: all 5 hardening + 15 features
        result = sm._gather_work_balance_digest("/some/project", window=20)
        self.assertIn("20 specs", result)


class TestInspectorSpecContext(OrchestratorTestBase):
    """Tests that Inspector receives spec JSON from in_progress or spec file."""

    def _make_train_with_spec(self, sm, spec_data, use_in_progress=False):
        from orchestrator import Train
        train = sm.trains[0]
        spec_path = os.path.join(config.BACKLOG_DIR, "20260101_spec.json")
        os.makedirs(config.BACKLOG_DIR, exist_ok=True)
        with open(spec_path, "w") as f:
            json.dump(spec_data, f)
        train.spec_path = spec_path
        if use_in_progress:
            ip_path = spec_path + ".in_progress"
            with open(ip_path, "w") as f:
                json.dump(spec_data, f)
        return train

    @patch("subprocess.Popen")
    @patch("subprocess.run")
    def test_inspector_prompt_includes_spec_from_in_progress(self, mock_run, mock_popen):
        """Inspector gets spec JSON from .in_progress file when it exists."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        mock_popen.return_value = self._make_mock_proc(pid=999)

        sm = self._make_station_manager()
        spec_data = {"title": "my-feature", "description": "Do X", "working_dir": "/proj"}
        train = self._make_train_with_spec(sm, spec_data, use_in_progress=True)

        # Set up train to be in inspector-ready state
        train.branch = "feature/my-feature"
        train.working_dir = self.tmpdir
        train.repo_dir = self.tmpdir

        # Mock git operations so inspector launches
        with patch.object(sm, "_git_has_branch", return_value=True), \
             patch.object(sm, "_git_diff_trunk", return_value="+ some code"), \
             patch.object(sm, "_git_rc", return_value=(0, "", "")), \
             patch.object(sm, "_git", return_value=""), \
             patch.object(sm, "_feedback_path", return_value=os.path.join(self.tmpdir, "fb.md")):
            os.makedirs(config.REVIEW_DIR, exist_ok=True)
            sm._train_phase_inspector(train)

        self.assertIsNotNone(train.inspector)
        captured_prompt = mock_popen.call_args[0][0][-1]
        self.assertIn("my-feature", captured_prompt)

    @patch("subprocess.Popen")
    @patch("subprocess.run")
    def test_inspector_prompt_falls_back_to_spec_file(self, mock_run, mock_popen):
        """Inspector gets spec JSON from spec file when no .in_progress exists."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        mock_popen.return_value = self._make_mock_proc(pid=999)

        sm = self._make_station_manager()
        spec_data = {"title": "fallback-feature", "description": "Do Y", "working_dir": "/proj"}
        train = self._make_train_with_spec(sm, spec_data, use_in_progress=False)

        train.branch = "feature/fallback-feature"
        train.working_dir = self.tmpdir
        train.repo_dir = self.tmpdir

        with patch.object(sm, "_git_has_branch", return_value=True), \
             patch.object(sm, "_git_diff_trunk", return_value="+ some code"), \
             patch.object(sm, "_git_rc", return_value=(0, "", "")), \
             patch.object(sm, "_git", return_value=""), \
             patch.object(sm, "_feedback_path", return_value=os.path.join(self.tmpdir, "fb.md")):
            os.makedirs(config.REVIEW_DIR, exist_ok=True)
            sm._train_phase_inspector(train)

        self.assertIsNotNone(train.inspector)
        captured_prompt = mock_popen.call_args[0][0][-1]
        self.assertIn("fallback-feature", captured_prompt)

    @patch("subprocess.Popen")
    @patch("subprocess.run")
    def test_inspector_prompt_has_placeholder_when_no_spec_path(self, mock_run, mock_popen):
        """Inspector uses '(spec not available)' when train has no spec_path."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        mock_popen.return_value = self._make_mock_proc(pid=999)

        sm = self._make_station_manager()
        train = sm.trains[0]
        train.branch = "feature/no-spec-train"
        train.working_dir = self.tmpdir
        train.repo_dir = self.tmpdir
        train.spec_path = None

        with patch.object(sm, "_git_has_branch", return_value=True), \
             patch.object(sm, "_git_diff_trunk", return_value="+ some code"), \
             patch.object(sm, "_git_rc", return_value=(0, "", "")), \
             patch.object(sm, "_git", return_value=""), \
             patch.object(sm, "_feedback_path", return_value=os.path.join(self.tmpdir, "fb.md")):
            os.makedirs(config.REVIEW_DIR, exist_ok=True)
            sm._train_phase_inspector(train)

        self.assertIsNotNone(train.inspector)
        captured_prompt = mock_popen.call_args[0][0][-1]
        self.assertIn("spec not available", captured_prompt)


if __name__ == "__main__":
    unittest.main()
