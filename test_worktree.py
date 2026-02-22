#!/usr/bin/env python3
"""Unit tests for git worktree isolation logic.

Exercises _create_worktree, _remove_worktree, and the merge-in-main-repo
flow against real temporary git repos. No Claude API calls needed.
"""

import json
import os
import subprocess
import tempfile
import unittest

# Patch config paths before importing orchestrator so StationManager.__init__
# doesn't depend on the real project directory layout.
import config

_orig_backlog = config.BACKLOG_DIR
_orig_review = config.REVIEW_DIR
_orig_logs = config.LOGS_DIR


def _git(repo, *args):
    """Run a git command in the given repo and return stdout."""
    result = subprocess.run(
        ["git"] + list(args),
        capture_output=True, text=True, cwd=repo,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr}")
    return result.stdout.strip()


def _make_temp_repo():
    """Create a temp git repo with one commit on main."""
    repo = tempfile.mkdtemp(prefix="yamanote_test_")
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@test.com")
    _git(repo, "config", "user.name", "Test")
    readme = os.path.join(repo, "README.md")
    with open(readme, "w") as f:
        f.write("# Test project\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "Initial commit")
    return repo


class TestCreateWorktree(unittest.TestCase):
    """Test _create_worktree and _remove_worktree."""

    def setUp(self):
        self.repo = _make_temp_repo()
        # Point config dirs to temp locations so StationManager.__init__ works
        self.tmpdir = tempfile.mkdtemp(prefix="yamanote_cfg_")
        config.BACKLOG_DIR = os.path.join(self.tmpdir, "backlog")
        config.REVIEW_DIR = os.path.join(self.tmpdir, "review")
        config.LOGS_DIR = os.path.join(self.tmpdir, "logs")
        config.ACTIVITY_LOG = os.path.join(self.tmpdir, "activity.log")

        from orchestrator import StationManager
        self.sm = StationManager()

    def tearDown(self):
        # Clean up temp dirs
        import shutil
        shutil.rmtree(self.repo, ignore_errors=True)
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        config.BACKLOG_DIR = _orig_backlog
        config.REVIEW_DIR = _orig_review
        config.LOGS_DIR = _orig_logs

    def test_create_worktree_new_branch(self):
        """Creating a worktree for a new branch should create the directory and branch."""
        path = self.sm._create_worktree(self.repo, "feature/test-new", "train-0")

        self.assertTrue(os.path.isdir(path))
        self.assertEqual(path, os.path.join(self.repo, ".worktrees", "train-0"))

        # Branch should exist in the main repo
        self.assertTrue(self.sm._git_has_branch("feature/test-new", cwd=self.repo))

        # Worktree should be on the correct branch
        branch = _git(path, "branch", "--show-current")
        self.assertEqual(branch, "feature/test-new")

    def test_create_worktree_existing_branch(self):
        """Creating a worktree for an existing branch should check it out."""
        # Create the branch in the main repo first
        _git(self.repo, "branch", "feature/existing")

        path = self.sm._create_worktree(self.repo, "feature/existing", "train-1")

        self.assertTrue(os.path.isdir(path))
        branch = _git(path, "branch", "--show-current")
        self.assertEqual(branch, "feature/existing")

    def test_gitignore_added(self):
        """Creating a worktree should add .worktrees/ to .gitignore."""
        self.sm._create_worktree(self.repo, "feature/ignore-test", "train-0")

        gitignore = os.path.join(self.repo, ".gitignore")
        self.assertTrue(os.path.exists(gitignore))
        with open(gitignore) as f:
            content = f.read()
        self.assertIn(".worktrees/", content)

    def test_gitignore_not_duplicated(self):
        """Creating two worktrees shouldn't duplicate the .gitignore entry."""
        self.sm._create_worktree(self.repo, "feature/first", "train-0")
        self.sm._remove_worktree(self.repo, os.path.join(self.repo, ".worktrees", "train-0"))
        self.sm._create_worktree(self.repo, "feature/second", "train-1")

        gitignore = os.path.join(self.repo, ".gitignore")
        with open(gitignore) as f:
            content = f.read()
        self.assertEqual(content.count(".worktrees/"), 1)

    def test_remove_worktree(self):
        """Removing a worktree should delete the directory."""
        path = self.sm._create_worktree(self.repo, "feature/remove-test", "train-0")
        self.assertTrue(os.path.isdir(path))

        self.sm._remove_worktree(self.repo, path)
        self.assertFalse(os.path.isdir(path))

    def test_remove_worktree_idempotent(self):
        """Removing an already-removed worktree should not error."""
        path = os.path.join(self.repo, ".worktrees", "ghost")
        # Should not raise
        self.sm._remove_worktree(self.repo, path)

    def test_remove_worktree_none_safe(self):
        """Passing None should not raise."""
        self.sm._remove_worktree(None, None)
        self.sm._remove_worktree(self.repo, None)
        self.sm._remove_worktree(None, "/tmp/fake")

    def test_stale_worktree_replaced(self):
        """Creating a worktree at an existing path should replace it."""
        path = self.sm._create_worktree(self.repo, "feature/stale", "train-0")
        # Write a marker file
        marker = os.path.join(path, "marker.txt")
        with open(marker, "w") as f:
            f.write("old")
        _git(path, "add", ".")
        _git(path, "commit", "-m", "marker")

        # Delete branch from worktree tracking, simulate stale state
        self.sm._remove_worktree(self.repo, path)
        self.sm._git("branch", "-D", "feature/stale", cwd=self.repo)

        # Recreate at same path
        path2 = self.sm._create_worktree(self.repo, "feature/stale-v2", "train-0")
        self.assertEqual(path, path2)
        self.assertFalse(os.path.exists(os.path.join(path2, "marker.txt")))


class TestMergeFromWorktree(unittest.TestCase):
    """Test the full flow: create worktree → commit in worktree → merge in main repo."""

    def setUp(self):
        self.repo = _make_temp_repo()
        self.tmpdir = tempfile.mkdtemp(prefix="yamanote_cfg_")
        config.BACKLOG_DIR = os.path.join(self.tmpdir, "backlog")
        config.REVIEW_DIR = os.path.join(self.tmpdir, "review")
        config.LOGS_DIR = os.path.join(self.tmpdir, "logs")
        config.ACTIVITY_LOG = os.path.join(self.tmpdir, "activity.log")

        from orchestrator import StationManager
        self.sm = StationManager()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.repo, ignore_errors=True)
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        config.BACKLOG_DIR = _orig_backlog
        config.REVIEW_DIR = _orig_review
        config.LOGS_DIR = _orig_logs

    def test_merge_in_main_repo(self):
        """Commits made in a worktree should be mergeable from the main repo."""
        branch = "feature/merge-test"
        wt = self.sm._create_worktree(self.repo, branch, "train-0")

        # Make a change in the worktree
        new_file = os.path.join(wt, "feature.txt")
        with open(new_file, "w") as f:
            f.write("new feature\n")
        _git(wt, "add", ".")
        _git(wt, "commit", "-m", "Add feature")

        # Verify diff exists
        diff = self.sm._git_diff_trunk(branch, cwd=wt)
        self.assertIn("feature.txt", diff)

        head_before = _git(self.repo, "rev-parse", "HEAD")

        # Merge in main repo (the orchestrator's approach)
        self.sm._git("merge", "--no-ff", branch, cwd=self.repo)

        head_after = _git(self.repo, "rev-parse", "HEAD")
        self.assertNotEqual(head_before, head_after)

        # The file should now exist in the main repo
        self.assertTrue(os.path.exists(os.path.join(self.repo, "feature.txt")))

        # Clean up worktree then delete branch
        self.sm._remove_worktree(self.repo, wt)
        self.assertFalse(os.path.isdir(wt))
        self.sm._git("branch", "-D", branch, cwd=self.repo)
        self.assertFalse(self.sm._git_has_branch(branch, cwd=self.repo))

    def test_two_worktrees_parallel(self):
        """Two worktrees for different branches should coexist."""
        wt0 = self.sm._create_worktree(self.repo, "feature/alpha", "train-0")
        wt1 = self.sm._create_worktree(self.repo, "feature/beta", "train-1")

        # Both should exist and be on different branches
        self.assertNotEqual(wt0, wt1)
        self.assertEqual(_git(wt0, "branch", "--show-current"), "feature/alpha")
        self.assertEqual(_git(wt1, "branch", "--show-current"), "feature/beta")

        # Make different changes in each
        with open(os.path.join(wt0, "alpha.txt"), "w") as f:
            f.write("alpha\n")
        _git(wt0, "add", ".")
        _git(wt0, "commit", "-m", "Add alpha")

        with open(os.path.join(wt1, "beta.txt"), "w") as f:
            f.write("beta\n")
        _git(wt1, "add", ".")
        _git(wt1, "commit", "-m", "Add beta")

        # Merge alpha first
        self.sm._git("merge", "--no-ff", "feature/alpha", cwd=self.repo)
        self.sm._remove_worktree(self.repo, wt0)
        self.sm._git("branch", "-D", "feature/alpha", cwd=self.repo)

        # Merge beta second
        self.sm._git("merge", "--no-ff", "feature/beta", cwd=self.repo)
        self.sm._remove_worktree(self.repo, wt1)
        self.sm._git("branch", "-D", "feature/beta", cwd=self.repo)

        # Both files should exist in main
        self.assertTrue(os.path.exists(os.path.join(self.repo, "alpha.txt")))
        self.assertTrue(os.path.exists(os.path.join(self.repo, "beta.txt")))


class TestServiceRecoveryFlow(unittest.TestCase):
    """Test _train_phase_service_recovery with worktrees end-to-end."""

    def setUp(self):
        self.repo = _make_temp_repo()
        self.tmpdir = tempfile.mkdtemp(prefix="yamanote_cfg_")
        config.BACKLOG_DIR = os.path.join(self.tmpdir, "backlog")
        config.REVIEW_DIR = os.path.join(self.tmpdir, "review")
        config.LOGS_DIR = os.path.join(self.tmpdir, "logs")
        config.ACTIVITY_LOG = os.path.join(self.tmpdir, "activity.log")
        # Disable deploy so service_recovery doesn't try to push/restart
        self._orig_railway = config.RAILWAY_PROJECT
        self._orig_restart = config.SERVICE_RESTART_CMD
        config.RAILWAY_PROJECT = ""
        config.SERVICE_RESTART_CMD = ""

        from orchestrator import StationManager
        self.sm = StationManager()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.repo, ignore_errors=True)
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        config.BACKLOG_DIR = _orig_backlog
        config.REVIEW_DIR = _orig_review
        config.LOGS_DIR = _orig_logs
        config.RAILWAY_PROJECT = self._orig_railway
        config.SERVICE_RESTART_CMD = self._orig_restart

    def test_approved_triggers_merge_and_cleanup(self):
        """APPROVED feedback should merge in main repo, remove worktree, delete branch."""
        train = self.sm.trains[0]
        branch = "feature/approved-test"

        # Set up worktree
        wt = self.sm._create_worktree(self.repo, branch, train.train_id)
        train.repo_dir = self.repo
        train.working_dir = wt
        train.branch = branch
        train.spec_path = os.path.join(config.BACKLOG_DIR, "test_spec.json")

        # Make a commit in the worktree
        with open(os.path.join(wt, "approved.txt"), "w") as f:
            f.write("approved feature\n")
        _git(wt, "add", ".")
        _git(wt, "commit", "-m", "Add approved feature")

        # Write APPROVED feedback
        os.makedirs(config.REVIEW_DIR, exist_ok=True)
        feedback_path = os.path.join(
            config.REVIEW_DIR,
            f"{branch.replace('/', '_')}_feedback.md",
        )
        with open(feedback_path, "w") as f:
            f.write("APPROVED\nLooks good!\n")

        # Run service recovery
        self.sm._train_phase_service_recovery(train)

        # Verify: file merged into main repo
        self.assertTrue(os.path.exists(os.path.join(self.repo, "approved.txt")))

        # Verify: worktree removed
        self.assertFalse(os.path.isdir(wt))

        # Verify: branch deleted
        self.assertFalse(self.sm._git_has_branch(branch, cwd=self.repo))

        # Verify: train pipeline reset
        self.assertIsNone(train.branch)
        self.assertIsNone(train.working_dir)
        self.assertIsNone(train.repo_dir)

        # Verify: feedback file removed
        self.assertFalse(os.path.exists(feedback_path))

    def test_changes_requested_does_not_merge(self):
        """CHANGES_REQUESTED feedback should NOT trigger merge."""
        train = self.sm.trains[0]
        branch = "feature/rework-test"

        wt = self.sm._create_worktree(self.repo, branch, train.train_id)
        train.repo_dir = self.repo
        train.working_dir = wt
        train.branch = branch

        with open(os.path.join(wt, "wip.txt"), "w") as f:
            f.write("work in progress\n")
        _git(wt, "add", ".")
        _git(wt, "commit", "-m", "WIP")

        os.makedirs(config.REVIEW_DIR, exist_ok=True)
        feedback_path = os.path.join(
            config.REVIEW_DIR,
            f"{branch.replace('/', '_')}_feedback.md",
        )
        with open(feedback_path, "w") as f:
            f.write("CHANGES_REQUESTED\nFix the thing.\n")

        head_before = _git(self.repo, "rev-parse", "HEAD")
        self.sm._train_phase_service_recovery(train)
        head_after = _git(self.repo, "rev-parse", "HEAD")

        # Should NOT have merged
        self.assertEqual(head_before, head_after)
        # Worktree should still exist
        self.assertTrue(os.path.isdir(wt))
        # Branch should still be set
        self.assertEqual(train.branch, branch)

        # Clean up
        self.sm._remove_worktree(self.repo, wt)


class TestCleanupPaths(unittest.TestCase):
    """Test that entropy, rework-limit, and no-diff cleanup paths work."""

    def setUp(self):
        self.repo = _make_temp_repo()
        self.tmpdir = tempfile.mkdtemp(prefix="yamanote_cfg_")
        config.BACKLOG_DIR = os.path.join(self.tmpdir, "backlog")
        config.REVIEW_DIR = os.path.join(self.tmpdir, "review")
        config.LOGS_DIR = os.path.join(self.tmpdir, "logs")
        config.ACTIVITY_LOG = os.path.join(self.tmpdir, "activity.log")

        from orchestrator import StationManager
        self.sm = StationManager()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.repo, ignore_errors=True)
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        config.BACKLOG_DIR = _orig_backlog
        config.REVIEW_DIR = _orig_review
        config.LOGS_DIR = _orig_logs

    def _setup_train_with_worktree(self, branch="feature/cleanup-test"):
        train = self.sm.trains[0]
        wt = self.sm._create_worktree(self.repo, branch, train.train_id)
        train.repo_dir = self.repo
        train.working_dir = wt
        train.branch = branch
        return train, wt

    def test_entropy_cleanup(self):
        """_fire_conductor_entropy should remove worktree and delete branch."""
        train, wt = self._setup_train_with_worktree("feature/entropy-test")

        self.sm._fire_conductor_entropy(train, "feature/entropy-test", cwd=wt)

        self.assertFalse(os.path.isdir(wt))
        self.assertFalse(self.sm._git_has_branch("feature/entropy-test", cwd=self.repo))
        self.assertIsNone(train.branch)

    def test_inspector_no_diff_cleanup(self):
        """Inspector phase with no diff should remove worktree and delete branch."""
        train, wt = self._setup_train_with_worktree("feature/empty-branch")
        # No commits on the branch, so diff is empty

        # Need spec_path set to avoid NoneType error
        spec_path = os.path.join(config.BACKLOG_DIR, "empty_spec.json")
        train.spec_path = spec_path

        self.sm._train_phase_inspector(train)

        self.assertFalse(os.path.isdir(wt))
        self.assertFalse(self.sm._git_has_branch("feature/empty-branch", cwd=self.repo))
        self.assertIsNone(train.branch)

    def test_rework_limit_cleanup(self):
        """Exceeding rework limit should remove worktree and delete branch."""
        train, wt = self._setup_train_with_worktree("feature/rework-limit")
        train.rework_count = config.MAX_REWORK_ATTEMPTS  # will be incremented to exceed

        # Create a CHANGES_REQUESTED feedback file
        os.makedirs(config.REVIEW_DIR, exist_ok=True)
        branch = "feature/rework-limit"
        feedback_path = os.path.join(
            config.REVIEW_DIR,
            f"{branch.replace('/', '_')}_feedback.md",
        )
        with open(feedback_path, "w") as f:
            f.write("CHANGES_REQUESTED\nStill broken.\n")

        spec_path = os.path.join(config.BACKLOG_DIR, "rework_spec.json")
        with open(spec_path, "w") as f:
            json.dump({"title": "rework-limit", "description": "test"}, f)
        train.spec_path = spec_path

        self.sm._train_phase_rework(train)

        self.assertFalse(os.path.isdir(wt))
        self.assertFalse(self.sm._git_has_branch(branch, cwd=self.repo))
        self.assertIsNone(train.branch)


if __name__ == "__main__":
    unittest.main()
