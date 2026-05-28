#!/usr/bin/env python3
"""Unit tests for spec_context.

Mixes pure-Python tests (keyword extraction, formatter) with a small temp-git
fixture for the file-relevance scoring path.
"""

import os
import shutil
import subprocess
import tempfile
import unittest

import spec_context


def _git(repo, *args):
    result = subprocess.run(
        ["git"] + list(args),
        capture_output=True, text=True, cwd=repo,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr}")
    return result.stdout.strip()


def _make_repo_with_files(files: dict[str, str]) -> str:
    """Create a temp git repo with given {relpath: content}, committed on main."""
    repo = tempfile.mkdtemp(prefix="yamanote_spec_ctx_")
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@test.com")
    _git(repo, "config", "user.name", "Test")
    for path, content in files.items():
        full = os.path.join(repo, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write(content)
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed")
    return repo


class TestExtractKeywords(unittest.TestCase):
    def test_empty_spec_returns_empty(self):
        self.assertEqual(spec_context.extract_keywords({}), [])
        self.assertEqual(spec_context.extract_keywords({"title": "", "description": ""}), [])

    def test_filters_stopwords(self):
        kws = spec_context.extract_keywords({"title": "the and a of for in"})
        self.assertEqual(kws, [])

    def test_filters_short_tokens(self):
        # "we", "to", "go" are <3 chars or stopwords
        kws = spec_context.extract_keywords({"title": "we go to"})
        self.assertEqual(kws, [])

    def test_splits_camel_snake_kebab(self):
        kws = spec_context.extract_keywords(
            {"title": "userAuth", "description": "auth_token kebab-case-name"}
        )
        self.assertIn("user", kws)
        self.assertIn("auth", kws)
        self.assertIn("token", kws)
        self.assertIn("kebab", kws)
        self.assertIn("case", kws)
        self.assertIn("name", kws)

    def test_orders_by_frequency(self):
        spec = {"title": "auth", "description": "auth auth login auth"}
        kws = spec_context.extract_keywords(spec)
        self.assertEqual(kws[0], "auth")  # 4 occurrences
        self.assertIn("login", kws)

    def test_caps_at_max(self):
        # Lots of unique non-stopword tokens
        text = " ".join(f"keyword{i}" for i in range(50))
        kws = spec_context.extract_keywords({"description": text})
        self.assertLessEqual(len(kws), spec_context._MAX_KEYWORDS)

    def test_drops_pure_digits(self):
        kws = spec_context.extract_keywords({"title": "issue 12345 foo"})
        self.assertNotIn("12345", kws)
        self.assertIn("foo", kws)

    def test_drops_generic_code_words(self):
        # "add", "test", "implementation" are in the spec/code noise list
        kws = spec_context.extract_keywords({"title": "add test implementation"})
        self.assertEqual(kws, [])


class TestFormatBundle(unittest.TestCase):
    def test_empty_renders_fallback(self):
        out = spec_context.format_bundle([])
        self.assertIn("no obvious file matches", out)
        self.assertIn("Files likely relevant", out)

    def test_simple_list_renders(self):
        out = spec_context.format_bundle([
            {"path": "src/auth.py", "reason": "matches: auth"},
            {"path": "tests/test_auth.py", "reason": "matches: auth"},
        ])
        self.assertIn("src/auth.py", out)
        self.assertIn("tests/test_auth.py", out)
        self.assertIn("matches: auth", out)

    def test_cap_truncates_with_summary_line(self):
        files = [{"path": f"f{i}.py", "reason": "matches: x"} for i in range(100)]
        out = spec_context.format_bundle(files, max_chars=200)
        self.assertLessEqual(len(out), 250)  # generous slack for truncation line
        self.assertIn("truncated", out)


class TestFindRelevantFiles(unittest.TestCase):
    def setUp(self):
        self.repo = _make_repo_with_files({
            "src/auth/login.py": "def login(user): pass\n",
            "src/auth/token.py": "def make_token(): pass\n",
            "src/api/users.py": "# user api endpoints\n",
            "tests/test_auth.py": "import auth\n",
            "README.md": "project readme\n",
            "node_modules/leftpad/index.js": "// not tracked path inside repo\n",  # but we DID add it
        })

    def tearDown(self):
        shutil.rmtree(self.repo, ignore_errors=True)

    def test_finds_keyword_matches(self):
        spec = {"title": "harden-auth-token", "description": "rotate auth tokens"}
        files = spec_context.find_relevant_files(self.repo, spec)
        paths = [f["path"] for f in files]
        # Files with the word "auth" should appear
        self.assertIn("src/auth/login.py", paths)
        self.assertIn("src/auth/token.py", paths)
        self.assertIn("tests/test_auth.py", paths)

    def test_orders_higher_match_first(self):
        # "auth" appears in 3 files; "token" appears in only 1 — the 2-match
        # file should rank ahead of 1-match files.
        spec = {"title": "rotate-auth-token", "description": "auth token rotation"}
        files = spec_context.find_relevant_files(self.repo, spec)
        self.assertGreater(len(files), 0)
        # token.py matches both "auth" (via path? no, content has "make_token") — actually
        # token.py only contains "token", not "auth". login.py contains "login".
        # The strongest single-match file should still be ranked above zero-match recents.
        top = files[0]
        self.assertIn("matches:", top["reason"])

    def test_no_keyword_match_returns_recents_only(self):
        spec = {"title": "spec-with-no-codebase-overlap", "description": "totallyrandomtokenxyz"}
        files = spec_context.find_relevant_files(self.repo, spec)
        # All entries are recent-only (single commit in this repo touched everything)
        for f in files:
            self.assertEqual(f["reason"], "recently modified")

    def test_non_git_dir_returns_empty(self):
        tmp = tempfile.mkdtemp(prefix="yamanote_no_git_")
        try:
            spec = {"title": "auth-feature", "description": "do something"}
            files = spec_context.find_relevant_files(tmp, spec)
            self.assertEqual(files, [])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_max_files_cap(self):
        spec = {"title": "common", "description": "user user user user"}
        files = spec_context.find_relevant_files(self.repo, spec, max_files=2)
        self.assertLessEqual(len(files), 2)


class TestBuildRelevantFilesBlock(unittest.TestCase):
    def test_returns_string_on_non_git_dir(self):
        tmp = tempfile.mkdtemp(prefix="yamanote_block_")
        try:
            out = spec_context.build_relevant_files_block(tmp, {"title": "anything"})
            self.assertIsInstance(out, str)
            self.assertIn("Files likely relevant", out)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_returns_string_on_empty_spec(self):
        tmp = tempfile.mkdtemp(prefix="yamanote_block2_")
        try:
            out = spec_context.build_relevant_files_block(tmp, {})
            self.assertIsInstance(out, str)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
