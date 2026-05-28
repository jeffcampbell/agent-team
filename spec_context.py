"""Deterministic per-spec file shortlist for Conductor prompts.

Conductor's most expensive cost is re-exploring the codebase on every launch
to figure out which files matter for the spec at hand. This module pre-computes
a tight shortlist using `git grep` + recent-modification signals, formatted as
a markdown block that gets injected into CONDUCTOR_PROMPT.

No LLM call. Safe to call on every Conductor launch — typical cost is a single
`git ls-files`, `git log`, and `git grep` invocation against tracked files only.
"""

import os
import re
import subprocess
from collections import Counter
from typing import Iterable

# Standard English stopwords + common spec/code words that match too broadly.
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "do", "does", "for", "from", "had", "has", "have", "if", "in", "into",
    "is", "it", "its", "of", "on", "or", "that", "the", "their", "then",
    "there", "these", "this", "those", "to", "was", "were", "when", "where",
    "which", "while", "will", "with", "would", "you", "your", "we", "our",
    # Spec/code noise — too generic to be useful as keywords
    "add", "use", "make", "fix", "new", "should", "must", "may", "can",
    "spec", "feature", "bug", "code", "file", "files", "function", "method",
    "test", "tests", "all", "any", "not", "set", "get", "run", "via", "also",
    "create", "update", "remove", "delete", "implement", "implementation",
    "support", "ensure", "allow", "enable", "include", "including",
})

_KEYWORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]{2,}")
_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|[_\-]+")
_MAX_KEYWORDS = 12
_MAX_FILES = 15
_RECENT_COMMITS_FOR_BONUS = 5
_GIT_TIMEOUT = 10


def extract_keywords(spec: dict) -> list[str]:
    """Pull a short list of distinct keywords from a spec's title + description.

    Splits camelCase, snake_case, and kebab-case so "user_auth" → ["user", "auth"].
    Filters stopwords, sub-3-char tokens, and pure-digit tokens. Returns up to
    _MAX_KEYWORDS, ordered by frequency.
    """
    text = " ".join(str(spec.get(k, "")) for k in ("title", "description"))
    if not text.strip():
        return []
    # Split compounds then re-tokenise so we catch both "userAuth" and "user_auth"
    text = _CAMEL_SPLIT_RE.sub(" ", text)
    tokens = (m.group(0).lower() for m in _KEYWORD_RE.finditer(text))
    counts: Counter[str] = Counter()
    for tok in tokens:
        if tok in _STOPWORDS or tok.isdigit():
            continue
        counts[tok] += 1
    return [w for w, _ in counts.most_common(_MAX_KEYWORDS)]


def _run_git(args: list[str], cwd: str) -> tuple[int, str]:
    # Best-effort: any failure (timeout, missing git, mocked subprocess in tests,
    # unexpected output shape, etc.) returns "no result" so the bundle just
    # degrades to empty and the Conductor falls back to its own exploration.
    try:
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
        )
        return result.returncode, result.stdout
    except Exception:
        return -1, ""


def _recently_modified(working_dir: str, limit: int = _RECENT_COMMITS_FOR_BONUS) -> set[str]:
    rc, out = _run_git(
        ["log", f"-{limit}", "--name-only", "--pretty=format:"],
        cwd=working_dir,
    )
    if rc != 0:
        return set()
    return {line for line in (l.strip() for l in out.splitlines()) if line}


def _grep_for_keywords(working_dir: str, keywords: list[str]) -> dict[str, list[str]]:
    """Return file → list of matching keywords. Uses git grep against tracked files only."""
    if not keywords:
        return {}
    # -l: list filenames only. -i: case-insensitive. -F: fixed strings (no regex).
    # -w: match whole words so "auth" doesn't match "author". One -e per keyword.
    # -I skips binary files (PNGs, etc.); -w whole-word so "auth" doesn't match "author".
    args = ["grep", "-l", "-I", "-i", "-F", "-w"]
    for kw in keywords:
        args.extend(["-e", kw])
    rc, out = _run_git(args, cwd=working_dir)
    if rc != 0:
        return {}
    files = [line.strip() for line in out.splitlines() if line.strip()]
    if not files:
        return {}
    # Re-score each candidate by re-grepping per keyword to attribute matches.
    matches: dict[str, list[str]] = {f: [] for f in files}
    for kw in keywords:
        rc, out = _run_git(
            ["grep", "-l", "-I", "-i", "-F", "-w", "-e", kw],
            cwd=working_dir,
        )
        if rc != 0:
            continue
        for line in out.splitlines():
            path = line.strip()
            if path in matches:
                matches[path].append(kw)
    return matches


def find_relevant_files(working_dir: str, spec: dict, max_files: int = _MAX_FILES) -> list[dict]:
    """Return a ranked list of {path, reason} dicts. Empty if nothing matches.

    Ranking: more keyword matches > recently modified > alphabetical.
    """
    keywords = extract_keywords(spec)
    keyword_matches = _grep_for_keywords(working_dir, keywords)
    recent = _recently_modified(working_dir)

    scored: list[tuple[int, int, str, list[str], bool]] = []
    seen: set[str] = set()
    for path, matched_kws in keyword_matches.items():
        seen.add(path)
        is_recent = path in recent
        scored.append((len(matched_kws), 1 if is_recent else 0, path, matched_kws, is_recent))

    # Recent-only files (no keyword match) get a tail entry.
    for path in recent - seen:
        scored.append((0, 1, path, [], True))

    scored.sort(key=lambda t: (-t[0], -t[1], t[2]))
    out: list[dict] = []
    for _, _, path, matched_kws, is_recent in scored[:max_files]:
        if matched_kws:
            reason = f"matches: {', '.join(matched_kws[:4])}"
            if is_recent:
                reason += "; recent"
        else:
            reason = "recently modified"
        out.append({"path": path, "reason": reason})
    return out


def format_bundle(files: list[dict], max_chars: int = 1500) -> str:
    """Render the file list as a markdown block for prompt injection."""
    header = "## Files likely relevant to this spec"
    if not files:
        return f"{header}\n(no obvious file matches — explore as needed)"
    lines = [header]
    used = len(header) + 1
    for entry in files:
        line = f"- {entry['path']} ({entry['reason']})"
        if used + len(line) + 1 > max_chars:
            lines.append(f"- … ({len(files) - (len(lines) - 1)} more truncated)")
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)


def build_relevant_files_block(working_dir: str, spec: dict) -> str:
    """One-shot helper: keywords → files → formatted block. Safe on any directory."""
    try:
        files = find_relevant_files(working_dir, spec)
    except (OSError, subprocess.SubprocessError):
        files = []
    return format_bundle(files)
