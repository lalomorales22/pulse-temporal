"""Git event source adapter for the PULSE daemon.

Reads git log and feeds commit/branch activity as temporal events.
Supports any local git repository.

Usage:
    from pulse_temporal.adapters import GitAdapter
    from pulse_temporal.daemon import PulseDaemon

    daemon = PulseDaemon()
    git = GitAdapter("/path/to/repo")
    git.sync(daemon)  # logs recent git events to daemon
"""

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict


class GitAdapter:
    """Feeds git repository activity into the PULSE daemon event stream."""

    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path).resolve()
        if not (self.repo_path / ".git").exists():
            raise ValueError(f"Not a git repository: {self.repo_path}")

    def _run_git(self, *args: str) -> str:
        """Run a git command and return stdout."""
        result = subprocess.run(
            ["git", "-C", str(self.repo_path)] + list(args),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
        return result.stdout.strip()

    def get_commits(self, since: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Get recent commits as structured events.

        Args:
            since: ISO 8601 timestamp or git date string (e.g., "2 days ago").
            limit: Maximum number of commits to return.
        """
        fmt = "%H%n%aI%n%s%n%an%n---"
        cmd = ["log", f"--format={fmt}", f"-{limit}"]
        if since:
            cmd.append(f"--since={since}")

        raw = self._run_git(*cmd)
        if not raw:
            return []

        commits = []
        blocks = raw.split("---\n")
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            lines = block.split("\n")
            if len(lines) < 4:
                continue
            commits.append({
                "hash": lines[0],
                "timestamp": lines[1],
                "message": lines[2],
                "author": lines[3],
            })
        return commits

    def get_branch_info(self) -> Dict:
        """Get current branch and recent branch activity."""
        branch = self._run_git("rev-parse", "--abbrev-ref", "HEAD")
        # Count commits ahead/behind origin if remote exists
        try:
            status = self._run_git("rev-list", "--left-right", "--count", f"origin/{branch}...HEAD")
            behind, ahead = status.split("\t")
            behind, ahead = int(behind), int(ahead)
        except (RuntimeError, ValueError):
            behind, ahead = 0, 0

        return {
            "branch": branch,
            "commits_ahead": ahead,
            "commits_behind": behind,
        }

    def get_activity_summary(self, hours: int = 24) -> Dict:
        """Summarize git activity over a time window."""
        since = f"{hours} hours ago"
        commits = self.get_commits(since=since, limit=200)

        if not commits:
            return {
                "period_hours": hours,
                "total_commits": 0,
                "authors": [],
                "activity_level": "none",
            }

        authors = list({c["author"] for c in commits})
        count = len(commits)

        if count > 20:
            level = "intense"
        elif count > 10:
            level = "active"
        elif count > 3:
            level = "moderate"
        else:
            level = "light"

        return {
            "period_hours": hours,
            "total_commits": count,
            "authors": authors,
            "activity_level": level,
            "latest_commit": commits[0]["message"] if commits else None,
            "latest_timestamp": commits[0]["timestamp"] if commits else None,
        }

    def sync(self, daemon, since: Optional[str] = None, limit: int = 50):
        """Sync recent git events into the PULSE daemon.

        Args:
            daemon: PulseDaemon instance to log events to.
            since: Only sync commits after this timestamp.
            limit: Max commits to sync.
        """
        commits = self.get_commits(since=since, limit=limit)
        logged = 0
        for commit in reversed(commits):  # oldest first
            daemon.log_event(
                event_type="git_commit",
                timestamp=commit["timestamp"],
                metadata={
                    "hash": commit["hash"][:8],
                    "message": commit["message"],
                    "author": commit["author"],
                },
            )
            logged += 1
        return logged

    def get_file_churn(self, hours: int = 24, limit: int = 10) -> List[Dict]:
        """Get most-changed files in the time window (hot spots)."""
        since = f"{hours} hours ago"
        try:
            raw = self._run_git(
                "log", f"--since={since}", "--name-only", "--format=",
            )
        except RuntimeError:
            return []

        if not raw:
            return []

        counts: dict = {}
        for line in raw.split("\n"):
            line = line.strip()
            if line:
                counts[line] = counts.get(line, 0) + 1

        sorted_files = sorted(counts.items(), key=lambda x: -x[1])[:limit]
        return [{"file": f, "changes": c} for f, c in sorted_files]

    def __repr__(self) -> str:
        return f"GitAdapter(repo='{self.repo_path}')"
