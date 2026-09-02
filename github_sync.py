"""Copy this project into a local clone, commit exact files, and push them."""

import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPO_URL = "https://github.com/hzolfaghari2022/machine-learning.git"
BRANCH = "main"


class GitError(RuntimeError):
    pass


def git(folder, *args, check=True):
    try:
        result = subprocess.run(["git", *args], cwd=folder, text=True,
                                capture_output=True, check=False)
    except FileNotFoundError as error:
        raise GitError("Git is not installed or is not on PATH.") from error
    if check and result.returncode:
        raise GitError(result.stderr.strip() or result.stdout.strip())
    return result


def repository(project):
    repo = project / "github_sync_machine_learning"
    if not repo.exists():
        result = git(project, "clone", REPO_URL, str(repo), check=False)
        if result.returncode:
            raise GitError(result.stderr.strip() or result.stdout.strip())
    if not (repo / ".git").is_dir():
        raise GitError(f"Not a Git repository: {repo}")
    if git(repo, "remote", "get-url", "origin").stdout.strip() != REPO_URL:
        raise GitError("The synchronization folder points to the wrong repository.")

    remote = git(repo, "ls-remote", "--heads", "origin", BRANCH).stdout.strip()
    if remote:
        git(repo, "checkout", BRANCH)
        git(repo, "pull", "--rebase", "origin", BRANCH)
    else:
        git(repo, "checkout", "-B", BRANCH)
    return repo


def sync_to_github(project, files):
    """Publish only the supplied project files; unrelated files are untouched."""
    repo = repository(Path(project))
    relative_files = []
    for source in map(Path, files):
        relative = source.relative_to(project)
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        relative_files.append(relative.as_posix())

    if not git(repo, "config", "user.name", check=False).stdout.strip() or not git(
        repo, "config", "user.email", check=False
    ).stdout.strip():
        raise GitError("Configure git user.name and user.email once, then rerun.")

    git(repo, "add", "--", *relative_files)
    changed = git(repo, "diff", "--cached", "--quiet", "--", *relative_files,
                  check=False).returncode
    if changed == 1:
        time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        git(repo, "commit", "-m", f"Update Iris experiment ({time})")
    elif changed:
        raise GitError("Git could not inspect the staged files.")

    pushed = git(repo, "push", "-u", "origin", BRANCH, check=False)
    if pushed.returncode:
        raise GitError("Sign in with Git Credential Manager, then rerun.\n" +
                       (pushed.stderr.strip() or pushed.stdout.strip()))
    print(f"\nGitHub updated: {REPO_URL}")
