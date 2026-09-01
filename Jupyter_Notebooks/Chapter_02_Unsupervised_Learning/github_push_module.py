"""Reusable one-call GitHub publisher for Python projects."""

import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


REPO_URL = "https://github.com/hzolfaghari2022/machine-learning.git"
BRANCH = "main"
SYNC_DIR = Path.home() / "machine_learning_github_sync"


class GitPushError(RuntimeError):
    pass


def _git(folder, *args, check=True):
    try:
        result = subprocess.run(
            ["git", *args], cwd=folder, text=True, capture_output=True, check=False
        )
    except FileNotFoundError as error:
        raise GitPushError("Git is not installed or is not on PATH.") from error
    if check and result.returncode:
        raise GitPushError(result.stderr.strip() or result.stdout.strip())
    return result


def _repo():
    if not SYNC_DIR.exists():
        result = _git(SYNC_DIR.parent, "clone", REPO_URL, str(SYNC_DIR), check=False)
        if result.returncode:
            raise GitPushError(result.stderr.strip() or result.stdout.strip())

    if not (SYNC_DIR / ".git").is_dir():
        raise GitPushError(f"Not a Git repository: {SYNC_DIR}")

    remote = _git(SYNC_DIR, "remote", "get-url", "origin").stdout.strip()
    expected = REPO_URL.removesuffix(".git").lower()
    actual = remote.replace("git@github.com:", "https://github.com/")
    if actual.removesuffix(".git").lower() != expected:
        raise GitPushError(f"Wrong GitHub origin: {remote}")

    if _git(SYNC_DIR, "ls-remote", "--heads", "origin", BRANCH).stdout.strip():
        _git(SYNC_DIR, "checkout", BRANCH)
        _git(SYNC_DIR, "pull", "--rebase", "origin", BRANCH)
    else:
        _git(SYNC_DIR, "checkout", "-B", BRANCH)
    return SYNC_DIR


def _copy(source, destination, staged):
    """Copy one file/folder and collect exact Git paths for staging."""
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        staged.append(destination.relative_to(SYNC_DIR).as_posix())
    elif source.is_dir():
        for file in source.rglob("*"):
            if file.is_file():
                _copy(file, destination / file.relative_to(source), staged)
    else:
        raise GitPushError(f"Result path does not exist: {source}")


def push_run(script_file, project_name, extra_paths=()):
    """Push root-level Python files and selected results to one repo folder."""
    project = Path(script_file).resolve().parent
    repo = _repo()
    folder_name = re.sub(r"[^A-Za-z0-9._-]+", "_", project_name).strip("._")
    if not folder_name:
        raise GitPushError("project_name must contain letters or numbers.")

    sources = sorted(project.glob("*.py"))
    sources += [project / path for path in extra_paths]
    staged = []
    for source in sources:
        _copy(source, repo / folder_name / source.name, staged)

    if not _git(repo, "config", "user.name", check=False).stdout.strip() or not _git(
        repo, "config", "user.email", check=False
    ).stdout.strip():
        raise GitPushError("Configure Git user.name and user.email once, then rerun.")

    _git(repo, "add", "--", *staged)
    changed = _git(repo, "diff", "--cached", "--quiet", "--", *staged, check=False)
    if changed.returncode == 1:
        message = f"Update {folder_name} ({datetime.now():%Y-%m-%d %H:%M})"
        _git(repo, "commit", "-m", message)
    elif changed.returncode:
        raise GitPushError("Git could not inspect the selected files.")

    pushed = _git(repo, "push", "-u", "origin", BRANCH, check=False)
    if pushed.returncode:
        detail = pushed.stderr.strip() or pushed.stdout.strip()
        raise GitPushError("GitHub push failed. Sign in to Git, then rerun.\n" + detail)
    print(f"GitHub updated: {REPO_URL} ({folder_name})")


def try_push(script_file, project_name, extra_paths=()):
    """Push without hiding results when Git/GitHub has a problem."""
    try:
        push_run(script_file, project_name, extra_paths)
    except GitPushError as error:
        print(f"Results were saved, but GitHub push failed:\n{error}")


def push_to_github(script_file):
    """Automatically detect the project name and common result files/folders."""
    project = Path(script_file).resolve().parent
    result_folders = {
        "result", "results", "figure", "figures", "figs", "plots",
        "output", "outputs", "report_figs", "presentation_figs",
    }
    result_extensions = {
        ".png", ".jpg", ".jpeg", ".svg", ".pdf", ".txt", ".csv",
        ".json", ".npy", ".npz", ".pkl", ".joblib", ".pt", ".pth",
    }
    detected = [
        item.name
        for item in project.iterdir()
        if (item.is_dir() and item.name.lower() in result_folders)
        or (item.is_file() and item.suffix.lower() in result_extensions)
    ]
    try_push(script_file, Path(script_file).stem, detected)
