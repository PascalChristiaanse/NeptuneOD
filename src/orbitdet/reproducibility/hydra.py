import subprocess
from omegaconf import OmegaConf


def _git_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        text=True,
    ).strip()


def _git_commit_safe():
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"],
        text=True,
    ).strip()

    commit = _git_commit()
    return f"{commit}_dirty" if dirty else commit


def register_resolvers():
    OmegaConf.register_new_resolver("git_commit", _git_commit_safe)
