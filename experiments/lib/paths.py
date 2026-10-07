# paths.py — repository locations and import setup for experiment scripts.
#
# Large outputs and the micro-blossom checkout are resolved in this order:
# environment variable, then .magicfirm.env in the repo root (written by
# setup_local.sh, not committed), then the in-repo default.
#
#   MAGICFIRM_OUT           large generated data        default <repo>/out
#   MAGICFIRM_MB_ROOT       micro-blossom checkout      default <repo>/micro-blossom
#   MAGICFIRM_MB_CONTAINER  toolchain container name    default mb-magicfirm-<user>
#
# Every experiment script starts with:
#   import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "lib"))
#   import paths; paths.setup_imports()

import getpass
import os
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
EXPERIMENTS = REPO / "experiments"
LIB = EXPERIMENTS / "lib"
ALGORITHMS = REPO / "algorithms"
CULTIV_SRC = REPO / "magic_state_cultivation" / "upstream" / "src"
CULTIV_TOOLS = REPO / "magic_state_cultivation"        # circuit_generator and patches
CIRCUITS = EXPERIMENTS / "data" / "circuits"
RESULT = EXPERIMENTS / "result"
CONFIG_FILE = REPO / ".magicfirm.env"


def _read_config(path: pathlib.Path) -> dict[str, str]:
    """KEY=VALUE lines; '#' comments; ~ and $VAR expanded in values."""
    cfg = {}
    if not path.is_file():
        return cfg
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        cfg[k.strip()] = os.path.expandvars(os.path.expanduser(v.strip().strip('"').strip("'")))
    return cfg


_CFG = _read_config(CONFIG_FILE)


def setting(name: str, default: str) -> str:
    """Environment variable, else .magicfirm.env, else default."""
    return os.environ.get(name) or _CFG.get(name) or default


def _dir(name: str, default: pathlib.Path) -> pathlib.Path:
    p = pathlib.Path(setting(name, str(default)))
    # Resolve symlinks (e.g. out -> /data/...) so the same absolute path is
    # valid inside the toolchain container, which mounts the real directory.
    return p.resolve() if p.exists() else p


OUT = _dir("MAGICFIRM_OUT", REPO / "out")
MB_ROOT = _dir("MAGICFIRM_MB_ROOT", REPO / "micro-blossom")
MB_CONTAINER = setting("MAGICFIRM_MB_CONTAINER", f"mb-magicfirm-{getpass.getuser()}")


def rel(p) -> str:
    """Path relative to the repo when it lies inside it (for result metadata), else absolute."""
    p = pathlib.Path(p)
    return str(p.relative_to(REPO)) if p.is_relative_to(REPO) else str(p)


def setup_imports() -> None:
    """Put the cultivation sources, algorithms/ and experiments/lib on sys.path
    (first entries, in that priority order: lib, algorithms, cultiv)."""
    for p in (CULTIV_TOOLS, CULTIV_SRC, ALGORITHMS, LIB):
        s = str(p)
        if s in sys.path:
            sys.path.remove(s)
        sys.path.insert(0, s)


def summary() -> str:
    return "\n".join(f"{k:14s} {v}" for k, v in (
        ("REPO", REPO), ("OUT", OUT), ("MB_ROOT", MB_ROOT), ("MB_CONTAINER", MB_CONTAINER),
        ("config file", CONFIG_FILE if CONFIG_FILE.is_file() else f"{CONFIG_FILE} (absent)")))


if __name__ == "__main__":
    print(summary())
