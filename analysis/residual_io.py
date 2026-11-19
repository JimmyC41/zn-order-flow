"""File integrity and publication of diagnostic outputs."""

from contextlib import contextmanager
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory


# Four official model columns ordered by session and order, independent of file encoding.
VALIDATION_DIGEST = "5504dae309a4a308b8e142267cd1c3ddb1e35d29532870a2e069ec963197ce1d"


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@contextmanager
def new_directory(destination: Path):
    """Publish completed results in one rename; failed runs leave no partial output."""
    if destination.exists():
        raise FileExistsError(f"Output directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as stage:
        output = Path(stage) / "results"
        output.mkdir()
        yield output
        if destination.exists():
            raise FileExistsError(f"Output directory already exists: {destination}")
        output.rename(destination)
