"""组装 openviking-openvino 的 Windows 与 Linux 发布包。"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

PROJECT_PREFIX = "openviking-openvino"
PLATFORMS = ("windows", "linux")
COMMON_ITEMS = (
    "install.py",
    "messages.py",
    "install.ps1",
    "README.md",
    "README.en.md",
    "CHANGELOG.md",
    "VERSION.toml",
    "LICENSE",
    "patches",
    "http-server",
    "configs",
    "plugins",
    "docs",
    "models",
)
WINDOWS_ONLY_ITEMS = ("install.cmd",)
IGNORE_PATTERNS = (
    "__pycache__",
    "*.pyc",
    ".ruff_cache",
    ".pytest_cache",
)


@dataclass(frozen=True)
class ReleasePackage:
    platform: str
    root: Path
    archive: Path
    checksum: Path
    file_count: int


def package_items(platform: str) -> tuple[str, ...]:
    """返回指定平台应打包的项目条目。"""
    if platform == "windows":
        return COMMON_ITEMS + WINDOWS_ONLY_ITEMS
    if platform == "linux":
        return COMMON_ITEMS
    raise ValueError(f"unsupported release platform: {platform}")


def _validate_tag(tag: str) -> None:
    if not tag or tag in {".", ".."} or "/" in tag or "\\" in tag or tag != tag.strip():
        raise ValueError(f"unsafe release tag: {tag!r}")


def _copy_item(source_root: Path, item: str, destination_root: Path) -> None:
    source = source_root / item
    destination = destination_root / item
    if source.is_dir():
        shutil.copytree(
            source,
            destination,
            ignore=shutil.ignore_patterns(*IGNORE_PATTERNS),
        )
    elif source.is_file():
        shutil.copy2(source, destination)
    else:
        raise FileNotFoundError(f"release item is missing: {source}")


def _prune_caches(root: Path) -> None:
    for cache_dir in sorted(root.rglob("__pycache__"), reverse=True):
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir)
    for compiled in root.rglob("*.pyc"):
        compiled.unlink()


def _files_under(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file())


def _write_archive(root: Path, archive: Path) -> None:
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in _files_under(root):
            bundle.write(path, (Path(root.name) / path.relative_to(root)).as_posix())
    with zipfile.ZipFile(archive) as bundle:
        failed = bundle.testzip()
    if failed is not None:
        raise RuntimeError(f"corrupt release archive member: {failed}")


def _write_checksum(root: Path, checksum: Path) -> int:
    lines = []
    files = _files_under(root)
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(root).as_posix()}")
    checksum.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return len(files)


def assemble_release(
    tag: str,
    source_root: Path | str = Path("."),
    dist_root: Path | str = Path("dist"),
) -> tuple[ReleasePackage, ...]:
    """组装两个平台目录、压缩包与 sha256 清单，返回产物信息。"""
    _validate_tag(tag)
    source = Path(source_root).expanduser().resolve()
    dist = Path(dist_root).expanduser().resolve()
    if not source.is_dir():
        raise NotADirectoryError(f"release source is not a directory: {source}")
    dist.mkdir(parents=True, exist_ok=True)

    packages: list[ReleasePackage] = []
    for platform in PLATFORMS:
        root = dist / f"{PROJECT_PREFIX}-{platform}-{tag}"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)

        for item in package_items(platform):
            _copy_item(source, item, root)
        _prune_caches(root)

        archive = dist / f"{root.name}.zip"
        checksum = dist / f"{root.name}.sha256"
        _write_archive(root, archive)
        file_count = _write_checksum(root, checksum)
        packages.append(
            ReleasePackage(
                platform=platform,
                root=root,
                archive=archive,
                checksum=checksum,
                file_count=file_count,
            )
        )
    return tuple(packages)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="release tag, for example v0.1.0")
    parser.add_argument("--source", default=".", help="repository root")
    parser.add_argument("--dist", default="dist", help="output directory")
    args = parser.parse_args(argv)

    packages = assemble_release(args.tag, args.source, args.dist)
    for package in packages:
        has_install_cmd = (package.root / "install.cmd").is_file()
        print(
            f"{package.platform}: files={package.file_count} "
            f"install.cmd={'yes' if has_install_cmd else 'no'} "
            f"archive={package.archive} checksum={package.checksum}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
