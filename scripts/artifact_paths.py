"""兼容标准发布目录与早期 GitHub 扁平目录，拒绝混用两套输入。"""
from pathlib import Path


def artifact_root(root: Path) -> Path:
    root = Path(root)
    candidates = [p for p in (root / 'research_artifacts', root) if (p / 'benchmarks').is_dir()]
    if len(candidates) > 1:
        raise ValueError('存在两套 benchmarks，请只保留一套实验数据布局')
    if not candidates:
        raise FileNotFoundError('缺少 research_artifacts/benchmarks 或 benchmarks')
    return candidates[0]


def package_path(root: Path, relative: str) -> Path:
    parts = Path(relative.replace('\\', '/')).parts
    if parts and parts[0] == 'research_artifacts':
        return artifact_root(root).joinpath(*parts[1:])
    return Path(root).joinpath(*parts)
