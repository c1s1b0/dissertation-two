"""发布目录定位测试：标准、扁平与歧义布局。"""
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


class ArtifactPaths(unittest.TestCase):
    def test_standard_layout(self):
        from artifact_paths import artifact_root, package_path
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'research_artifacts/benchmarks').mkdir(parents=True)
            self.assertEqual(artifact_root(root), root / 'research_artifacts')
            self.assertEqual(package_path(root, 'research_artifacts/benchmarks/x.json'),
                             root / 'research_artifacts/benchmarks/x.json')

    def test_flat_layout_maps_manifest_and_data_paths(self):
        from artifact_paths import artifact_root, package_path
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'benchmarks').mkdir()
            self.assertEqual(artifact_root(root), root)
            self.assertEqual(package_path(root, 'research_artifacts/benchmarks/x.json'), root / 'benchmarks/x.json')
            self.assertEqual(package_path(root, 'src/x.py'), root / 'src/x.py')

    def test_ambiguous_and_missing_layout_rejected(self):
        from artifact_paths import artifact_root
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with self.assertRaises(FileNotFoundError):
                artifact_root(root)
            (root / 'benchmarks').mkdir()
            (root / 'research_artifacts/benchmarks').mkdir(parents=True)
            with self.assertRaises(ValueError):
                artifact_root(root)


if __name__ == '__main__':
    unittest.main(verbosity=2)
