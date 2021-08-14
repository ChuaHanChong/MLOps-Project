"""Test for loading ml module."""
import unittest
import inspect

from pathlib import Path


class TestImportModule(unittest.TestCase):
    """Module testing."""

    def test_import_ml(self):
        """Test ml module."""
        import ml
        self.assertEqual(
            inspect.getfile(ml),
            str(Path.cwd().joinpath('src', 'ml', '__init__.py'))
        )


if __name__ == '__main__':
    unittest.main()