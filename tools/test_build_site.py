"""Publication guardrails; no browser, CUDA or third-party packages required."""

from pathlib import Path
import shutil
import tempfile
import unittest

from build_site import SITE, build_site, validate_site


class BuildSiteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="particlesplat-build-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.site = self.root / "source"
        shutil.copytree(SITE, self.site)

    def test_runtime_only_and_provenance_retained(self):
        output = self.root / "public"
        build_site(self.site, output)
        self.assertFalse((output / "tools").exists())
        self.assertFalse((output / "README.md").exists())
        self.assertFalse((output / ".git").exists())
        self.assertTrue((output / ".nojekyll").exists())
        manifest = "static/scenes/close-jar/scene.json"
        self.assertEqual((output / manifest).read_bytes(), (self.site / manifest).read_bytes())
        self.assertIn('"postprocessing"', (output / manifest).read_text())
        validate_site(output)

    def test_rejects_private_assets_hidden_files_and_symlinks(self):
        for name in ["paper.pdf", "paper.zip", "model.pth", "audit.npz", ".env"]:
            with self.subTest(name=name):
                path = self.site / "static" / name
                path.touch()
                with self.assertRaises(ValueError):
                    validate_site(self.site)
                path.unlink()
        (self.site / "static/leak.png").symlink_to(self.site / "index.html")
        with self.assertRaises(ValueError):
            validate_site(self.site)

    def test_never_overwrites_or_nests_output(self):
        for output in [self.site, self.site / "build", self.root]:
            with self.subTest(output=output), self.assertRaises(ValueError):
                build_site(self.site, output)

    def test_detects_corrupt_geometry_and_broken_links(self):
        path = self.site / "static/scenes/close-jar/particle-8.ply"
        original = path.read_bytes()
        path.write_bytes(b"invalid")
        with self.assertRaisesRegex(ValueError, "changed asset"):
            validate_site(self.site)
        path.write_bytes(original)
        page = self.site / "index.html"
        page.write_text(page.read_text() + '<img src="missing.png">')
        with self.assertRaisesRegex(ValueError, "Missing page asset"):
            validate_site(self.site)


if __name__ == "__main__":
    unittest.main()
