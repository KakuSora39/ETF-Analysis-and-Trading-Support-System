import unittest
from pathlib import Path

from etf_universe.__main__ import _parser


class ReadmeCliTests(unittest.TestCase):
    def test_documented_universe_flags_exist_in_parser(self):
        root = Path(__file__).resolve().parents[1]
        documentation = (root / "README.md").read_text(encoding="utf-8")
        help_text = _parser().format_help()
        for flag in (
            "--config", "--output", "--screen-only", "--offline",
            "--write-holdings-template", "--include-simulation-holdings",
        ):
            self.assertIn(flag, documentation)
            self.assertIn(flag, help_text)

    def test_root_readme_documents_source_and_license(self):
        root = Path(__file__).resolve().parents[1]
        readme = (root / "README.md").read_text(encoding="utf-8")
        self.assertIn("zhuleimed/etf-daily-sync-and-backtest", readme)
        self.assertIn("MIT License", readme)
        self.assertTrue((root / "LICENSE").exists())

    def test_local_links_resolve(self):
        import re
        root = Path(__file__).resolve().parents[1]
        for source in (root / "README.md", root / "etf_universe" / "README.md"):
            content = source.read_text(encoding="utf-8")
            for link in re.findall(r"\]\(([^)]+)\)", content):
                if not link.startswith(("https://", "http://", "#")):
                    self.assertTrue((source.parent / link).exists(), f"{source}: {link}")


if __name__ == "__main__":
    unittest.main()
