import unittest

from phd_rag.ingest import load_registry
from phd_rag.settings import get_settings


class RegistryTests(unittest.TestCase):
    def test_registry_has_194_sources_across_twenty_countries(self) -> None:
        sources = load_registry(get_settings().registry_path)
        countries = {source.country for source in sources}
        self.assertEqual(
            countries,
            {
                "Australia",
                "Austria",
                "Belgium",
                "Canada",
                "China",
                "Finland",
                "France",
                "Germany",
                "India",
                "Italy",
                "Japan",
                "Malaysia",
                "Netherlands",
                "Norway",
                "South Korea",
                "Spain",
                "Sweden",
                "Switzerland",
                "United Kingdom",
                "United States",
            },
        )
        self.assertEqual(len(sources), 194)
        identities = {(source.country, source.university) for source in sources}
        self.assertEqual(len(identities), len(sources))
        self.assertTrue(all("http" in str(source.url) for source in sources))
