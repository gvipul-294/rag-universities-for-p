import unittest

from phd_rag.ingest import load_registry
from phd_rag.settings import get_settings


class RegistryTests(unittest.TestCase):
    def test_registry_has_ten_countries_with_ten_sources_each(self) -> None:
        sources = load_registry(get_settings().registry_path)
        countries = {source.country for source in sources}
        self.assertEqual(len(countries), 10)
        self.assertEqual(len(sources), 100)
        self.assertTrue(all(sum(source.country == country for source in sources) == 10 for country in countries))
        self.assertTrue(all("http" in str(source.url) for source in sources))
