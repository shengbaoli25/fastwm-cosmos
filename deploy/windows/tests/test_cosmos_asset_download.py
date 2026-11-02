import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location('cosmos_asset_download',
    Path(__file__).resolve().parents[1] / 'cosmos_asset_download.py')
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


class LocalImportTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.source = self.root/'download.pt'
        self.source.write_bytes(b'verified-test-payload')
        self.target = self.root/'assets'/'target.pt'
        self.asset = dict(id='test', bytes=self.source.stat().st_size,
            sha256=hashlib.sha256(self.source.read_bytes()).hexdigest())

    def test_import_preserves_source_and_is_idempotent(self):
        first = download.import_verified(self.source, self.target, self.asset, {})
        self.assertTrue(first['copied'])
        self.assertEqual(self.source.read_bytes(), self.target.read_bytes())
        second = download.import_verified(self.source, self.target, self.asset, {})
        self.assertFalse(second['copied'])
        self.assertTrue(second['existing_identical'])
        self.assertEqual(list(self.target.parent.glob('*.part')), [])

    def test_different_destination_is_never_overwritten(self):
        self.target.parent.mkdir()
        self.target.write_bytes(b'existing-user-file')
        with self.assertRaises(ValueError):
            download.import_verified(self.source, self.target, self.asset, {})
        self.assertEqual(self.target.read_bytes(), b'existing-user-file')

    def test_corrupt_source_is_never_published(self):
        self.source.write_bytes(b'x'*self.asset['bytes'])
        with self.assertRaises(ValueError):
            download.import_verified(self.source, self.target, self.asset, {})
        self.assertFalse(self.target.exists())

    def test_matching_browser_temporary_blocks_import(self):
        Path(str(self.source)+'.crswap').write_bytes(b'pending')
        with self.assertRaises(download.Incomplete):
            download.import_verified(self.source, self.target, self.asset, {})
        self.assertFalse(self.target.exists())

    def test_download_url_is_not_treated_as_path(self):
        with self.assertRaisesRegex(ValueError, 'Only local'):
            download.local_path('https://example.invalid/private?token=do-not-fetch')

    def test_full_manifest_identity_is_whitespace_independent(self):
        self.assertEqual(download.canonical_identity({'b': 2, 'a': '文本'}),
                         download.canonical_identity({'a': '文本', 'b': 2}))
        self.assertNotEqual(download.canonical_identity({'a': '文本'}),
                            download.canonical_identity({'a': '文本', 'created': 1}))


if __name__ == '__main__':
    unittest.main()
