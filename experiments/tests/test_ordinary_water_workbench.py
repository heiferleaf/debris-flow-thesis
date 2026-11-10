"""Hot-read gallery and independent consent boundary regression checks, offline."""
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib import request

from experiments.common.video.synthetic_wan import atomic_json
from experiments.tools.synthetic_video_workbench_server import DrySceneGallery, OrdinaryWaterGallery, handler_for


class OrdinaryWaterGalleryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.inventory = self.root / 'data/metadata/water/source-inventory.json'
        self.dry_inventory = self.root / 'data/metadata/dry/source-inventory.json'
        self.source = self.root / 'data/source.jpg'
        self.source.parent.mkdir(parents=True)
        self.source.write_bytes(b'original')
        self.dry = self.root / 'data/dry/dry-reference.png'
        self.dry.parent.mkdir()
        self.dry.write_bytes(b'accepted-dry')
        self.output = self.root / 'data/water/ordinary-water-reference.png'
        self.output.parent.mkdir()
        scene = {'index': 1, 'scene_id': 'camera-a', 'camera': 'camera A', 'source_path': 'data/source.jpg',
                 'dry_path': 'data/dry/dry-reference.png', 'output_path': 'data/water/ordinary-water-reference.png',
                 'metadata_path': 'data/water/composition.json'}
        atomic_json(self.inventory, {'items': [scene]})
        atomic_json(self.dry_inventory, {'items': [{**scene, 'output_path': 'data/dry/dry-reference.png',
                     'metadata_path': 'data/dry/composition.json'}]})
        atomic_json(self.dry_inventory.parent / 'user-acceptance.json', {'items': [
            {'path': 'data/dry/dry-reference.png', 'status': 'accepted_by_user'}]})
        self.gallery = OrdinaryWaterGallery(self.inventory, root=self.root)

    def test_new_files_appear_without_restart_and_dry_acceptance_does_not_transfer(self):
        before = self.gallery.snapshot()
        self.assertEqual(before['summary']['expected'], 1)
        self.assertEqual(before['summary']['pending'], 1)
        self.assertEqual(before['summary']['accepted_by_user'], 0)
        self.assertIsNone(before['scenes'][0]['first_frame_url'])
        self.output.write_bytes(b'new-water')
        after = self.gallery.snapshot()
        self.assertEqual(after['summary']['generated'], 1)
        self.assertEqual(after['summary']['pending'], 0)
        self.assertEqual(after['summary']['accepted_by_user'], 0)
        self.assertTrue(after['scenes'][0]['first_frame_url'].startswith('/media/ordinary-water/'))
        self.assertFalse(after['training_ready'])
        atomic_json(self.inventory.parent / 'user-acceptance.json', {'items': [
            {'path': 'data/water/ordinary-water-reference.png', 'status': 'accepted_by_user'}]})
        self.assertEqual(self.gallery.snapshot()['summary']['accepted_by_user'], 1)

    def test_registered_water_media_and_nonnull_dry_firstframe_are_exposed(self):
        self.output.write_bytes(b'new-water')
        dry = DrySceneGallery(self.dry_inventory, root=self.root)
        server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(dry, water=self.gallery))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = 'http://127.0.0.1:%d' % server.server_port
            with request.urlopen(base + '/api/workbench') as response:
                snapshot = json.load(response)
            self.assertEqual(snapshot['ordinary_water']['summary']['generated'], 1)
            self.assertEqual(snapshot['summary']['accepted_by_user'], 1)
            self.assertTrue(snapshot['scenes'][0]['first_frame_url'])
            self.assertEqual(snapshot['scenes'][0]['first_frame_acceptance'], {})
            for kind, expected in (('ordinary-water', b'new-water'), ('ordinary-water-source', b'original'),
                                   ('ordinary-water-dry', b'accepted-dry')):
                with request.urlopen(base + '/media/' + kind + '/camera-a') as response:
                    self.assertEqual(response.read(), expected)
            self.assertIsNone(self.gallery.media('ordinary-water', 'unregistered'))
            self.assertIsNone(self.gallery.media('secret', 'camera-a'))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == '__main__':
    unittest.main()
