import gzip
import json
from pathlib import Path
import subprocess
import tempfile
import sys
import unittest
from uninavbench.data import download

SCRIPT = str(Path(__file__).resolve().parents[1] / 'scripts' / 'build_release.py')
class ReleaseTests(unittest.TestCase):
    def test_mini_roundtrip_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'raw'; release=Path(tmp)/'release'; dest=Path(tmp)/'data'
            for scene in ['0','4','6']:
                for level in ['static','small','all']:
                    base=root/'test'/scene/'episodes'/level; base.mkdir(parents=True)
                    for task in ['nav','eqa']:
                        ep={'scene_type':'test','scene_id':int(scene),'image_name':['Cup|goal.png'] if task=='nav' else [None]}
                        (base/(task+'.json')).write_text(json.dumps([ep]))
                base=root/'test'/scene/'dynamic_strategy';base.mkdir()
                for level in ['small','all']:(base/('strategy_final_'+level+'.json')).write_text('{}')
                base=root/'attributes'/'test'/scene/'images';base.mkdir(parents=True)
                (base/'Cup_goal.png').write_bytes(b'image fixture')
            command=[sys.executable,SCRIPT,str(root),str(release),'--splits','mini_test','--workers','1']
            subprocess.run(command,check=True,stdout=subprocess.DEVNULL)
            manifest=json.loads((release/'manifest.json').read_text()); shard=manifest['shards'][0]
            self.assertEqual(shard['source_split'],'test');self.assertEqual(shard['scenes'],['0','4','6'])
            download(['mini_test'],data_dir=str(dest),from_dir=str(release))
            self.assertFalse((dest/'mini_test').exists())
            for scene in ['0','4','6']:
                with gzip.open(dest/'test'/scene/'episodes'/'all'/'nav.json.gz','rt') as f:
                    self.assertEqual(json.load(f)[0]['scene_type'],'test')
                self.assertEqual((dest/'attributes'/'test'/scene/'images'/'Cup|goal.png').read_bytes(),b'image fixture')
            archive=release/shard['file']; before=archive.stat().st_mtime_ns
            subprocess.run(command+['--resume'],check=True,stdout=subprocess.DEVNULL)
            self.assertEqual(archive.stat().st_mtime_ns,before)
            with self.assertRaisesRegex(ValueError,'does not contain split'):
                download(['train'],data_dir=str(dest),from_dir=str(release))
            with self.assertRaisesRegex(ValueError,'does not contain scenes'):
                download(['mini_test'],scenes=['99'],data_dir=str(dest),from_dir=str(release))
            archive.write_bytes(b'corrupt')
            with self.assertRaisesRegex(IOError,'sha256 mismatch'):
                download(['mini_test'],data_dir=str(Path(tmp)/'fresh'),from_dir=str(release))

if __name__ == "__main__":
    unittest.main()
