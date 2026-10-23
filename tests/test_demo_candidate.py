import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import inference.demo_candidate as candidate


class CandidatePointerTests(unittest.TestCase):
    def fixture(self, root, verified=True):
        (root/'configs/runtime').mkdir(parents=True)
        (root/'results').mkdir()
        checkpoint=root/'results/candidate.pth'
        checkpoint.write_bytes(b'isolated test checkpoint')
        digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        record={'id':'test','checkpoint':str(checkpoint),'checkpoint_sha256':digest,
                'source':'image_continuation','evaluation_unit':'image',
                'full_validation':{'checkpoint_sha256':digest,'completed':True},
                'held_out_test':{'checkpoint_sha256':digest,'completed':True}}
        deployment=root/'results/deployment.json'
        deployment.write_text(json.dumps({'verified':verified,'gates':{'reload':True,'evaluation':True},'record':record}),encoding='utf-8')
        pointer={'format':'demo_image_pointer_v1','summary':str(deployment),
                 'summary_sha256':hashlib.sha256(deployment.read_bytes()).hexdigest()}
        (root/'configs/runtime/active_image_model.json').write_text(json.dumps(pointer),encoding='utf-8')
        return checkpoint,record

    def test_unverified_or_absent_pointer_keeps_baseline(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve()
            with patch.object(candidate,'PROJECT_ROOT',root):
                self.assertIsNone(candidate.active_image_candidate())
                self.fixture(root,verified=False)
                self.assertIsNone(candidate.active_image_candidate())

    def test_verified_candidate_is_rejected_after_checkpoint_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve()
            checkpoint,record=self.fixture(root)
            with patch.object(candidate,'PROJECT_ROOT',root):
                self.assertEqual(candidate.active_image_candidate(),record)
                checkpoint.write_bytes(b'changed checkpoint contents')
                self.assertIsNone(candidate.active_image_candidate())

    def current_fixture(self, root):
        checkpoint, old = self.fixture(root)
        records = []
        for identity in ['R18_FT_20261021', 'E02_efficientnet_b0']:
            record = dict(old, id=identity)
            record['validation'] = {'completed': True, 'checkpoint_sha256': old['checkpoint_sha256']}
            record['full_validation'] = dict(record['validation'], samples=20000)
            record['held_out_test'] = dict(record['validation'], samples=20000)
            records.append(record)
        deployment = root/'results/current.json'
        deployment.write_text(json.dumps({'verified': True, 'gates': {'reload': True}, 'records': records}), encoding='utf-8')
        pointer = root/'configs/runtime/current_image_models.json'
        pointer.write_text(json.dumps({'format': 'current_image_models_pointer_v1', 'summary': str(deployment),
                                      'summary_sha256': hashlib.sha256(deployment.read_bytes()).hexdigest()}), encoding='utf-8')
        return records, deployment, pointer

    def test_current_aliases_require_both_unique_ids_and_bound_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            records, deployment, pointer = self.current_fixture(root)
            with patch.object(candidate, 'PROJECT_ROOT', root):
                self.assertEqual(candidate.current_image_candidates(), records)
                value = json.loads(deployment.read_text())
                value['records'][1]['id'] = 'R18_FT_20261021'
                deployment.write_text(json.dumps(value))
                activation = json.loads(pointer.read_text())
                activation['summary_sha256'] = hashlib.sha256(deployment.read_bytes()).hexdigest()
                pointer.write_text(json.dumps(activation))
                # Bad multi-model updates leave the last working single pointer.
                self.assertEqual(len(candidate.current_image_candidates()), 1)
                self.assertEqual(candidate.current_image_candidates()[0]['id'], 'test')

    def test_current_summary_tampering_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            records, deployment, pointer = self.current_fixture(root)
            deployment.write_text(deployment.read_text()+' ')
            with patch.object(candidate, 'PROJECT_ROOT', root):
                self.assertEqual(len(candidate.current_image_candidates()), 1)


if __name__=='__main__':
    unittest.main()
