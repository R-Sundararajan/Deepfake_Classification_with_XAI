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


if __name__=='__main__':
    unittest.main()
