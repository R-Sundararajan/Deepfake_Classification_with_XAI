"""Offline contract tests; generated media is synthetic and never a DFDC result."""
import csv
import io
import json
import tempfile
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from evaluation.dfdc import binary_metrics,mean_fake_probability
from tools.acquire_dfdc import PREFIX,SafeRedirect,download,permitted_url,sha256
from tools.dfdc_storage import ROOT,LIMIT,HEADROOM,check_budget,owned_path
from training.dfdc_data import FaceCropper,iter_frames,load_metadata,prepare,probe_video,sample_indices,source_groups,split_groups


class MetadataTests(unittest.TestCase):
    def setUp(self):
        base=ROOT/'results/.work';base.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(prefix='dfdc_test_',dir=base)
        self.path=Path(self.tmp.name).resolve()
        assert self.path.is_relative_to(base.resolve())

    def tearDown(self):
        # Delete only this test-created, resolved directory under the workspace.
        assert self.path.is_relative_to((ROOT/'results/.work').resolve())
        self.tmp.cleanup()

    def write_metadata(self,data):
        p=self.path/'metadata.json';p.write_text(json.dumps(data),encoding='utf-8');return p

    def data(self,n=12):
        d={}
        for i in range(n):
            d[f'r{i}.mp4']={'label':'REAL','split':'train'}
            for j in range(3):d[f'f{i}_{j}.mp4']={'label':'FAKE','original':f'r{i}.mp4','split':'train'}
        return d

    def test_metadata_labels(self):
        data=self.data();self.assertEqual(load_metadata(self.write_metadata(data)),data)

    def test_unsafe_paths_rejected(self):
        for name in ['../bad.mp4','C:bad.mp4','sub/bad.mp4','sub\\bad.mp4','..mp4']:
            with self.subTest(name=name),self.assertRaises(ValueError):load_metadata(self.write_metadata({name:{'label':'REAL'}}))

    def test_unknown_labels_and_missing_original_rejected(self):
        for data in [{'a.mp4':{'label':'UNKNOWN'}},{'a.mp4':{'label':'FAKE'}}]:
            with self.assertRaises(ValueError):load_metadata(self.write_metadata(data))

    def test_duplicate_keys_rejected(self):
        p=self.path/'metadata.json';p.write_text('{"a.mp4":{"label":"REAL"},"a.mp4":{"label":"REAL"}}')
        with self.assertRaises(ValueError):load_metadata(p)

    def test_case_collision_rejected(self):
        with self.assertRaises(ValueError):load_metadata(self.write_metadata({'a.mp4':{'label':'REAL'},'A.mp4':{'label':'REAL'}}))

    def test_missing_external_source_still_groups_siblings(self):
        d={'a.mp4':{'label':'FAKE','original':'missing.mp4'},'b.mp4':{'label':'FAKE','original':'missing.mp4'}}
        self.assertEqual(len(source_groups(d)),1)

    def test_transitive_groups(self):
        d={'r.mp4':{'label':'REAL'},'a.mp4':{'label':'FAKE','original':'r.mp4'},'b.mp4':{'label':'FAKE','original':'a.mp4'}}
        self.assertEqual(len(source_groups(d)),1)

    def test_original_aliases_are_case_insensitive(self):
        d={'Real.mp4':{'label':'REAL'},'fake.mp4':{'label':'FAKE','original':'REAL.mp4'}}
        self.assertEqual(len(source_groups(d)),1)

    def test_split_determinism_and_no_source_leakage(self):
        d=self.data();one,groups,warnings=split_groups(d,42);two,_,_=split_groups(dict(reversed(list(d.items()))),42)
        self.assertEqual(one,two)
        for members in groups.values():self.assertEqual(len({one[n][1] for n in members}),1)
        self.assertEqual(set(s for _,s in one.values()),{'train','validation','test'})
        for split in ['train','validation','test']:self.assertEqual({d[n]['label'] for n in d if one[n][1]==split},{'REAL','FAKE'})
        self.assertTrue(any('identity' in w for w in warnings))

    def test_tiny_group_count_fails(self):
        with self.assertRaises(ValueError):split_groups(self.data(2))

    def test_manifest_reports_missing_videos_without_fabricated_provenance(self):
        report=prepare(self.write_metadata(self.data(4)),self.path/'prepared')
        self.assertEqual(len(report['invalid_videos']),16)
        self.assertFalse(report['official_provenance_verified'])
        with (self.path/'prepared/manifest.csv').open(newline='') as f:rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),16)
        with self.assertRaises(FileExistsError):prepare(self.path/'metadata.json',self.path/'prepared')

    def video(self):
        p=self.path/'tiny.mp4';w=cv2.VideoWriter(str(p),cv2.VideoWriter_fourcc(*'mp4v'),10,(64,64))
        self.assertTrue(w.isOpened())
        for i in range(30):w.write(np.full((64,64,3),i*7,dtype=np.uint8))
        w.release();return p

    def test_video_probe_and_stream_sampling(self):
        p=self.video();probe=probe_video(p);self.assertEqual(probe['frame_count'],30)
        frames=list(iter_frames(p,interval=1,max_frames=2));self.assertEqual([f[0] for f in frames],[0,20])
        self.assertEqual(frames[0][2].shape,(64,64,3))
        self.assertLess(float(frames[0][2].mean()),float(frames[1][2].mean()))

    def test_corrupt_video_rejected(self):
        p=self.path/'bad.mp4';p.write_bytes(b'not an mp4')
        with self.assertRaises(ValueError):probe_video(p)
        with self.assertRaises(ValueError):list(iter_frames(p))

    def test_no_face_fallback_is_explicit(self):
        frame=np.zeros((64,64,3),dtype=np.uint8)
        if hasattr(cv2,'CascadeClassifier'):
            crop,method=FaceCropper('haar')(frame)
            self.assertEqual(method,'no_face_full_frame_fallback');self.assertEqual(crop.shape,frame.shape)
        else:
            with self.assertRaisesRegex(RuntimeError,'lacks Haar'):FaceCropper('haar')
        self.assertEqual(FaceCropper('full_frame')(frame)[1],'full_frame')

    def test_acquisition_reuses_only_verified_originals(self):
        p=self.path/'existing.mp4';p.write_bytes(b'original')
        item={'name':PREFIX+p.name,'bytes':p.stat().st_size}
        known={item['name']:{**item,'sha256':sha256(p)}}
        with patch('tools.acquire_dfdc.open_official') as network:
            self.assertEqual(download(item,self.path,known),known[item['name']]);network.assert_not_called()
        with self.assertRaises(FileExistsError):download(item,self.path,{})
        self.assertEqual(p.read_bytes(),b'original')

    def test_acquisition_rejects_remote_size_overrun(self):
        class Response(io.BytesIO):headers={}
        item={'name':PREFIX+'new.mp4','bytes':3}
        with patch('tools.acquire_dfdc.open_official',return_value=Response(b'toolong')):
            with self.assertRaisesRegex(RuntimeError,'exceeded'):download(item,self.path,{})
        self.assertFalse((self.path/'new.mp4').exists())
        self.assertTrue((self.path/'new.mp4.part').exists())

    def test_partial_download_is_preserved(self):
        p=self.path/'new.mp4.part';p.write_bytes(b'partial')
        with patch('tools.acquire_dfdc.open_official') as network:
            with self.assertRaises(FileExistsError):download({'name':PREFIX+'new.mp4','bytes':10},self.path,{})
            network.assert_not_called()
        self.assertEqual(p.read_bytes(),b'partial')


class MathAndSafetyTests(unittest.TestCase):
    def test_long_video_sampling_bounded(self):
        self.assertEqual(len(sample_indices(10**9,30,1,16)),16)
        self.assertEqual(sample_indices(3,30,1,1),[0])
        for args in [(0,30,1,16),(30,0,1,16),(30,30,0,16),(30,30,1,0)]:
            with self.assertRaises(ValueError):sample_indices(*args)

    def test_probability_aggregation_not_majority_labels(self):
        self.assertAlmostEqual(mean_fake_probability([.49,.49,.99]),.6566666667)
        for bad in [[],[float('nan')],[-.1],[1.1]]:
            with self.assertRaises(ValueError):mean_fake_probability(bad)

    def test_metrics_fake_positive_and_confusion_order(self):
        m=binary_metrics([0,0,1,1],[.1,.8,.7,.9])
        self.assertEqual(m['accuracy'],.75);self.assertEqual(m['confusion_matrix_real_fake'],[[1,1],[0,2]])
        self.assertAlmostEqual(m['precision'],2/3);self.assertEqual(m['recall'],1)
        self.assertAlmostEqual(m['f1'],.8);self.assertAlmostEqual(m['roc_auc'],.75)

    def test_single_class_auc_not_fabricated(self):
        self.assertIsNone(binary_metrics([1,1],[.1,.9])['roc_auc'])

    def test_budget_exact_boundary_and_headroom(self):
        check_budget(1,used=LIMIT-1,free=HEADROOM+1)
        with self.assertRaises(RuntimeError):check_budget(2,used=LIMIT-1,free=100*LIMIT)
        with self.assertRaises(RuntimeError):check_budget(2,used=0,free=HEADROOM+1)

    def test_outputs_stay_owned_on_d(self):
        self.assertTrue(owned_path(ROOT/'results/test.json').is_relative_to(ROOT))
        for p in [Path('C:/tmp/test.json'),ROOT/'models/new.pth',ROOT/'data/dfdc/../../../outside.json']:
            with self.assertRaises(ValueError):owned_path(p)

    def test_redirect_strips_cross_host_credentials(self):
        request=urllib.request.Request('https://www.kaggle.com/file',headers={'Authorization':'Bearer TEST_ONLY'})
        redirected=SafeRedirect().redirect_request(request,None,302,'',{},'https://storage.googleapis.com/official/file')
        self.assertIsNone(redirected.get_header('Authorization'))
        with self.assertRaises(RuntimeError):SafeRedirect().redirect_request(request,None,302,'',{},'https://unofficial.example/file')
        self.assertFalse(permitted_url('http://www.kaggle.com/file'))


if __name__=='__main__':unittest.main()
