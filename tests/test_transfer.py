"""Contract tests on synthetic fixtures; never a DFDC performance benchmark."""
import csv
import gc
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from evaluation.dfdc import binary_metrics
from evaluation.experiments import aggregate_frames, select_champion
from explainability.gradcam import GradCAM
from models.transfer import CLASS_TO_IDX, FrequencyDescriptor, TransferClassifier, load_transfer_checkpoint
from tools.acquire_dfdc_cli import parse_listing
from tools.dfdc_storage import ROOT
from training.dfdc_cache import FrameDataset, image_transform, load_cache, load_manifest, prepare_cache
from training.dfdc_data import prepare
from training.dfdc_data import source_groups
from training.experiment import read_config, train

torch.set_num_threads(2)


def make_fixture(base):
    video_dir=base/'videos';video_dir.mkdir();metadata={}
    for group in range(6):
        for kind in ['real','fake']:
            name=f'{kind}_{group}.mp4'
            writer=cv2.VideoWriter(str(video_dir/name),cv2.VideoWriter_fourcc(*'mp4v'),4,(64,64))
            if not writer.isOpened(): raise RuntimeError('Fixture encoder unavailable')
            for frame in range(4):
                array=np.random.default_rng(group*100+frame+(50 if kind=='fake' else 0)).integers(0,255,(64,64,3),dtype=np.uint8)
                writer.write(array)
            writer.release()
            metadata[name]={'label':kind.upper(),'split':'train'}
            if kind=='fake':metadata[name]['original']=f'real_{group}.mp4'
    path=video_dir/'metadata.json';path.write_text(json.dumps(metadata),encoding='utf-8')
    prepare(path,base/'prepared',42)
    prepare_cache(base/'prepared/manifest.csv',base/'cache',frames=1,faces='full_frame',align=False,require_official=False)
    return base/'cache'


class ModelTests(unittest.TestCase):
    def test_shapes_frozen_parameters_and_bn(self):
        for name in ['resnet18','efficientnet_b0','mobilenet_v3_large']:
            with self.subTest(backbone=name):
                model=TransferClassifier(name,pretrained=False);model.train()
                self.assertFalse(any(p.requires_grad for p in model.features.parameters()))
                self.assertTrue(all(not m.training for m in model.features.modules() if isinstance(m,torch.nn.BatchNorm2d)))
                logits=model(torch.randn(2,3,64,64));self.assertEqual(tuple(logits.shape),(2,2))
                logits.sum().backward();self.assertIsNotNone(model.head[-1].weight.grad)
                model.set_trainable(1);model.train();self.assertTrue(any(p.requires_grad for p in model.features[-1].parameters()))
                self.assertTrue(all(not m.training for m in model.features.modules() if isinstance(m,torch.nn.BatchNorm2d)))
                del model;gc.collect()

    def test_attention_and_frequency_backward(self):
        for attention,fusion in [('eca','none'),('cbam','none'),('none','frequency')]:
            model=TransferClassifier(attention=attention,fusion=fusion,pretrained=False)
            model(torch.randn(2,3,64,64)).square().mean().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            if attention!='none':self.assertTrue(any(p.grad is not None for p in model.attention.parameters()))
        descriptor=FrequencyDescriptor();self.assertEqual(tuple(descriptor(torch.zeros(2,3,64,64)).shape),(2,16))

    def test_frozen_cam_and_cam_plus(self):
        model=TransferClassifier(pretrained=False).eval();tensor=torch.randn(1,3,64,64)
        generator=GradCAM(model,model.cam_layer)
        try:
            for method in ['gradcam','gradcam++']:
                cam=generator.generate(tensor,0,method)
                self.assertEqual(cam.shape,(64,64));self.assertTrue(np.isfinite(cam).all());self.assertGreaterEqual(cam.min(),0);self.assertLessEqual(cam.max(),1)
        finally:generator.close()

    def test_transform_is_deterministic(self):
        image=Image.fromarray(np.random.default_rng(4).integers(0,255,(140,110,3),dtype=np.uint8))
        transform=image_transform('none');self.assertTrue(torch.equal(transform(image),transform(image)))
        for mode in ['basic','jpeg','noise']:
            value=image_transform(mode)(image);self.assertEqual(tuple(value.shape),(3,224,224));self.assertTrue(torch.isfinite(value).all())


class ContractTests(unittest.TestCase):
    def test_fake_index_aggregation_and_metrics(self):
        self.assertEqual(CLASS_TO_IDX,{'fake':0,'real':1})
        rows=[{'filename':'a','source_group':'g','label':'FAKE','frame_index':i,'fake_probability':p} for i,p in enumerate([.8,.4])]
        self.assertAlmostEqual(aggregate_frames(rows)[0]['fake_probability'],.6)
        with self.assertRaises(ValueError):aggregate_frames(rows+[rows[0]])
        m=binary_metrics([0,0,1,1],[.1,.9,.8,.7])
        self.assertEqual(m['confusion_matrix_real_fake'],[[1,1],[0,2]]);self.assertEqual(m['accuracy'],.75);self.assertAlmostEqual(m['f1'],.8)

    def test_cli_listing(self):
        rows,token=parse_listing('Next Page Token = abc\nname,size,creationDate\n\ntrain_sample_videos/a.mp4,123,date\n')
        self.assertEqual(rows,[{'name':'train_sample_videos/a.mp4','bytes':123}]);self.assertEqual(token,'abc')

    def test_optional_identity_groups(self):
        groups=source_groups({'a.mp4':{'label':'REAL','actor_id':'same'},'b.mp4':{'label':'REAL','actor_id':'same'},'c.mp4':{'label':'REAL','actor_id':'different'}})
        self.assertEqual(len(groups),2)

    def test_invalid_metric_inputs(self):
        for y,p in [([.5,1],[.2,.8]),([0,1],[[.2],[.8]]),([0,1],[.1,float('nan')])]:
            with self.assertRaises(ValueError):binary_metrics(y,p)
        with self.assertRaises(ValueError):binary_metrics([0,1],[.1,.9],threshold=2)

    def test_configs_are_small_and_explicit(self):
        configs=[read_config(p) for p in (ROOT/'configs').glob('*.json')]
        self.assertTrue(configs);self.assertEqual(len({c['experiment_id'] for c in configs}),len(configs))
        for c in configs:self.assertLessEqual(c['head_epochs']+c['fine_tune_epochs'],6)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parent=ROOT/'results/smoke';parent.mkdir(parents=True,exist_ok=True)
        cls.tmp=tempfile.TemporaryDirectory(prefix='contracts_',dir=parent);cls.base=Path(cls.tmp.name).resolve();cls.cache=make_fixture(cls.base)

    @classmethod
    def tearDownClass(cls):
        assert cls.base.is_relative_to((ROOT/'results/smoke').resolve());cls.tmp.cleanup()

    def test_cache_and_manifest_integrity(self):
        rows,report=load_cache(self.cache,False);self.assertEqual(len(rows),12)
        dataset=FrameDataset(self.cache,rows,'validation');self.assertGreater(len(dataset),0)
        with self.assertRaises(ValueError):load_cache(self.cache,True)
        manifest=self.base/'prepared/manifest.csv';original=manifest.read_bytes()
        manifest.write_bytes(original+b'\n')
        try:
            with self.assertRaises(ValueError):load_manifest(manifest,False)
        finally:manifest.write_bytes(original)

    def test_synthetic_complete_vertical_slice(self):
        config={'experiment_id':'synthetic_'+self.base.name,'cache':str(self.cache),'model':{'backbone':'resnet18'},'head_epochs':1,'batch_size':2,'accumulation_steps':2,'device':'cuda' if torch.cuda.is_available() else 'cpu','augmentation':'none','max_minutes':5}
        path=self.base/'config.json';path.write_text(json.dumps(config),encoding='utf-8')
        report=train(path,synthetic_test=True)
        self.assertTrue(report['completed']);self.assertEqual(report['dataset_provenance'],'synthetic_fixture')
        model,saved=load_transfer_checkpoint(report['checkpoint']);self.assertEqual(saved['class_to_idx'],CLASS_TO_IDX)
        self.assertEqual(tuple(model(torch.randn(1,3,64,64)).shape),(1,2))
        output=Path(report['checkpoint']).parent
        self.assertTrue((output/'validation_video_predictions.csv').exists())
        with self.assertRaises(ValueError):select_champion(output.parent,self.base/'champion.json')
        # Delete only this test-created directory within the smoke output root.
        import shutil
        assert output.is_relative_to((ROOT/'results/smoke').resolve());shutil.rmtree(output)

    def test_final_test_cannot_silently_rerun(self):
        from evaluation.experiments import final_test
        champion=self.base/'test_champion.json'
        champion.write_text(json.dumps({'experiment_id':'synthetic_gate_test','checkpoint':str(self.base/'missing.pt'),'checkpoint_sha256':'missing'}),encoding='utf-8')
        with patch('evaluation.experiments.ROOT',self.base):
            with self.assertRaises(FileNotFoundError):final_test(champion,'cpu')
            with self.assertRaises(FileExistsError):final_test(champion,'cpu')

    def test_resume_matches_uninterrupted_weights(self):
        import shutil
        import training.experiment as engine
        common={'cache':str(self.cache),'model':{'backbone':'resnet18'},'head_epochs':2,'batch_size':2,'accumulation_steps':2,
                'device':'cuda' if torch.cuda.is_available() else 'cpu','augmentation':'none','max_minutes':5,'patience':3}
        paths=[]
        try:
            for name in ['uninterrupted','resumed']:
                config={**common,'experiment_id':name+'_'+self.base.name};path=self.base/(name+'.json');path.write_text(json.dumps(config),encoding='utf-8')
                directory=ROOT/'results/smoke'/config['experiment_id'];paths.append(directory)
                if name=='resumed':
                    original=engine.save_torch
                    def interrupted(file,payload):
                        original(file,payload)
                        if Path(file).name=='best.pt':raise RuntimeError('Synthetic power-interruption fixture')
                    with patch('training.experiment.save_torch',side_effect=interrupted):
                        with self.assertRaisesRegex(RuntimeError,'power-interruption'):train(path,synthetic_test=True)
                    train(path,resume=True,synthetic_test=True)
                else:train(path,synthetic_test=True)
            left=torch.load(paths[0]/'last.pt',weights_only=True,map_location='cpu')['model_state_dict']
            right=torch.load(paths[1]/'last.pt',weights_only=True,map_location='cpu')['model_state_dict']
            self.assertEqual(set(left),set(right))
            for key in left:self.assertTrue(torch.equal(left[key],right[key]),key)
        finally:
            for path in paths:
                path=path.resolve();assert path.is_relative_to((ROOT/'results/smoke').resolve())
                if path.exists():shutil.rmtree(path)
