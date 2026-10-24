"""Independent known-feature and saturation checks from the end-to-end audit."""
import unittest
import json
import numpy as np
import torch
from torch import nn
from explainability.methods import occlusion_sensitivity, integrated_gradients
from explainability.gradcam import create_visualizations
from inference.unified import ResearchPredictor
from unittest.mock import patch
from PIL import Image


class KnownPatch(nn.Module):
    def __init__(self, offset=0.):
        super().__init__(); self.offset=offset
    def forward(self,x):
        score=self.offset+x[:,0,:8,:8].mean((1,2))
        return torch.stack([score,torch.zeros_like(score)],-1)


class AuditTests(unittest.TestCase):
    def test_occlusion_coordinates_target_sign_and_raw_effect(self):
        x=torch.ones(1,3,16,16)
        model=KnownPatch().eval()
        for target, sign in [(0,1),(1,-1)]:
            cam,detail=occlusion_sensitivity(model,x,target,patch=8)
            self.assertEqual(cam.shape,(16,16))
            np.testing.assert_array_equal(cam[:8,:8],np.full((8,8),sign))
            self.assertEqual(np.count_nonzero(cam),64)
            self.assertEqual(detail['patch_bounds_y1_y2_x1_x2'][0],[0,8,0,8])
            expected=float(model(x).softmax(-1)[0,target])-.5
            self.assertAlmostEqual(detail['raw_attribution_statistics']['signed_sum'],64*expected,places=5)
            self.assertEqual(detail['score_space'],'selected class probability')

    def test_saturated_probability_keeps_margin_diagnostic_separate(self):
        x=torch.ones(1,3,16,16)
        cam,detail=occlusion_sensitivity(KnownPatch(40),x,0,patch=8)
        np.testing.assert_array_equal(cam,np.zeros((16,16)))
        self.assertEqual(detail['max_absolute_probability_change'],0)
        self.assertTrue(detail['probability_resolution_limited'])
        self.assertEqual(detail['target_vs_best_alternative_margin_drops'],[1.,0.,0.,0.])
        json.dumps(detail,allow_nan=False)

    def test_ig_channel_cancellation_is_valid_and_complete(self):
        class OpposingChannels(nn.Module):
            def forward(self,x):
                s=(x[:,0]-x[:,1]).sum((1,2))
                return torch.stack([s,-s],-1)
        cam,detail=integrated_gradients(OpposingChannels(),torch.ones(1,3,8,8),0,8)
        np.testing.assert_allclose(cam,0,atol=1e-6)
        self.assertGreater(detail['raw_attribution_statistics']['abs_sum'],127)
        self.assertLess(detail['completeness_absolute_residual'],1e-5)
        image=np.full((8,8,3),127,np.uint8)
        _,overlay=create_visualizations(image,cam,signed=True)
        np.testing.assert_array_equal(overlay,image)

    def test_research_probability_metadata_is_not_overwritten(self):
        p=ResearchPredictor.__new__(ResearchPredictor)
        p.methods=['occlusion']; p.names=[]; p.encoders=[]; p.layer=None
        p.id='S6'; p.image_spatial_id='S2'; p.checkpoint_sha256='test'
        p.tensor=lambda array:torch.ones(1,3,16,16)
        p.attribution_model=lambda *args:KnownPatch()
        with patch('inference.unified.explain_tensor',return_value=(np.zeros((16,16),np.float32),{'score_space':'selected class probability'})):
            result=p.explain(Image.new('RGB',(16,16)),0,'occlusion')
        self.assertEqual(result['details']['score_space'],'selected class probability')


if __name__=='__main__': unittest.main()
