import unittest
import numpy as np
import torch
from torch import nn
from evaluation.dfdc import binary_metrics
from explainability.methods import integrated_gradients
from models.unified import FeatureHead
from training.video_protocol import normalized_indices
from training.unified_experiment import calibrate,ece,probability,pixels


class UnifiedProtocolTests(unittest.TestCase):
    def test_normalized_samples_have_no_padding_shortcut(self):
        for count in [8,9,30,300,30000]:
            indices=normalized_indices(count)
            self.assertEqual(len(indices),8);self.assertEqual(len(set(indices)),8)
            self.assertTrue(all(0<=i<count for i in indices))
            self.assertTrue(np.max(np.abs((np.asarray(indices)+.5)/count-(np.arange(8)+.5)/8))<=1/count)
        with self.assertRaises(ValueError):normalized_indices(7)

    def test_all_fake_high_accuracy_does_not_hide_real_recall(self):
        y=[0]*74+[1]*1088;m=binary_metrics(y,np.ones(len(y)))
        self.assertAlmostEqual(m['accuracy'],1088/1162)
        self.assertEqual(m['real_recall'],0);self.assertEqual(m['fake_recall'],1)
        self.assertEqual(m['balanced_accuracy'],.5);self.assertLess(m['macro_f1'],.5)

    def test_temperature_preserves_labels_and_is_validation_only(self):
        logits=torch.tensor([[0.,4.],[1.,0.],[0.,3.],[3.,0.]])
        result=calibrate([1,1,0,0],logits)
        self.assertEqual(result['fit_split'],'validation')
        self.assertFalse(result['inconclusive_enabled'])
        np.testing.assert_array_equal(probability(logits)>=.5,probability(logits,result['temperature'])>=.5)
        self.assertLessEqual(result['after']['log_loss'],result['before']['log_loss']+1e-5)
        self.assertEqual(ece([0,1],[0,1]),0)

    def test_spatial_head_uses_probability_mean(self):
        torch.manual_seed(1);head=FeatureHead(3);features=torch.randn(2,8,3)
        expected=head.frame_logits(features).softmax(-1).mean(1)
        torch.testing.assert_close(head(features).softmax(-1),expected)
        torch.testing.assert_close(head(features.flip(1)),head(features))

    def test_augmentation_preserves_order_count_and_shared_transform(self):
        image=np.full((8,224,224,3),128,np.uint8)
        x=pixels(image,True,42);self.assertEqual(tuple(x.shape),(8,3,224,224))
        torch.testing.assert_close(x[0],x[-1]);torch.testing.assert_close(x,pixels(image,True,42))

    def test_integrated_gradients_completeness_for_linear_model(self):
        model=nn.Sequential(nn.Flatten(),nn.Linear(12,2,bias=False))
        tensor=torch.arange(12,dtype=torch.float32).reshape(1,3,2,2)/12
        cam,details=integrated_gradients(model,tensor,1,8)
        self.assertTrue(np.isfinite(cam).all());self.assertEqual(cam.shape,(2,2))
        self.assertLess(details['completeness_absolute_residual'],1e-5)

    def test_combined_system_routes_only_cnn_features_to_lstm(self):
        from inference.unified import ResearchPredictor
        p=ResearchPredictor.__new__(ResearchPredictor);p.device=torch.device('cpu');p.id='S8';p.fusion_weight=.25
        p.spatial_head=FeatureHead(7);p.temporal_head=FeatureHead(3,'temporal')
        p.spatial_saved={'feature_mean':torch.zeros(7),'feature_std':torch.ones(7)}
        p.temporal_saved={'feature_mean':torch.zeros(3),'feature_std':torch.ones(3),'dimension':3}
        x=torch.randn(2,8,7)
        expected=.25*p.spatial_head(x).softmax(-1)+.75*p.temporal_head(x[...,:3]).softmax(-1)
        torch.testing.assert_close(p.decision(x).softmax(-1),expected)
        torch.testing.assert_close(p.decision(x,image=True),p.spatial_head(x))

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA RNN regression')
    def test_eval_lstm_cuda_explanations_backpropagate_without_changing_scores(self):
        from inference.unified import ResearchPredictor
        torch.manual_seed(42)
        p=ResearchPredictor.__new__(ResearchPredictor);p.device=torch.device('cuda');p.id='S6';p.fusion_weight=None
        p.spatial_head=FeatureHead(3).cuda().eval();p.temporal_head=FeatureHead(3,'temporal').cuda().eval()
        p.spatial_saved={'feature_mean':torch.zeros(3),'feature_std':torch.ones(3)}
        p.temporal_saved={'feature_mean':torch.zeros(3),'feature_std':torch.ones(3),'dimension':3}
        x=torch.randn(1,8,3,device='cuda',requires_grad=True)
        with torch.no_grad():reference=p.decision(x.detach())
        explained=p.decision(x);gradient=torch.autograd.grad(explained[0,1],x)[0]
        self.assertTrue(torch.isfinite(gradient).all());self.assertGreater(float(gradient.abs().sum()),0)
        # Native and cuDNN kernels have different float32/TF32 accumulation.
        # Check practical probability/decision parity, not bitwise logit parity.
        torch.testing.assert_close(explained.detach().softmax(-1),reference.softmax(-1),atol=1e-4,rtol=1e-4)
        self.assertEqual(int(explained.argmax(-1)),int(reference.argmax(-1)))


if __name__=='__main__':unittest.main()
