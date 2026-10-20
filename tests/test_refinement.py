import unittest
import numpy as np
import torch
from torch import nn
from PIL import Image
from explainability.gradcam import GradCAM, create_visualizations
from explainability.methods import integrated_gradients, occlusion_sensitivity, lime_explanation
from inference.contracts import class_probabilities
from training.unified_experiment import pixels
from training.dfdc_cache import image_transform


class LocalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 2, 1, bias=False)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[0, 0] = 1
            self.conv.weight[1, 1] = 1
    def forward(self, x):
        return self.conv(x).mean((2, 3))


class RefinementTests(unittest.TestCase):
    def test_logits_mapping_and_predicted_class_score(self):
        new = torch.tensor([[2., -1.], [-1., 2.], [0., 0.]])
        p = class_probabilities(new, {'real': 0, 'fake': 1})
        old = class_probabilities(new[:, [1, 0]], {'fake': 0, 'real': 1})
        torch.testing.assert_close(p['fake'], old['fake'])
        self.assertEqual(p['predicted_fake'].tolist(), [False, True, True])
        torch.testing.assert_close(p['predicted_class_score'], torch.maximum(p['fake'], p['real']))
        self.assertGreater(float(p['real'][0]), .95)
        calibrated = class_probabilities(new, {'real': 0, 'fake': 1}, temperature=3)
        self.assertEqual(p['predicted_fake'].tolist(), calibrated['predicted_fake'].tolist())
        with self.assertRaises(ValueError):
            class_probabilities(torch.zeros(1, 1), {'real': 0, 'fake': 1})

    def test_ig_preserves_sign_and_linear_completeness(self):
        model = nn.Sequential(nn.Flatten(), nn.Linear(12, 2, bias=False))
        with torch.no_grad():
            model[1].weight[0] = torch.tensor([1., -2., 3., -1.]*3)
            model[1].weight[1] = -model[1].weight[0]
        x = torch.ones(1, 3, 2, 2)
        first, details = integrated_gradients(model, x, 0, 8)
        other, _ = integrated_gradients(model, x, 1, 8)
        np.testing.assert_allclose(first, -other, atol=1e-6)
        self.assertGreater(first.max(), 0)
        self.assertLess(first.min(), 0)
        self.assertLess(details['completeness_absolute_residual'], 1e-5)
        self.assertTrue(details['converged'])
        self.assertGreater(details['raw_attribution_statistics']['positive_sum'], 0)

    def test_cam_target_hooks_and_score_preservation(self):
        model = LocalModel().eval()
        x = torch.zeros(1, 3, 16, 16)
        x[:, 0, :8] = 1
        x[:, 1, 8:] = 1
        reference = model(x).detach()
        for method in ['gradcam', 'gradcam++', 'layercam']:
            before = len(model.conv._forward_hooks)
            generator = GradCAM(model, model.conv)
            try:
                first = generator.generate(x, 0, method)
                other = generator.generate(x, 1, method)
                self.assertGreater(first[:8].mean(), first[8:].mean())
                self.assertGreater(other[8:].mean(), other[:8].mean())
                self.assertTrue(np.isfinite(first).all())
            finally:
                generator.close()
            self.assertEqual(len(model.conv._forward_hooks), before)
            torch.testing.assert_close(model(x), reference)

    def test_zero_map_has_no_overlay_tint_and_signed_legend_colors(self):
        image = np.full((8, 8, 3), 128, np.uint8)
        for signed in [False, True]:
            _, overlay = create_visualizations(image, np.zeros((8, 8)), signed=signed)
            np.testing.assert_array_equal(image, overlay)
        heatmap, _ = create_visualizations(image, np.tile([-1., 1.], (8, 4)), signed=True)
        self.assertEqual(heatmap[0, 0].tolist(), [0, 0, 255])
        self.assertEqual(heatmap[0, 1].tolist(), [255, 0, 0])

    def test_normalization_is_shared_rgb_and_not_bgr(self):
        image = np.full((224, 224, 3), [200, 40, 10], np.uint8)
        reference = image_transform('none')(Image.fromarray(image)).unsqueeze(0)
        torch.testing.assert_close(pixels(image[None]), reference)
        self.assertGreater(float(reference[0, 0].mean()), float(reference[0, 2].mean()))

    def test_perturbation_methods_keep_targets_and_scores(self):
        model = LocalModel().eval()
        x = torch.rand(1, 3, 16, 16)
        reference = model(x).detach()
        cam, details = occlusion_sensitivity(model, x, 0, patch=8)
        self.assertEqual(cam.shape, (16, 16))
        self.assertEqual(details['regions'], 4)
        first, detail = lime_explanation(model, x, 0, samples=80, grid=2)
        repeat, _ = lime_explanation(model, x, 0, samples=80, grid=2)
        np.testing.assert_array_equal(first, repeat)
        self.assertTrue(np.isfinite(first).all())
        self.assertIn('weighted_surrogate_r2', detail)
        torch.testing.assert_close(model(x), reference)

    def test_video_decision_view_batches_keep_other_frames_fixed(self):
        from inference.unified import ResearchPredictor, DecisionView
        from models.unified import FeatureHead
        p = ResearchPredictor.__new__(ResearchPredictor)
        p.id = 'S2'; p.temporal_head = None; p.device = torch.device('cpu')
        p.encoders = [nn.Sequential(nn.Flatten(), nn.Linear(12, 3))]
        p.spatial_head = FeatureHead(3)
        p.spatial_saved = {'feature_mean': torch.zeros(3), 'feature_std': torch.ones(3)}
        context = torch.randn(1, 8, 3)
        view = DecisionView(p, context, 2)
        x = torch.randn(4, 3, 2, 2)
        torch.testing.assert_close(view(x), torch.cat([view(row[None]) for row in x]))

    def test_family_means_exclude_experimental_and_other_protocol_runs(self):
        from inference.catalog import grouped
        def record(identifier, score, ranked=True, protocol='same'):
            return {'id':identifier,'family':'CNN','ranked':ranked,'evaluation_protocol':protocol,
                    'validation':{'balanced_accuracy':score,'macro_f1':score,'roc_auc':score}}
        result=grouped([record('a',.8),record('b',.6),record('experimental',.99,False),record('other_protocol',.7,protocol='other')])
        self.assertEqual(result[0]['best']['id'],'a')
        self.assertEqual(result[0]['comparable_runs'],2)
        self.assertAlmostEqual(result[0]['mean_balanced_accuracy'],.7)

    def test_negligible_attribution_is_not_amplified(self):
        model=nn.Sequential(nn.Flatten(),nn.Linear(12,2,bias=False))
        with torch.no_grad():model[1].weight.fill_(1e-15)
        cam,detail=integrated_gradients(model,torch.ones(1,3,2,2),0,8)
        self.assertTrue(detail['low_signal'])
        np.testing.assert_array_equal(cam,np.zeros((2,2)))

    def test_cam_plus_retains_small_valid_gradients(self):
        base=LocalModel().eval()
        class SmallScore(nn.Module):
            def __init__(self):
                super().__init__();self.base=base
            def forward(self,x):return self.base(x)*1e-4
        x=torch.ones(1,3,16,16)
        generator=GradCAM(SmallScore(),base.conv)
        try:
            cam=generator.generate(x,0,'gradcam++')
            self.assertGreater(float(cam.max()),.9)
            self.assertGreater(generator.details['gradient_abs_sum'],0)
        finally:generator.close()


if __name__ == '__main__':
    unittest.main()
