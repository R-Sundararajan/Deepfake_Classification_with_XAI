import threading
import unittest
from concurrent.futures import wait

import numpy as np
import torch
from torch import nn

from explainability.api import explain
from explainability.suite import ParallelExplainer, clone_for_explanation
from models.unified import FeatureHead


class ParallelExplanationTests(unittest.TestCase):
    def test_clone_isolates_weights_and_preserves_hook_layer_reference(self):
        class Predictor:
            pass
        original = Predictor()
        original.model = nn.Sequential(nn.Conv2d(3, 2, 1), nn.AdaptiveAvgPool2d(1))
        original.layer = original.model[0]
        original.encoders = [original.model]
        original.lock = threading.RLock()
        clone = clone_for_explanation(original)
        self.assertIs(clone.layer, clone.model[0])
        self.assertIs(clone.encoders[0], clone.model)
        self.assertIsNot(clone.layer, original.layer)
        self.assertNotEqual(clone.layer.weight.data_ptr(), original.layer.weight.data_ptr())
        with torch.no_grad():
            clone.layer.weight.zero_()
        self.assertGreater(float(original.layer.weight.abs().sum()), 0)

    def test_concurrent_models_keep_hooks_targets_and_scores_isolated(self):
        barrier = threading.Barrier(2)
        instances = []

        class Predictor:
            def __init__(self, record):
                self.device = torch.device('cpu')
                self.lock = threading.RLock()
                self.model = nn.Sequential(nn.Conv2d(3, 2, 1, bias=False), nn.AdaptiveAvgPool2d(1), nn.Flatten()).eval()
                with torch.no_grad():
                    self.model[0].weight.copy_(torch.arange(6).reshape(2, 3, 1, 1) / 5)
                self.first = True
                self.calls = []
                instances.append(self)

            def explain_result(self, result, method, target, **parameters):
                if self.first:
                    self.first = False
                    barrier.wait(timeout=10)  # Fails if the suite serializes work.
                self.calls.append((method, target))
                tensor = result['tensor']
                reference = self.model(tensor).detach()
                if method == 'lime' and result.get('fail_lime'):
                    raise ValueError('deliberate method failure')
                cam, details = explain(self.model, tensor, int(target == 'FAKE'), method, layer=self.model[0], **parameters)
                torch.testing.assert_close(self.model(tensor), reference)
                if self.model[0]._forward_hooks:
                    raise AssertionError('Attribution leaked a model hook')
                return {'cam': cam, 'details': details, 'method': method}

        record = {'id': 'toy', 'checkpoint_sha256': 'hash', 'source': 'test', 'checkpoint': 'toy', 'config': {'backbones': ['cnn']}}
        result = {'prediction': 'FAKE', 'tensor': torch.linspace(0, 1, 3 * 64 * 64).reshape(1, 3, 64, 64)}
        runner = ParallelExplainer(factory=Predictor, max_suites=2)
        try:
            suite = runner.start(record, result, 'first', 'Image')
            outputs = {method: future.result(timeout=30) for method, future in suite.futures.items()}
            self.assertEqual(len(instances), 2)
            self.assertEqual(len(outputs), 6)
            self.assertTrue(all(np.isfinite(row['cam']).all() for row in outputs.values()))
            self.assertEqual({target for p in instances for _, target in p.calls}, {'FAKE'})
            self.assertIs(suite, runner.start(record, result, 'first', 'Image'))
            self.assertEqual(sum(len(p.calls) for p in instances), 6)
            spans = [row['details']['execution'] for row in outputs.values()]
            self.assertEqual(len({row['worker'] for row in spans}), 2)
            self.assertTrue(any(a['started'] < b['finished'] and b['started'] < a['finished']
                                for a in spans for b in spans if a['worker'] != b['worker']))
            # A failing method does not discard other maps, and a new upload
            # or changed decision cannot reuse the previous suite.
            changed = {**result, 'prediction': 'REAL', 'fail_lime': True}
            other = runner.start(record, changed, 'second', 'Image')
            wait(other.futures.values(), timeout=30)
            self.assertIsNot(other, suite)
            with self.assertRaisesRegex(ValueError, 'deliberate'):
                other.futures['lime'].result()
            self.assertTrue(all(future.exception() is None for method, future in other.futures.items() if method != 'lime'))
            self.assertEqual({target for p in instances for _, target in p.calls[-1:]}, {'REAL'})
        finally:
            runner.close()

    def test_explicit_lstm_gates_match_outputs_and_input_gradients(self):
        torch.manual_seed(21)
        head = FeatureHead(5, 'temporal').eval()
        x = torch.randn(3, 8, 5, requires_grad=True)
        expected = head(x)
        actual = head.attribution_forward(x)
        torch.testing.assert_close(actual, expected)
        expected_gradient = torch.autograd.grad(expected[:, 1].sum(), x)[0]
        actual_gradient = torch.autograd.grad(actual[:, 1].sum(), x)[0]
        torch.testing.assert_close(actual_gradient, expected_gradient)
        self.assertTrue(torch.isfinite(actual_gradient).all())


if __name__ == '__main__':
    unittest.main()
