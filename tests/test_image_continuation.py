"""Focused protocol, state, mapping, and validation-selection regression checks."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch import nn

from inference.contracts import class_probabilities
from training.demo_accuracy import image_metrics, sha
from training.image_continuation import (
    IDENTITIES, PROTOCOL, atomic_checkpoint, better, epoch_numbers, loss_progress,
    checked_training_state, SplitFolder, SplitSampler, PrefetchLoader,
)


class ContinuationTests(unittest.TestCase):
    def test_existing_identities_and_epoch_caps(self):
        self.assertEqual(IDENTITIES, ('R18_FT_20261021', 'E02_efficientnet_b0'))
        self.assertEqual(list(epoch_numbers(IDENTITIES[0], 6)), [7, 8, 9, 10])
        self.assertEqual(list(epoch_numbers(IDENTITIES[1], 0)), list(range(1, 11)))
        self.assertEqual(list(epoch_numbers(IDENTITIES[1], 3)), list(range(4, 11)))
        self.assertEqual(list(epoch_numbers(IDENTITIES[0], 10)), [])
        for identity, epoch in [(IDENTITIES[0], 2), (IDENTITIES[0], 11), ('E05', 0)]:
            with self.assertRaises(ValueError):
                epoch_numbers(identity, epoch)

    def test_loss_stopping_is_separate_from_balanced_accuracy_selection(self):
        self.assertEqual(PROTOCOL['patience'], 3)
        loss, bad = .04, 0
        for value in [.0401, .03995, .04]:
            loss, bad = loss_progress(value, loss, bad)
        self.assertEqual(bad, 3)
        self.assertTrue(better({'balanced_accuracy': .99, 'loss': .05}, {'balanced_accuracy': .98, 'loss': .04}))
        self.assertFalse(better({'balanced_accuracy': .98, 'loss': .01}, {'balanced_accuracy': .99, 'loss': .04}))
        self.assertTrue(better({'balanced_accuracy': .99, 'loss': .03}, {'balanced_accuracy': .99, 'loss': .04}))
        # Test scores are neither read nor accepted as a selection criterion.
        self.assertFalse(better({'balanced_accuracy': .98, 'loss': .04, 'test_accuracy': 1.}, {'balanced_accuracy': .99, 'loss': .04}))

    def test_fake_positive_conversion_and_classwise_metrics(self):
        logits = torch.tensor([[4., 0.], [0., 4.], [0., 4.], [4., 0.]])
        legacy_targets = torch.tensor([0, 1, 0, 1])
        probabilities = class_probabilities(logits, {'fake': 0, 'real': 1})['fake']
        metrics = image_metrics((legacy_targets == 0).int().tolist(), probabilities.tolist())
        self.assertEqual(metrics['confusion_matrix_real_fake'], [[1, 1], [1, 1]])
        self.assertEqual(metrics['false_positives'], 1)
        self.assertEqual(metrics['false_negatives'], 1)
        self.assertEqual(metrics['weighted_f1'], .5)
        self.assertEqual(metrics['classwise']['FAKE']['recall'], .5)
        self.assertEqual(metrics['specificity'], .5)

    def test_atomic_roundtrip_and_best_restoration(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'selected.pth'
            model = nn.Linear(2, 2)
            original = {k: v.clone() for k, v in model.state_dict().items()}
            digest = atomic_checkpoint(path, {'class_to_idx': {'fake': 0, 'real': 1}, 'model_state_dict': original, 'epoch': 6})
            self.assertEqual(digest, sha(path))
            with torch.no_grad():
                model.weight.add_(3)
            loaded = torch.load(path, map_location='cpu', weights_only=True)
            model.load_state_dict(loaded['model_state_dict'])
            self.assertEqual(loaded['epoch'], 6)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, original[key])
            self.assertFalse(path.with_suffix('.pth.tmp').exists())
            with self.assertRaises(ValueError):
                atomic_checkpoint(path, {'class_to_idx': {'real': 0, 'fake': 1}, 'model_state_dict': original})
            self.assertEqual(sha(path), digest)

    def test_resume_hash_and_pending_commit_recovery(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'last.pt'
            model = nn.Linear(2, 2)
            payload = {'class_to_idx': {'fake': 0, 'real': 1}, 'model_state_dict': model.state_dict(), 'completed_epoch': 7}
            digest = atomic_checkpoint(path, payload)
            self.assertEqual(checked_training_state(path)['completed_epoch'], 7)
            sidecar = path.with_suffix('.pt.sha256.json')
            sidecar.write_text(json.dumps({'sha256': 'old transaction'}))
            path.with_suffix('.pt.pending.json').write_text(json.dumps({'sha256': digest}))
            self.assertEqual(checked_training_state(path)['completed_epoch'], 7)
            self.assertFalse(path.with_suffix('.pt.pending.json').exists())
            with path.open('ab') as stream:
                stream.write(b'tampered')
            with self.assertRaisesRegex(ValueError, 'SHA-256'):
                checked_training_state(path)

    def test_shared_pool_preserves_predefined_split_isolation(self):
        import numpy as np
        from PIL import Image
        from torch.utils.data import DataLoader
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for split, per_class in [('train', 2), ('valid', 1)]:
                for target, cls in enumerate(['fake', 'real']):
                    directory = root/split/cls; directory.mkdir(parents=True)
                    for index in range(per_class):
                        Image.fromarray(np.full((20, 20, 3), target*100+index, np.uint8)).save(directory/f'{index}.png')
            with patch('training.image_continuation.TRAIN_PATH', root/'train'), patch('training.image_continuation.VALID_PATH', root/'valid'):
                dataset = SplitFolder()
                generator = torch.Generator().manual_seed(42)
                data = DataLoader(dataset, batch_size=2, sampler=SplitSampler(dataset, generator),
                                  num_workers=1, persistent_workers=True)
                dataset.mode.value = 0; dataset.epoch.value = 7
                training = list(data)
                self.assertEqual(sum(len(batch[1]) for batch in training), 4)
                self.assertTrue(all('train' in Path(p).parts for p, _ in dataset.samples))
                dataset.mode.value = 1
                validation = list(data)
                self.assertEqual(sum(len(batch[1]) for batch in validation), 2)
                self.assertEqual(validation[0][1].tolist(), [0, 1])
                self.assertEqual(validation[0][2].tolist(), [0, 1])
                self.assertTrue(all('valid' in Path(p).parts for p, _ in dataset.samples))

    def test_threaded_decode_matches_existing_pil_augmentation_exactly(self):
        import numpy as np
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for split in ['train', 'valid']:
                for cls in ['fake', 'real']:
                    directory = root/split/cls; directory.mkdir(parents=True)
                    image = np.random.default_rng(42).integers(0, 255, (256, 256, 3), dtype=np.uint8)
                    Image.fromarray(image).save(directory/'0.jpg')
            with patch('training.image_continuation.TRAIN_PATH', root/'train'), patch('training.image_continuation.VALID_PATH', root/'valid'):
                dataset = SplitFolder(); dataset.mode.value = 0; dataset.epoch.value = 8
                before_cuda = torch.cuda.get_rng_state().clone() if torch.cuda.is_initialized() else None
                from torch.utils.data import DataLoader
                for epoch in range(7, 23):
                    dataset.epoch.value = epoch
                    reference = [dataset[i] for i in [0, 1]]
                    dataset.threaded = True
                    batches = list(PrefetchLoader(DataLoader(dataset, batch_size=2, shuffle=False)))
                    actual = [(batches[0][0][i], int(batches[0][1][i]), int(batches[0][2][i])) for i in [0, 1]]
                    for a, b in zip(reference, actual):
                        self.assertEqual(a[1:], b[1:])
                        torch.testing.assert_close(a[0], b[0], atol=0, rtol=0)
                dataset.close()
                if before_cuda is not None:
                    self.assertTrue(torch.equal(before_cuda, torch.cuda.get_rng_state()))


if __name__ == '__main__':
    unittest.main()
