"""Prevent silent image-data leakage and training on held-out images."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from tools.dfdc_storage import ROOT
from training.image_experiment import ImageRows, load_rows, prepare_manifest


class ImageSplitTests(unittest.TestCase):
    def test_hash_leakage_and_heldout_isolation(self):
        base=ROOT/'results/smoke';base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base,prefix='image_split_') as directory:
            directory=Path(directory).resolve()
            self.assertTrue(directory.is_relative_to(base.resolve()))
            dataset=directory/'images'
            for i,split in enumerate(['train','valid','test']):
                for j,label in enumerate(['fake','real']):
                    folder=dataset/split/label;folder.mkdir(parents=True)
                    for k in range(2):
                        Image.fromarray(np.full((16,16,3),i*60+j*20+k,np.uint8)).save(folder/f'{k}.png')
            with patch('training.image_experiment.DATASET_PATH',dataset):
                report=prepare_manifest(directory/'prepared',4,4)
                rows,_=load_rows(directory/'prepared/manifest.csv')
                training=ImageRows(rows,'train',augmentation='jpeg')
                validation=ImageRows(rows,'validation')
                self.assertEqual(len(training),4)
                self.assertTrue(all(row['split']=='train' for row in training.rows))
                self.assertTrue(all(row['split']=='validation' for row in validation.rows))
                self.assertFalse(report['exact_hash_overlap_across_selected_splits'])
                # A renamed copy across train/test must be rejected by content.
                duplicate=dataset/'test/real/renamed.png'
                duplicate.write_bytes((dataset/'train/fake/0.png').read_bytes())
                with self.assertRaisesRegex(ValueError,'duplicate crosses splits'):
                    prepare_manifest(directory/'leaked',4,4)


if __name__=='__main__': unittest.main()
