"""Audit saved evaluations and validation-only threshold alternatives; never open test media."""
import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np
from evaluation.dfdc import binary_metrics
from inference.catalog import catalog, grouped
from tools.acquire_dfdc import sha256
from training.video_protocol import RUN


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def select_threshold(y, p):
    y, p = np.asarray(y), np.asarray(p)
    candidates = np.unique(np.r_[p, .5, 0., 1.])
    predictions = p[None, :] >= candidates[:, None]
    tp = ((y == 1) & predictions).sum(1)
    tn = ((y == 0) & ~predictions).sum(1)
    fp = ((y == 0) & predictions).sum(1)
    fn = ((y == 1) & ~predictions).sum(1)
    ba = .5*(tp/max(int((y == 1).sum()), 1)+tn/max(int((y == 0).sum()), 1))
    f1 = .5*(2*tp/np.maximum(2*tp+fp+fn, 1)+2*tn/np.maximum(2*tn+fp+fn, 1))
    index = max(range(len(candidates)), key=lambda i: (ba[i], f1[i], -abs(candidates[i]-.5)))
    return float(candidates[index])


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = read_csv(RUN/'manifest.csv')
    groups, hashes, aliases = {}, {}, {}
    by_name = {row['filename'].casefold(): row for row in rows}
    for row in rows:
        if groups.setdefault(row['source_group'], row['split']) != row['split']:
            raise ValueError('Source group leakage')
        if hashes.setdefault(row['sha256'], row['split']) != row['split']:
            raise ValueError('Duplicate file leakage')
        original = row['original'].casefold()
        if original:
            if aliases.setdefault(original, row['source_group']) != row['source_group']:
                raise ValueError('Missing-original sibling leakage')
            if original in by_name and by_name[original]['source_group'] != row['source_group']:
                raise ValueError('Original/derivative leakage')
    audit = {'manifest_sha256': sha256(RUN/'manifest.csv'), 'split_counts': dict(Counter(row['split'] for row in rows)),
             'split_class_counts': dict(Counter(row['split']+'/'+row['label'] for row in rows)),
             'groups_by_split': {split: len({row['source_group'] for row in rows if row['split'] == split}) for split in ['train','validation','test']},
             'source_and_duplicate_leakage_found': False, 'models': {},
             'threshold_policy': 'Alternatives below use validation only. Historical frozen threshold 0.5 and test outputs are unchanged.'}
    for summary_path in sorted(RUN.glob('S*/summary.json')):
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        predictions = read_csv(summary_path.parent/'validation_predictions.csv')
        y = np.array([int(row['label'] == 'FAKE') for row in predictions])
        p = np.array([float(row['raw_fake_probability']) for row in predictions])
        recalculated = binary_metrics(y, p)
        for key in ['accuracy', 'balanced_accuracy', 'macro_f1']:
            if abs(recalculated[key]-summary['validation'][key]) > 1e-10:
                raise ValueError('Saved metric mismatch: '+str(summary_path))
        threshold = select_threshold(y, p)
        group_names = np.array([row['source_group'] for row in predictions])
        heldout_scores = np.zeros(len(y))
        folds = []
        for group in sorted(set(group_names)):
            train = group_names != group
            test = ~train
            if len(set(y[train])) < 2:
                threshold_fold = .5
            else:
                threshold_fold = select_threshold(y[train], p[train])
            heldout_scores[test] = (p[test] >= threshold_fold).astype(float)
            folds.append({'excluded_source_group': group, 'threshold': threshold_fold})
        audit['models'][summary_path.parent.name] = {'recalculated_validation': recalculated,
            'calibration': summary['calibration'], 'best_in_sample_validation_threshold': threshold,
            'threshold_fitted_validation': binary_metrics(y, p, threshold),
            'leave_one_source_group_out_threshold_decisions': binary_metrics(y, heldout_scores),
            'threshold_folds': folds, 'deployed_threshold_changed': False}
    audit['leaderboards'] = {}
    for modality in ['Image', 'Video']:
        records = catalog(modality)
        audit['leaderboards'][modality] = [{'family': group['family'], 'best': group['best']['id'],
            'best_balanced_accuracy': group['best']['validation']['balanced_accuracy'],
            'mean_balanced_accuracy': group['mean_balanced_accuracy'], 'comparable_runs': group['comparable_runs']}
            for group in grouped(records)]
    with (output/'evaluation_audit.json').open('x', encoding='utf-8') as stream:
        json.dump(audit, stream, indent=2, allow_nan=False)
    print(json.dumps({'splits':audit['split_class_counts'], 'leaderboards':audit['leaderboards'],
                      'S2':{key:value for key,value in audit['models']['S2'].items() if key not in ['calibration','threshold_folds']}}, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
