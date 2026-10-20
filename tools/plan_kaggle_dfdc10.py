"""Produce an audit and NOT-FOR-TRAINING candidate from the complete file catalog."""
import csv
import hashlib
import io
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import PurePosixPath

from tools.dfdc_storage import ROOT, RESEARCH, GIB, LIMIT, PHASE_DIRS, atomic_json, check_budget, phase_bytes, tree_bytes
from tools.inventory_kaggle_dfdc10 import OUT
from training.dfdc_data import load_metadata, split_groups


def save_csv(path, rows, fields):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    data = stream.getvalue().encode('utf-8')
    check_budget(len(data))
    path.write_bytes(data)


def main():
    inventory = json.loads((OUT / 'catalog.json').read_text(encoding='utf-8'))
    if not inventory['pagination_complete']:
        raise ValueError('Refusing to plan from an incomplete catalog')
    cached_metadata_path = RESEARCH / 'analyses/mirror_candidates/hf_part00_metadata.json'
    cached = load_metadata(cached_metadata_path)
    prefix = 'dfdc_train_part_00/dfdc_train_part_0/'
    videos = [row for row in inventory['files'] if row['name'].startswith(prefix) and row['name'].endswith('.mp4')]
    by_name = {PurePosixPath(row['name']).name: row for row in videos}
    if set(cached) != set(by_name):
        raise ValueError('Other-mirror metadata names do not exactly match Kaggle part 00; candidate refused')
    source_listing = json.loads((RESEARCH / 'analyses/mirror_candidates/hf_part00_complete_tree.json').read_text(encoding='utf-8'))
    if isinstance(source_listing, dict):
        source_listing = source_listing.get('files', source_listing.get('entries', []))
    # Match sizes across sources where the previous full HF listing is available.
    hf_sizes = {PurePosixPath(row['path']).name: row['size'] for row in source_listing
                if isinstance(row, dict) and row.get('path', '').endswith('.mp4')}
    if set(hf_sizes) != set(by_name) or any(hf_sizes[name] != row['size'] for name, row in by_name.items()):
        raise ValueError('Kaggle and cached HF video file names/sizes differ')
    mapping, groups, warnings = split_groups(cached, seed=42)
    target = ROOT.parent / f"datasets/dfdc10_kaggle_v{inventory['pinned_dataset'].split('/')[-1]}"
    if target.is_relative_to(ROOT) or target.drive.upper() != 'D:':
        raise ValueError('Dataset destination must be on D: outside implementation repository')
    candidate = []
    counts = defaultdict(Counter)
    groups_by_split = defaultdict(set)
    for name, info in sorted(cached.items()):
        group, split = mapping[name]
        row = {'dataset_ref': inventory['pinned_dataset'], 'remote_name': by_name[name]['name'],
               'expected_bytes': by_name[name]['size'], 'label_from_other_mirror': info['label'],
               'original_from_other_mirror': info.get('original', ''), 'source_group': group,
               'provisional_split': split, 'planned_local_path': str(target / by_name[name]['name']),
               'status': 'NOT_DOWNLOADED_NOT_KAGGLE_LABEL_VERIFIED', 'training_eligible': False}
        candidate.append(row)
        counts[split][info['label']] += 1
        groups_by_split[split].add(group)
    splits = ['train', 'validation', 'test']
    assert all(not groups_by_split[a].intersection(groups_by_split[b]) for i, a in enumerate(splits) for b in splits[i+1:])
    assert all(not info.get('original') or mapping[name][1] == mapping[info['original']][1] for name, info in cached.items())
    save_csv(OUT / 'candidate_manifest_NOT_FOR_TRAINING.csv', candidate, list(candidate[0]))
    part_rows = [{'part': part, 'video_count': item['video_count'], 'video_bytes': item['video_bytes'],
                  'metadata_count': len(item['metadata_files']),
                  'metadata_bytes': sum(row['bytes'] for row in item['metadata_files']),
                  'all_file_count': item['file_count'], 'all_file_bytes': item['listed_bytes']}
                 for part, item in inventory['parts'].items()]
    save_csv(OUT / 'part_totals.csv', part_rows, list(part_rows[0]))
    dataset_paths = [ROOT.parent / 'datasets', ROOT.parent / 'data']
    external_bytes = sum(tree_bytes(path) for path in dataset_paths if path.exists())
    current_bytes = phase_bytes() + external_bytes
    raw_bytes = sum(row['size'] for row in videos)
    metadata_file = inventory['parts']['00']['metadata_files'][0]
    reserves = {'preprocessing_caches': 3 * GIB, 'temporary_single_file_downloads': 2 * GIB,
                'pretrained_weights_and_checkpoints': 4 * GIB, 'experiment_outputs': GIB}
    budget = {'limit_bytes': LIMIT, 'D_free_bytes': shutil.disk_usage(ROOT).free,
              'existing_counted_phase_bytes': current_bytes, 'external_dataset_bytes': external_bytes,
              'remaining_phase_bytes_now': LIMIT - current_bytes,
              'proposed_raw_video_bytes': raw_bytes, 'proposed_metadata_bytes': metadata_file['bytes'],
              'reserved_future_bytes': reserves,
              'projected_total_with_reserves': current_bytes + raw_bytes + metadata_file['bytes'] + sum(reserves.values()),
              'projected_remaining_phase_bytes': LIMIT - current_bytes - raw_bytes - metadata_file['bytes'] - sum(reserves.values()),
              'accounting_note': 'Conservative count includes all phase directories, research, existing model checkpoint, and measured environment growth; original environment and old image dataset are baseline.'}
    assert budget['projected_total_with_reserves'] < LIMIT
    assert budget['D_free_bytes'] - raw_bytes - sum(reserves.values()) > 5 * GIB
    fake = min((name for name, info in cached.items() if info['label'] == 'FAKE'),
               key=lambda name: (by_name[name]['size'] + by_name[cached[name]['original']]['size'], name))
    first = {'remote_video': by_name[fake]['name'], 'expected_video_bytes': by_name[fake]['size'],
             'provisional_label_from_other_mirror': 'FAKE', 'provisional_original': cached[fake]['original'],
             'remote_metadata': metadata_file['name'], 'expected_metadata_bytes': metadata_file['bytes'],
             'total_expected_bytes': by_name[fake]['size'] + metadata_file['bytes'],
             'status': 'NOT_ATTEMPTED_PENDING_PERMISSION_EVIDENCE_AND_KAGGLE_METADATA_VALIDATION'}
    plan = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
            'dataset': inventory['pinned_dataset'], 'conditional_user_download_instruction_received': True,
            'permission_established': False, 'status': 'HOLD_UNRESOLVED_REDISTRIBUTION_PERMISSION',
            'planned_dataset_root': str(target), 'root_exists': target.exists(),
            'dataset_media_bytes_downloaded': 0, 'metadata_payloads_downloaded_from_kaggle': 0,
            'decoded_video_count': 0, 'training_video_count': 0,
            'metadata_files_found': inventory['metadata_json_count'],
            'kaggle_label_contents_verified': False,
            'candidate': {'part': '00', 'video_count': len(videos), 'raw_video_bytes': raw_bytes,
                          'labels_from_cached_other_mirror': dict(Counter(info['label'] for info in cached.values())),
                          'source_groups': len(groups), 'labels_source_path': str(cached_metadata_path),
                          'labels_source_sha256': hashlib.sha256(cached_metadata_path.read_bytes()).hexdigest(),
                          'kaggle_names_match_cached_labels': True, 'kaggle_video_names_sizes_match_cached_hf_listing': True,
                          'metadata_content_identity_between_sources_established': False,
                          'provisional_split_counts': {split: dict(counts[split]) for split in splits},
                          'provisional_groups_by_split': {split: len(groups_by_split[split]) for split in splits},
                          'no_source_group_or_original_fake_split_overlap': True, 'training_eligible': False,
                          'warnings': warnings + ['Only 86 real source videos; strongly imbalanced development subset, not a full DFDC benchmark.']},
            'first_validation_candidate': first, 'storage': budget,
            'actual_splits': {split: {'videos': 0, 'REAL': 0, 'FAKE': 0} for split in splits}}
    atomic_json(OUT / 'conditional_subset_plan.json', plan)
    # Recalculate after writing all major audit artifacts.
    final_storage = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
                     'D_free_bytes': shutil.disk_usage(ROOT).free,
                     'counted_phase_bytes': phase_bytes() + external_bytes,
                     'external_dataset_bytes': external_bytes,
                     'remaining_phase_bytes': LIMIT - phase_bytes() - external_bytes,
                     'audit_folder_bytes': tree_bytes(OUT), 'limit_bytes': LIMIT,
                     'directory_bytes': {name: tree_bytes(ROOT / name) for name in PHASE_DIRS},
                     'research_bytes': tree_bytes(RESEARCH)}
    atomic_json(OUT / 'storage_snapshot.json', final_storage)
    print(json.dumps({'plan': str(OUT / 'conditional_subset_plan.json'), 'candidate': plan['candidate'],
                      'first_validation_candidate': first, 'storage': budget, 'current_storage_snapshot': final_storage}, indent=2))


if __name__ == '__main__':
    main()
