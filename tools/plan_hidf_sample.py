"""Create a review-only paired HiDF sample plan; HTTP HEAD only, no media GET."""
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import PurePosixPath

from tools.dfdc_storage import RESEARCH, atomic_json
from tools.probe_dfdc_payloads import head
from tools.runtime import configure_runtime


def main():
    configure_runtime()
    audit = json.loads((RESEARCH / 'analyses/licensed_dataset_source_audit.json').read_text(encoding='utf-8'))
    source = audit['results']['hidf_tree']['data']
    if source.get('truncated'):
        raise ValueError('Cannot plan from a truncated source listing')
    revision = source['sha']
    rows = [row for row in source['tree'] if row['type'] == 'blob'
            and row['path'].startswith('Code/SampleData/HiDF/') and row['path'].endswith('.mp4')]
    real = {PurePosixPath(row['path']).stem: row for row in rows if '/Real/' in row['path']}
    pairs = []
    used = set()
    for fake in sorted((row for row in rows if '/Fake/' in row['path']), key=lambda row: row['path']):
        source_id = PurePosixPath(fake['path']).stem.split('_')[0]
        if source_id in real and source_id not in used:
            pairs.append((source_id, real[source_id], fake))
            used.add(source_id)
    selected = []
    for source_id, real_row, fake_row in pairs[:10]:
        for label, row in [('REAL', real_row), ('FAKE', fake_row)]:
            selected.append({'source_group': source_id, 'label': label, 'path': row['path'],
                             'size_bytes': row['size'], 'git_blob_sha1': row['sha'],
                             'url': f"https://raw.githubusercontent.com/DSAIL-SKKU/HiDF/{revision}/{row['path']}"})
    if len(selected) != 20:
        raise ValueError('Need exactly ten matched source pairs')
    with ThreadPoolExecutor(max_workers=3) as executor:
        headers = list(executor.map(head, (row['url'] for row in selected)))
    for row, headers_row in zip(selected, headers):
        row['head'] = headers_row
    extra = {'lavdf_unauthenticated_head': head('https://huggingface.co/datasets/ControlNet/LAV-DF/resolve/c9158b3b9ad7f05eca712da18b8722d79d7f4d8e/LAV-DF.tar')}
    result = {
        'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'REVIEW_ONLY_AWAITING_USER_CHOICE_NO_DOWNLOAD_AUTHORIZATION',
        'dataset': 'HiDF author-provided evaluation MP4 samples; unrelated to DFDC',
        'repository': 'https://github.com/DSAIL-SKKU/HiDF', 'revision': revision,
        'license': 'CC-BY-NC-4.0 for dataset; code MIT is separate',
        'license_url': f'https://github.com/DSAIL-SKKU/HiDF/blob/{revision}/LICENSE',
        'author_dataset_license_statement': 'https://github.com/DSAIL-SKKU/HiDF#-license',
        'labels_from': 'Author-provided Real/Fake directories, normalized to REAL/FAKE',
        'source_groups_from': 'Author-documented base_video_id_target_image_id naming; first component matched to real stem',
        'selection': 'Ten distinct source IDs in lexicographic fake-file order with a matching real file; smoke testing only',
        'full_public_hidf_sample': {'videos': len(rows), 'bytes': sum(row['size'] for row in rows),
                                   'by_class': dict(Counter('REAL' if '/Real/' in row['path'] else 'FAKE' for row in rows))},
        'matched_source_pairs_available': len(pairs),
        'selected_video_bytes': sum(row['size_bytes'] for row in selected),
        'selected_labels': dict(Counter(row['label'] for row in selected)),
        'all_selected_head_sizes_match': all(row['head'].get('status') == 200
                                           and row['head'].get('content_length') == str(row['size_bytes']) for row in selected),
        'dataset_media_bytes_downloaded': 0,
        'not_yet_verified': ['MP4 decoding, resolution, duration, hashes of downloaded bodies',
                             'Whether these evaluation samples preserve full-release resolution',
                             'Decoded-frame equality/content matching of source pairs'],
        'files': selected, 'other_source_headers': extra,
    }
    atomic_json(RESEARCH / 'analyses/hidf_review_only_sample_plan.json', result)
    print(json.dumps({k: value for k, value in result.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()
