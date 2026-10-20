"""Enumerate the version-pinned Kaggle catalog with CLI --page-token; no payload download."""
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import PurePosixPath

from tools.dfdc_storage import RESEARCH, ROOT, atomic_json, check_budget
from tools.runtime import configure_runtime

DATASET = 'pranay22077/dfdc-10'
OUT = RESEARCH / 'analyses/kaggle_dfdc10_inventory'
CLI = ROOT / 'dfenv/Scripts/kaggle.exe'


def now():
    return datetime.now(timezone.utc).isoformat()


def cli(arguments):
    response = subprocess.run([str(CLI), *arguments], capture_output=True, text=True,
                              encoding='utf-8', errors='replace', timeout=90)
    if response.returncode:
        # Avoid retaining credential-bearing HTTP errors or signed URLs.
        codes = re.findall(r'\b(?:401|403|404|429|500|502|503|504)\b', response.stderr + response.stdout)
        raise RuntimeError(f'Kaggle CLI request failed (exit {response.returncode}; HTTP codes {sorted(set(codes))})')
    return response.stdout


def parse_page(stdout):
    token = re.search(r'^Next Page Token = (\S+)\s*$', stdout, flags=re.MULTILINE)
    start = re.search(r'^\[\s*$', stdout, flags=re.MULTILINE)
    if start is None:
        raise ValueError('Kaggle response has no JSON file array; catalog completeness not established')
    rows = json.loads(stdout[start.start():])
    if not isinstance(rows, list) or len(rows) > 200:
        raise ValueError('Invalid file-list page')
    for row in rows:
        path = row.get('name')
        size = row.get('size')
        if not isinstance(path, str) or not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError('Invalid file name/size')
        if PurePosixPath(path).is_absolute() or '..' in PurePosixPath(path).parts or '\\' in path:
            raise ValueError('Unsafe catalog path')
    return rows, token.group(1) if token else ''


def summarize(rows):
    parts = defaultdict(lambda: {'file_count': 0, 'listed_bytes': 0, 'video_count': 0,
                                  'video_bytes': 0, 'metadata_files': [], 'other_files': []})
    for row in rows:
        match = re.match(r'dfdc_train_part_(\d+)/', row['name'])
        part = f'{int(match.group(1)):02d}' if match else 'outside_training_parts'
        item = parts[part]
        item['file_count'] += 1
        item['listed_bytes'] += row['size']
        if row['name'].lower().endswith('.mp4'):
            item['video_count'] += 1
            item['video_bytes'] += row['size']
        elif PurePosixPath(row['name']).name == 'metadata.json':
            item['metadata_files'].append({'name': row['name'], 'bytes': row['size']})
        else:
            item['other_files'].append({'name': row['name'], 'bytes': row['size']})
    return {key: value for key, value in sorted(parts.items())}


def main():
    configure_runtime()
    os.environ['VIRTUAL_ENV'] = str(ROOT / 'dfenv')
    os.environ['PATH'] = str(ROOT / 'dfenv/Scripts') + os.pathsep + os.environ['PATH']
    check_budget(24 * 1024**2)
    OUT.mkdir(parents=True, exist_ok=True)
    state_path = OUT / 'state.json'
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding='utf-8'))
        if state['dataset'] != DATASET:
            raise ValueError('Existing state belongs to another dataset')
    else:
        # Public version metadata does not require the owner-only processing-status route.
        status_text = cli(['datasets', 'status', DATASET, '--format', 'json(current_version_number)'])
        start = status_text.find('{')
        status = json.loads(status_text[start:]) if start >= 0 else {}
        version = status.get('current_version_number')
        if not isinstance(version, int) or version < 1:
            raise ValueError('Unable to pin current Kaggle dataset version')
        state = {'dataset': DATASET, 'version': version, 'pinned_dataset': f'{DATASET}/{version}',
                 'started_at_utc': now(), 'complete': False, 'page_count': 0}
        atomic_json(state_path, state)
        cli(['datasets', 'metadata', state['pinned_dataset'], '-p', str(OUT)])
    rows = []
    seen_names = set()
    seen_tokens = set()
    token = ''
    page_number = 1
    while page_number <= 500:
        page_path = OUT / f'page_{page_number:03d}.json'
        if page_path.exists():
            page = json.loads(page_path.read_text(encoding='utf-8'))
            if page['dataset'] != state['pinned_dataset'] or page['input_token'] != token:
                raise ValueError('Saved page version/token chain mismatch')
            entries, next_token = page['files'], page['next_token']
        else:
            args = ['datasets', 'files', state['pinned_dataset'], '--page-size', '200', '--format', 'json']
            if token:
                args.extend(['--page-token', token])
            stdout = cli(args)
            entries, next_token = parse_page(stdout)
            page = {'dataset': state['pinned_dataset'], 'page': page_number, 'checked_at_utc': now(),
                    'input_token': token, 'next_token': next_token,
                    'stdout_sha256': hashlib.sha256(stdout.encode('utf-8')).hexdigest(), 'files': entries}
            atomic_json(page_path, page)
        for entry in entries:
            if entry['name'].casefold() in seen_names:
                raise ValueError('Duplicate file across catalog pages; refusing incomplete/mixed inventory')
            seen_names.add(entry['name'].casefold())
            rows.append(entry)
        state.update({'page_count': page_number, 'listed_file_count': len(rows),
                      'listed_bytes': sum(row['size'] for row in rows),
                      'last_checked_at_utc': now(), 'complete': not bool(next_token)})
        atomic_json(state_path, state)
        if page_number % 10 == 0 or not next_token:
            print(json.dumps({'pages': page_number, 'files': len(rows), 'listed_GiB': round(state['listed_bytes'] / 1024**3, 3),
                              'next_page_available': bool(next_token)}), flush=True)
        if not next_token:
            break
        if not entries or next_token in seen_tokens:
            raise ValueError('Repeated pagination token or empty nonterminal page')
        seen_tokens.add(next_token)
        token = next_token
        page_number += 1
    else:
        raise ValueError('Safety page limit reached; catalog incomplete')
    first = json.loads((OUT / 'page_001.json').read_text(encoding='utf-8'))['files']
    catalog = {'dataset': DATASET, 'pinned_dataset': state['pinned_dataset'], 'checked_at_utc': now(),
               'pagination_complete': state['complete'], 'page_count': state['page_count'],
               'unique_file_count': len(rows), 'listed_bytes': sum(row['size'] for row in rows),
               'video_count': sum(row['name'].lower().endswith('.mp4') for row in rows),
               'video_bytes': sum(row['size'] for row in rows if row['name'].lower().endswith('.mp4')),
               'metadata_json_count': sum(PurePosixPath(row['name']).name == 'metadata.json' for row in rows),
               'first_page_count': len(first), 'first_page_bytes': sum(row['size'] for row in first),
               'extensions': dict(Counter(PurePosixPath(row['name']).suffix.lower() for row in rows)),
               'parts': summarize(rows), 'files': rows,
               'label_contents_verified_from_kaggle_payload': False,
               'dataset_media_bytes_downloaded': 0, 'license_authorization_established': False}
    atomic_json(OUT / 'catalog.json', catalog)
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=['name', 'size', 'creationDate'])
    writer.writeheader()
    writer.writerows(rows)
    csv_payload = buffer.getvalue().encode('utf-8')
    check_budget(len(csv_payload))
    (OUT / 'all_files.csv').write_bytes(csv_payload)
    print(json.dumps({key: value for key, value in catalog.items() if key != 'files'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
