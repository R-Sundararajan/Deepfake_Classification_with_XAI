"""Read public source documentation/APIs and HEAD headers; never read media bodies."""
import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from tools.dfdc_storage import RESEARCH, atomic_json, check_budget
from tools.probe_dfdc_payloads import head
from tools.runtime import configure_runtime


def document(url):
    if urllib.parse.urlparse(url).path.lower().endswith(('.mp4', '.zip', '.gz', '.tar', '.csv')):
        raise ValueError('Dataset payload GET is prohibited in this documentation audit')
    request = urllib.request.Request(url, headers={'User-Agent': 'Dataset-license-documentation-audit'})
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            raw = response.read(4 * 1024**2 + 1)
            if len(raw) > 4 * 1024**2:
                raise ValueError('Documentation response too large')
            value = raw.decode('utf-8', errors='replace')
            return {'url': url, 'status': response.status, 'bytes': len(raw),
                    'sha256': hashlib.sha256(raw).hexdigest(), 'text': value}
    except urllib.error.HTTPError as error:
        return {'url': url, 'status': error.code}
    except (OSError, ValueError) as error:
        return {'url': url, 'error': type(error).__name__}


def metadata(url):
    value = document(url)
    if 'text' in value:
        try:
            value['data'] = json.loads(value.pop('text'))
        except ValueError:
            value['error'] = 'Not JSON'
    return value


def main():
    configure_runtime()
    check_budget(24 * 1024**2)
    jobs = {
        'hidf_record': 'https://zenodo.org/api/records/16140829',
        'timit_record': 'https://zenodo.org/api/records/4068245',
        'hidf_repo': 'https://api.github.com/repos/DSAIL-SKKU/HiDF',
        'hidf_tree': 'https://api.github.com/repos/DSAIL-SKKU/HiDF/git/trees/master?recursive=1',
        'lavdf_repo': 'https://api.github.com/repos/ControlNet/LAV-DF',
        'lavdf_hf': 'https://huggingface.co/api/datasets/ControlNet/LAV-DF',
        'lavdf_files': 'https://huggingface.co/api/datasets/ControlNet/LAV-DF/tree/main?limit=100',
        'mirror_info': 'https://huggingface.co/api/datasets/gonnerthetooner/DFDC-extracted-full',
        'mirror_tree': 'https://huggingface.co/api/datasets/gonnerthetooner/DFDC-extracted-full/tree/main?limit=100',
        'mirror_discussions': 'https://huggingface.co/api/datasets/gonnerthetooner/DFDC-extracted-full/discussions',
        'mirror_commits': 'https://huggingface.co/api/datasets/gonnerthetooner/DFDC-extracted-full/commits/main',
    }
    with ThreadPoolExecutor(max_workers=3) as executor:
        values = dict(zip(jobs, executor.map(metadata, jobs.values())))
    doc_jobs = {
        'hidf_readme': 'https://raw.githubusercontent.com/DSAIL-SKKU/HiDF/master/README.md',
        'hidf_license': 'https://raw.githubusercontent.com/DSAIL-SKKU/HiDF/master/LICENSE',
        'lavdf_readme': 'https://raw.githubusercontent.com/ControlNet/LAV-DF/master/README.md',
        'lavdf_terms': 'https://raw.githubusercontent.com/ControlNet/LAV-DF/master/TERMS_AND_CONDITIONS.md',
        'lavdf_license': 'https://raw.githubusercontent.com/ControlNet/LAV-DF/master/LICENSE',
        'dfdc_portal': 'https://dfdc.ai/',
        'dfdc_login': 'https://dfdc.ai/login',
        'dfdc_reset': 'https://dfdc.ai/reset-password',
        'dfdc_mirror_readme': 'https://huggingface.co/datasets/gonnerthetooner/DFDC-extracted-full/raw/main/README.md',
        'dfdc_mirror_license': 'https://huggingface.co/datasets/gonnerthetooner/DFDC-extracted-full/raw/main/LICENSE',
    }
    with ThreadPoolExecutor(max_workers=3) as executor:
        values.update(zip(doc_jobs, executor.map(document, doc_jobs.values())))
    # The author-linked archive URLs are inspected strictly with HEAD. No ranges,
    # archive contents, CSV rows, authentication, signups or agreement acceptance.
    for name in ('hidf_record', 'timit_record'):
        data = values[name].get('data', {})
        for entry in data.get('files', []):
            if entry['key'].endswith(('.zip', '.gz', '.csv')):
                if name == 'hidf_record' and entry['key'] not in ('Fake-vid.zip', 'Real-vid.zip', 'metadata.csv'):
                    continue
                url = entry.get('links', {}).get('self')
                if url:
                    entry['head_audit'] = head(url)
    for name in ('dfdc_portal', 'dfdc_login', 'dfdc_reset'):
        value = values[name]
        value['public_links'] = re.findall(r'(?:href|src)=[\"\']([^\"\']+)', value.get('text', ''))
    result = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Public documentation, API records, directory lists and HEAD only. No dataset payload bodies.',
              'user_download_authorization': False, 'dataset_media_bytes_downloaded': 0,
              'results': values}
    atomic_json(RESEARCH / 'analyses/licensed_dataset_source_audit.json', result)
    summary = {}
    for name, value in values.items():
        item = {k: value.get(k) for k in ('status', 'bytes', 'error') if k in value}
        data = value.get('data', {})
        if name.endswith('record') and isinstance(data, dict):
            item.update({'title': data.get('metadata', {}).get('title'),
                         'license': data.get('metadata', {}).get('license'),
                         'access_right': data.get('metadata', {}).get('access_right'),
                         'files': [{k: entry.get(k) for k in ('key', 'size', 'checksum', 'head_audit')}
                                   for entry in data.get('files', [])]})
        elif name.endswith(('files', 'tree')) and isinstance(data, list):
            item['files'] = data
        elif name.endswith('_hf') or name == 'mirror_info':
            item.update({k: data.get(k) for k in ('id', 'sha', 'private', 'gated', 'cardData')})
        elif name == 'hidf_tree':
            item['revision'] = data.get('sha')
            item['mp4_count'] = sum(row['path'].endswith('.mp4') for row in data.get('tree', []))
        if 'public_links' in value:
            item['public_links'] = value['public_links']
        summary[name] = item
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
