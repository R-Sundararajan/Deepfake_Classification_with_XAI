"""Create a pinned, source-linked small-sample proposal; no video download."""
import hashlib
import json
import urllib.parse
import urllib.request
from collections import Counter

from tools.audit_dfdc_mirrors import get_json
from tools.dfdc_storage import ROOT, RESEARCH, atomic_json
from tools.probe_dfdc_payloads import head
from tools.runtime import configure_runtime

REPO='gonnerthetooner/DFDC-extracted-full'
REVISION='a8a094fcdf8f10bd03971691d9b0b46445d95692'
FOLDER='dfdc_train_part_00/dfdc_train_part_0'


def list_all(url):
    rows=[];pages=0
    while url:
        if not url.startswith('https://huggingface.co/api/'):raise ValueError('Unexpected pagination host')
        request=urllib.request.Request(url,headers={'User-Agent':'DFDC-source-audit'})
        with urllib.request.urlopen(request,timeout=30) as response:
            data=json.loads(response.read(4*1024**2));link=response.headers.get('Link','')
        rows.extend(data);pages+=1
        if pages>10:raise RuntimeError('Part listing bound exceeded')
        url=next((chunk.split('>')[0].strip().lstrip('<') for chunk in link.split(',') if 'rel="next"' in chunk),'')
    return rows


def main():
    configure_runtime()
    rows=list_all(f'https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/{FOLDER}?limit=1000')
    atomic_json(RESEARCH/'analyses/mirror_candidates/hf_part00_complete_tree.json',rows)
    files={row['path'].split('/')[-1]:row for row in rows if row['type']=='file'}
    metadata=json.loads((RESEARCH/'analyses/mirror_candidates/hf_part00_metadata.json').read_text(encoding='utf-8'))
    assert set(metadata)=={name for name in files if name.endswith('.mp4')}
    pairs=[]
    for fake,info in metadata.items():
        original=info.get('original')
        if info['label']=='FAKE' and original in metadata and metadata[original]['label']=='REAL':
            pairs.append((files[fake]['size']+files[original]['size'],original,fake))
    chosen=[];sources=set()
    for size,original,fake in sorted(pairs):
        if original in sources:continue
        sources.add(original);chosen.extend([original,fake])
        if len(sources)==10:break
    selected=[]
    for name in chosen:
        row=files[name]
        selected.append({'filename':name,'path':row['path'],'bytes':row['size'],'sha256':row['lfs']['oid'],
                         'label':metadata[name]['label'],'original':metadata[name].get('original'),
                         'url':f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{row["path"]}'})
    checks=[{'filename':row['filename'],**head(row['url'])} for row in selected[:2]]
    plan={'dataset_family':'DFDC full training corpus, part 0 subset; NOT DFDC Preview or the complete corpus',
          'host':'Hugging Face public third-party mirror','repository':REPO,'revision':REVISION,
          'upstream':'https://ai.meta.com/datasets/dfdc/','part_videos':len(metadata),'part_labels':dict(Counter(v['label'] for v in metadata.values())),
          'part_video_bytes':sum(v['size'] for name,v in files.items() if name.endswith('.mp4')),
          'repository_parts_observed':list(range(10)),'license_status':'No README or data license published by this mirror. Upstream usage/redistribution permission is not verified.',
          'media_download_authorized':False,'selection':'10 smallest complete REAL/FAKE source pairs by combined size; intended for pipeline smoke testing, not an unbiased benchmark',
          'videos':len(selected),'source_groups':len(sources),'labels':dict(Counter(v['label'] for v in selected)),'selected_video_bytes':sum(v['bytes'] for v in selected),
          'metadata_bytes':files['metadata.json']['size'],'metadata_url':f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{FOLDER}/metadata.json',
          'files':selected,'http_head_checks':checks,'approval_needed':'Confirm this source and exact size, and provide/confirm permission under the upstream DFDC terms before media transfer.'}
    atomic_json(RESEARCH/'analyses/dfdc_mirror_sample_plan.json',plan)
    print(json.dumps({k:v for k,v in plan.items() if k!='files'},indent=2))


if __name__=='__main__':main()
