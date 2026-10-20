"""Acquire only the official labeled Kaggle DFDC training sample, directly.

No full-competition or test-data download route is implemented. Credentials are
read only from the process environment; never serialized or printed.
"""
import argparse
import base64
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath

from tools.dfdc_storage import ROOT, atomic_json, check_budget, owned_path, tree_bytes

COMPETITION = 'deepfake-detection-challenge'
BASE = 'https://www.kaggle.com/api/v1/competitions/data'
PREFIX = 'train_sample_videos/'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def permitted_url(url):
    u = urllib.parse.urlparse(url)
    host = (u.hostname or '').lower()
    return u.scheme == 'https' and (host in {'www.kaggle.com','kaggle.com','storage.googleapis.com'} or host.endswith('.googleapis.com') or host.endswith('.kaggle.com'))


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not permitted_url(newurl):
            raise RuntimeError('Refused download redirect outside official HTTPS storage')
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if urllib.parse.urlparse(req.full_url).netloc != urllib.parse.urlparse(newurl).netloc:
            new.remove_header('Authorization')
        return new


def auth_header():
    token = os.environ.get('KAGGLE_API_TOKEN')
    if token:
        return 'Bearer ' + token
    user, key = os.environ.get('KAGGLE_USERNAME'), os.environ.get('KAGGLE_KEY')
    if user and key:
        return 'Basic ' + base64.b64encode(f'{user}:{key}'.encode()).decode()
    raise RuntimeError('Kaggle authentication missing. Set KAGGLE_API_TOKEN in your local environment; do not paste credentials into chat.')


def open_official(url):
    if not permitted_url(url):
        raise ValueError('Nonofficial URL refused')
    req = urllib.request.Request(url, headers={'Authorization': auth_header(), 'User-Agent':'DFDC-Research-Baseline/1.0'})
    return urllib.request.build_opener(SafeRedirect()).open(req, timeout=90)


def list_sample():
    selected = {}
    token = ''
    seen = set()
    for _ in range(200):
        query = urllib.parse.urlencode({'pageSize':1000, **({'pageToken':token} if token else {})})
        with open_official(f'{BASE}/list/{COMPETITION}?{query}') as f:
            # The listing is metadata, but still bounded in memory.
            raw = f.read(16*1024*1024 + 1)
            if len(raw)>16*1024*1024:
                raise RuntimeError('File-list response exceeded metadata bound')
            result = json.loads(raw)
        for item in result['files']:
            name = item['name']
            if name.startswith(PREFIX):
                leaf = name[len(PREFIX):]
                if PurePosixPath(leaf).name != leaf or '\\' in leaf or leaf in {'','..','.'}:
                    raise ValueError('Unsafe official-list filename')
                if not (leaf.endswith('.mp4') or leaf == 'metadata.json'):
                    continue
                size = int(item['totalBytes'])
                if size <= 0:
                    raise ValueError('Missing positive official file size')
                selected[name] = {'name':name, 'bytes':size}
        token = result.get('nextPageToken') or ''
        if not token:
            break
        if token in seen:
            raise RuntimeError('Kaggle file pagination repeated; sample completeness cannot be established')
        seen.add(token)
    else:
        raise RuntimeError('File-list page bound exceeded; no media downloaded')
    if PREFIX+'metadata.json' not in selected:
        raise RuntimeError('Official expanded training sample/metadata not listed. No unofficial or full-dataset fallback attempted.')
    return sorted(selected.values(), key=lambda x:(not x['name'].endswith('metadata.json'), x['name']))


def download(item, destination, known):
    name, expected = item['name'],item['bytes']
    p = owned_path(destination / name[len(PREFIX):])
    if p.exists():
        digest = sha256(p)
        if p.stat().st_size == expected and known.get(name,{}).get('sha256')==digest:
            return known[name]
        raise FileExistsError(f'Existing unverified media preserved: {p.name}')
    part = p.with_suffix(p.suffix+'.part')
    if part.exists():
        raise FileExistsError(f'Unfinished download preserved: {part.name}; inspect it before retrying')
    check_budget(expected + 1024*1024)
    p.parent.mkdir(parents=True,exist_ok=True)
    url=f'{BASE}/download/{COMPETITION}/{urllib.parse.quote(name,safe="")}'
    # Permission check occurs before any local media file is opened.
    with open_official(url) as response:
        ct=response.headers.get('Content-Type','').lower()
        if 'text/html' in ct:
            raise RuntimeError('Kaggle returned an HTML login/rules page, not the requested file')
        length=response.headers.get('Content-Length')
        if length and int(length)!=expected:
            raise RuntimeError('Remote content length differs from official listing; no bytes written')
        count=0;h=hashlib.sha256()
        with part.open('xb') as out:
            while True:
                block=response.read(min(1024*1024,expected-count+1))
                if not block:break
                if count+len(block)>expected:
                    raise RuntimeError('Download exceeded declared file size; partial file retained')
                check_budget(len(block)+4096)
                out.write(block);h.update(block);count+=len(block)
        if count!=expected:
            raise RuntimeError('Truncated download; partial file retained')
    if p.suffix=='.json':
        data=json.loads(part.read_text(encoding='utf-8'))
        if not isinstance(data,dict) or not data or any(v.get('label') not in {'REAL','FAKE'} for v in data.values()):
            raise RuntimeError('Official metadata validation failed; partial file retained')
    part.replace(p)
    return {'name':name,'bytes':count,'sha256':h.hexdigest(),'source_url':url}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download',action='store_true',help='Without this flag, only list/estimate; no media transfer.')
    args=parser.parse_args()
    dest=ROOT/'data/dfdc/train_sample_videos'
    provenance=ROOT/'data/dfdc/provenance.json'
    start=time.perf_counter()
    initial_storage=check_budget()
    existing_project_bytes=tree_bytes(ROOT)
    items=list_sample()
    total=sum(i['bytes'] for i in items)
    # Direct media transfer avoids a simultaneously retained ZIP/extracted copy.
    budget=check_budget(total + 256*1024*1024)
    plan={'source':'official_kaggle_dfdc','competition':COMPETITION,'collection':PREFIX,
          'strategy':'Individual original competition files; no archive/extraction duplication',
          'selected_files':len(items),'selected_bytes':total,'files':items,'storage_before':budget,
          'existing_project_bytes':existing_project_bytes,'initial_storage':initial_storage}
    atomic_json(ROOT/'results/dfdc_acquisition_plan.json',plan)
    print(json.dumps({k:plan[k] for k in ['selected_files','selected_bytes','strategy']}),flush=True)
    if not args.download:return
    known={}
    if provenance.exists():known={v['name']:v for v in json.loads(provenance.read_text(encoding='utf-8'))['files']}
    for n,item in enumerate(items,1):
        known[item['name']]=download(item,dest,known)
        atomic_json(provenance,{'source':'official_kaggle_dfdc','competition':COMPETITION,'collection':PREFIX,'files':list(known.values()),'complete':n==len(items)})
        if n==1 or n%25==0 or n==len(items):print(f'Verified {n}/{len(items)} files; phase {check_budget()["phase_bytes"]} bytes',flush=True)
    print(json.dumps({'completed':True,'elapsed_seconds':time.perf_counter()-start,'storage_after':check_budget()}),flush=True)
    atomic_json(ROOT/'results/dfdc_access_status.json',{'status':'download_complete','selected_files':len(items),'selected_bytes':total})


if __name__=='__main__':
    try:main()
    except urllib.error.HTTPError as e:
        # Do not print response URLs, bodies or signed redirect credentials.
        if e.code in {401,403}:
            print(f'Kaggle HTTP {e.code}: authenticate locally and accept applicable competition rules at https://www.kaggle.com/competitions/{COMPETITION}/rules',file=sys.stderr)
        else:print(f'Official Kaggle request failed: HTTP {e.code}',file=sys.stderr)
        atomic_json(ROOT/'results/dfdc_access_status.json',{'status':'access_blocked' if e.code in {401,403} else 'http_error','http_status':e.code,
                    'action_url':f'https://www.kaggle.com/competitions/{COMPETITION}/rules','credentials_logged':False})
        sys.exit(2)
    except urllib.error.URLError:
        print('Official Kaggle connection failed (network/DNS/timeout); no credentials logged.',file=sys.stderr);sys.exit(2)
