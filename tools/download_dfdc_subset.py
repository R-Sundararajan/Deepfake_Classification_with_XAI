"""Resumable pinned DFDC subset acquisition, gated by source/size permission."""
import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from tools.acquire_dfdc import sha256
from tools.dfdc_storage import ROOT, RESEARCH, atomic_json, check_budget, owned_path
from tools.runtime import configure_runtime
from training.dfdc_data import load_metadata, prepare, probe_video


def allowed_url(url):
    parsed=urllib.parse.urlparse(url);host=(parsed.hostname or '').lower()
    return parsed.scheme=='https' and not parsed.username and not parsed.password and (host=='huggingface.co' or host.endswith('.hf.co'))


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        if not allowed_url(newurl):raise ValueError('Refused redirect outside HTTPS Hugging Face storage')
        forwarded={k:v for k,v in req.headers.items() if k.lower() in {'range','if-range','user-agent'}}
        return urllib.request.Request(newurl,headers=forwarded,method=req.get_method())


def open_public(url,offset=0):
    if not allowed_url(url):raise ValueError('Unapproved source host')
    headers={'User-Agent':'DFDC-approved-subset/1.0'}
    if offset:headers['Range']=f'bytes={offset}-'
    return urllib.request.build_opener(SafeRedirect()).open(urllib.request.Request(url,headers=headers),timeout=45)


def transfer(file,destination,opener=open_public):
    """Keep partial bytes; append only to a verified Content-Range response."""
    path=owned_path(destination/file['filename']);expected=int(file['bytes']);digest=file['sha256']
    if path.exists():
        if path.stat().st_size==expected and sha256(path)==digest:return {'reused':True,'bytes':expected,'sha256':digest}
        raise FileExistsError('Existing unverified destination preserved: '+path.name)
    partial=path.with_suffix(path.suffix+'.part');offset=partial.stat().st_size if partial.exists() else 0
    if offset>expected:raise ValueError('Partial file exceeds approved size')
    if offset==expected:
        if sha256(partial)!=digest:raise ValueError('Completed partial has wrong checksum; preserved')
        partial.replace(path);return {'reused':True,'bytes':expected,'sha256':digest}
    check_budget(expected-offset+1024**2);path.parent.mkdir(parents=True,exist_ok=True)
    with opener(file['url'],offset) as response:
        if offset:
            match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
            if response.status!=206 or not match or tuple(map(int,match.groups()))!=(offset,expected-1,expected):
                raise RuntimeError('Server did not honor the approved resume range; partial preserved')
        elif response.status!=200:raise RuntimeError('Unexpected initial HTTP status')
        content_type=response.headers.get('Content-Type','').lower()
        if 'text/html' in content_type or 'application/json' in content_type:raise RuntimeError('Login/error payload refused')
        length=response.headers.get('Content-Length')
        if length and int(length)!=expected-offset:raise RuntimeError('Remote length differs from approved size')
        with partial.open('ab' if offset else 'xb') as stream:
            count=offset
            while True:
                block=response.read(min(1024**2,expected-count+1))
                if not block:break
                if count+len(block)>expected:raise RuntimeError('Payload exceeded approved size; partial preserved')
                check_budget(len(block)+4096);stream.write(block);count+=len(block)
        if count!=expected:raise RuntimeError('Transfer interrupted/truncated; rerun with the same approved plan to resume')
    if sha256(partial)!=digest:raise RuntimeError('SHA-256 mismatch; unverified partial preserved')
    partial.replace(path);return {'reused':False,'bytes':expected,'sha256':digest}


def validate_plan(plan):
    if plan.get('repository')!='gonnerthetooner/DFDC-extracted-full' or not re.fullmatch('[a-f0-9]{40}',plan.get('revision','')):
        raise ValueError('This downloader supports only the audited pinned DFDC mirror plan')
    files=plan['files'];names=set()
    for file in files:
        name=file['filename']
        if not re.fullmatch('[A-Za-z0-9_-]+\\.mp4',name) or name in names:raise ValueError('Unsafe/duplicate planned filename')
        names.add(name)
        if not allowed_url(file['url']) or '/resolve/'+plan['revision']+'/' not in file['url']:raise ValueError('Source is not pinned to the approved revision')
        if not re.fullmatch('[a-f0-9]{64}',file['sha256']) or not 0<int(file['bytes'])<=32*1024**2:raise ValueError('Invalid approved checksum/file size')
        if file['label'] not in {'REAL','FAKE'}:raise ValueError('Invalid planned class')
    total=sum(int(file['bytes']) for file in files)
    if total!=plan['selected_video_bytes'] or total>128*1024**2 or len(files)>32:raise ValueError('Approved small-sample bound exceeded')
    for file in files:
        if file['label']=='FAKE' and file.get('original') not in names:raise ValueError('FAKE lacks an included original source')
    return total


def acquire(plan_path,output,approved_hash=None,permission_note=None,download=False,prepare_manifest=False):
    configure_runtime();plan_path=Path(plan_path).resolve();plan=json.loads(plan_path.read_text(encoding='utf-8'));total=validate_plan(plan)
    identity=sha256(plan_path)
    print(json.dumps({'repository':plan['repository'],'revision':plan['revision'],'videos':len(plan['files']),'video_bytes':total,'mib':total/1024**2,
                      'plan_sha256':identity,'license_status':plan['license_status'],'download':download},indent=2),flush=True)
    if not download:return
    if approved_hash!=identity or not permission_note or len(permission_note.strip())<12:
        raise ValueError('Media download requires confirmation of this exact plan hash and a truthful upstream-permission reference. Public access alone does not establish redistribution rights.')
    output=owned_path(output);output.mkdir(parents=True,exist_ok=True);check_budget(total+16*1024**2)
    receipt={'dataset_family':plan['dataset_family'],'provenance':'third_party_dfdc_mirror','official_kaggle_download':False,'repository':plan['repository'],
             'revision':plan['revision'],'approved_plan_sha256':identity,'permission_reference':permission_note,'complete':False,'files':[]}
    atomic_json(output/'provenance.json',receipt)
    for index,file in enumerate(plan['files'],1):
        for attempt in range(3):
            try:result=transfer(file,output);break
            except (urllib.error.URLError,TimeoutError,ConnectionError):
                if attempt==2:raise
                time.sleep(attempt+1)
        receipt['files'].append({**file,**result});atomic_json(output/'provenance.json',receipt)
        print(f'Verified approved video {index}/{len(plan["files"])}: {file["filename"]}',flush=True)
    # Metadata was already inspected during source research; preserve its labels
    # and source relationships verbatim in a derived subset mapping.
    source=RESEARCH/'analyses/mirror_candidates/hf_part00_metadata.json'
    metadata=load_metadata(source);subset={f['filename']:metadata[f['filename']] for f in plan['files']}
    for file in plan['files']:
        info=subset[file['filename']]
        if info['label']!=file['label'] or info.get('original')!=file.get('original'):raise ValueError('Planned labels/source links differ from inspected original metadata')
        with (output/file['filename']).open('rb') as stream:
            if b'ftyp' not in stream.read(64):raise ValueError('Downloaded file is not an MP4 container')
        probe_video(output/file['filename'])
    atomic_json(output/'metadata.json',subset)
    receipt['complete']=True;receipt['source_metadata_sha256']=sha256(source);receipt['subset_metadata_sha256']=sha256(output/'metadata.json')
    receipt['processing_note']='Uses existing DFDC preparation without changing training code. Official-provenance flags remain false for a third-party mirror.'
    atomic_json(output/'provenance.json',receipt)
    if prepare_manifest:
        prepared=output.parent/(output.name+'_prepared_seed42')
        report=prepare(output/'metadata.json',prepared,42)
        assert not report['official_provenance_verified']
        print('Prepared source-safe manifest. Training trust policy still requires explicit approval for mirror data.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=RESEARCH/'analyses/dfdc_mirror_sample_plan.json')
    parser.add_argument('--output',type=Path,default=ROOT/'data/dfdc/mirror_sample20')
    parser.add_argument('--download',action='store_true');parser.add_argument('--approved-plan-sha256');parser.add_argument('--permission-reference')
    parser.add_argument('--prepare-manifest',action='store_true')
    args=parser.parse_args()
    try:acquire(args.plan,args.output,args.approved_plan_sha256,args.permission_reference,args.download,args.prepare_manifest)
    except (RuntimeError,ValueError,OSError) as error:print(str(error),file=sys.stderr);sys.exit(2)
