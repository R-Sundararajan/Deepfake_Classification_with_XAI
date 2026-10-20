"""Bounded acquisition through the configured dfenv Kaggle CLI, official DFDC only."""
import argparse
import csv
import io
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

from tools.acquire_dfdc import COMPETITION, PREFIX, sha256
from tools.dfdc_storage import ROOT, atomic_json, check_budget
from tools.runtime import configure_runtime


def cli(*args, timeout=120):
    executable=Path(sys.executable).parent/'kaggle.exe'
    if not executable.exists():
        raise RuntimeError('Run this module with dfenv/Scripts/python.exe; Kaggle CLI is missing')
    result=subprocess.run([str(executable),*args],capture_output=True,text=True,timeout=timeout)
    if result.returncode:
        output=result.stdout+'\n'+result.stderr
        match=re.search(r'\b(401|403|404|429)\b',output)
        status={'completed':False,'stage':'official_kaggle_cli','http_status':int(match[1]) if match else None,
                'blocker':'Official Kaggle download/list permission denied' if match and match[1] in {'401','403'} else 'Kaggle CLI request failed',
                'action_url':f'https://www.kaggle.com/competitions/{COMPETITION}/rules','credentials_logged':False}
        atomic_json(ROOT/'results/dfdc_access_status.json',status)
        raise RuntimeError(f"{status['blocker']}; HTTP {status['http_status']}. See results/dfdc_access_status.json")
    return result.stdout+'\n'+result.stderr


def parse_listing(text):
    token=re.search(r'Next Page Token\s*=\s*(\S+)',text)
    lines=text.splitlines()
    start=next((i for i,s in enumerate(lines) if s.startswith('name,size,')),None)
    if start is None:
        raise ValueError('Unrecognized Kaggle CSV listing; no download attempted')
    rows=[]
    for row in csv.DictReader(io.StringIO('\n'.join(s for s in lines[start:] if s and not s.startswith('Next Page Token')))):
        if row.get('name') and row.get('size','').isdigit():
            rows.append({'name':row['name'],'bytes':int(row['size'])})
    return rows,token[1] if token else ''


def list_sample():
    selected={}; token=''; seen=set()
    for _ in range(16):
        args=['competitions','files','-c',COMPETITION,'--page-size','1000','-v']
        if token: args+=['--page-token',token]
        rows,token=parse_listing(cli(*args))
        for row in rows:
            if row['name']=='train_sample_videos.zip' or row['name'].startswith(PREFIX):
                leaf=row['name'][len(PREFIX):] if row['name'].startswith(PREFIX) else row['name']
                if PurePosixPath(leaf).name!=leaf or '\\' in leaf or '..' in leaf or row['bytes']<=0:
                    raise ValueError('Unsafe/missing-size sample entry')
                if leaf.endswith('.mp4') or leaf in {'metadata.json','train_sample_videos.zip'}:
                    selected[row['name']]=row
        if not token: break
        if token in seen: raise RuntimeError('Repeated Kaggle page token')
        seen.add(token)
    else: raise RuntimeError('Kaggle listing exceeded page bound')
    if 'train_sample_videos.zip' in selected:
        return [selected['train_sample_videos.zip']]
    if PREFIX+'metadata.json' not in selected:
        raise RuntimeError('The official labeled sample is not listed. No alternative dataset requested.')
    return sorted(selected.values(),key=lambda v:(not v['name'].endswith('metadata.json'),v['name']))


def acquire(download=False):
    configure_runtime()
    files=list_sample(); total=sum(v['bytes'] for v in files)
    archive=files[0]['name'].endswith('.zip')
    check_budget(total*(4 if archive else 1)+256*1024**2)
    plan={'source':'official_kaggle_dfdc','competition':COMPETITION,'cli':cli('--version').strip(),'selected_files':len(files),'selected_bytes':total,'files':files,
          'strategy':'Official sample ZIP' if archive else 'Kaggle now lists expanded sample files; direct official files avoid ZIP duplication', 'storage':check_budget()}
    atomic_json(ROOT/'results/dfdc_acquisition_plan.json',plan)
    print(json.dumps({k:plan[k] for k in ['selected_files','selected_bytes','strategy']}),flush=True)
    if not download: return plan
    dest=ROOT/'data/dfdc/train_sample_videos'; dest.mkdir(parents=True,exist_ok=True)
    provenance=ROOT/'data/dfdc/provenance.json'
    known={v['name']:v for v in json.loads(provenance.read_text(encoding='utf-8'))['files']} if provenance.exists() else {}
    if archive:
        raise RuntimeError('ZIP listed: inspect its central-directory extraction sizes before enabling archive acquisition. No archive downloaded.')
    for index,item in enumerate(files,1):
        path=dest/Path(item['name']).name
        if path.exists():
            if path.stat().st_size!=item['bytes'] or known.get(item['name'],{}).get('sha256')!=sha256(path):
                raise FileExistsError('Unverified existing sample file preserved: '+path.name)
        else:
            check_budget(item['bytes']+1024**2)
            cli('competitions','download','-c',COMPETITION,'-f',item['name'],'-p',str(dest),'-q',timeout=600)
            if not path.exists() or path.stat().st_size!=item['bytes']:
                raise RuntimeError('Downloaded size differs from official listing; file preserved for inspection')
            known[item['name']]={**item,'sha256':sha256(path),'source_url':'https://www.kaggle.com/competitions/'+COMPETITION+'/data?select='+item['name']}
        atomic_json(provenance,{'source':'official_kaggle_dfdc','competition':COMPETITION,'collection':PREFIX,'files':list(known.values()),'complete':index==len(files)})
        if index==1 or index%25==0 or index==len(files): print(f'Verified {index}/{len(files)} sample files',flush=True)
    atomic_json(ROOT/'results/dfdc_access_status.json',{'completed':True,'status':'download_complete','selected_bytes':total,'selected_files':len(files)})
    return plan


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download',action='store_true')
    try:
        acquire(parser.parse_args().download)
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as error:
        print(str(error),file=sys.stderr); sys.exit(2)
