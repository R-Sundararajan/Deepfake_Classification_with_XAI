"""Targeted bibliographic lookup for the nine existing candidates only."""
import concurrent.futures
import json
import urllib.request
import urllib.parse
from datetime import datetime,timezone
from pathlib import Path

OUT=Path(__file__).resolve().parents[1]/'../research/working/expansion'
DOIS={
 'pdf_059':'10.1016/j.imavis.2025.105696',
 'pdf_074':'10.1109/ICSCN67106.2025.11308255',
 'pdf_075':'10.1109/IJCB62174.2024.10744484',
 'pdf_078':'10.1109/IITCEE67948.2026.11394363',
 'pdf_090':'10.1109/TBIOM.2024.3407650',
 'pdf_093':'10.1109/ICC-ROBINS60238.2024.10533980',
 'pdf_103':'10.1109/ACCESS.2025.3649128',
 'pdf_104':'10.1109/I-SMAC65734.2025.11597656',
}


def fetch(key,doi):
    result={'key':key,'doi':doi,'retrieved_utc':datetime.now(timezone.utc).isoformat()}
    for label,url in [('crossref','https://api.crossref.org/works/'+urllib.parse.quote(doi,safe='')),
                      ('openalex','https://api.openalex.org/works/https://doi.org/'+doi)]:
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'Research-Literature-Verification/1.0','Accept':'application/json'})
            with urllib.request.urlopen(req,timeout=35) as r:data=json.loads(r.read(4*1024*1024))
            result[label]=data.get('message',data) if label=='crossref' else data
        except Exception as e:result[label+'_error']=f'{type(e).__name__}: {e}'
    return result


if __name__=='__main__':
    results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for r in pool.map(lambda item:fetch(*item),DOIS.items()):
            results.append(r)
            c=r.get('crossref',{});o=r.get('openalex',{})
            print(json.dumps({'key':r['key'],'title':c.get('title'),'venue':c.get('container-title'),'type':c.get('type'),
                              'published':c.get('published'),'crossmark_updates':c.get('update-to',[]),
                              'oa':o.get('open_access'),'best_oa':o.get('best_oa_location'),
                              'errors':{k:v for k,v in r.items() if k.endswith('_error')}},ensure_ascii=False),flush=True)
    (OUT/'bibliographic_records.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
