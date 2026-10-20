"""Re-read nine original local PDFs for independent quality/access decisions."""
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]
WORK=ROOT/'../research/working'
KEYS=['pdf_059','pdf_074','pdf_075','pdf_077','pdf_078','pdf_090','pdf_093','pdf_103','pdf_104']
inventory=json.loads((WORK/'paper_inventory.json').read_text(encoding='utf-8'))
results=[]
for key in KEYS:
    source=next(x for x in inventory if x['key']==key)
    archive,member=source['source'].split('!',1)
    with zipfile.ZipFile(archive) as z:raw=z.read(member)
    assert raw.startswith(b'%PDF-')
    digest=hashlib.sha256(raw).hexdigest();assert digest==source['sha256']
    reader=PdfReader(io.BytesIO(raw),strict=True)
    pages=[page.extract_text() or '' for page in reader.pages]
    assert len(pages)==source['pages'] and sum(len(t)>40 for t in pages)>len(pages)*.8
    text='\n'.join(pages)
    record={'key':key,'source':source['source'],'sha256':digest,'bytes':len(raw),'pages':len(pages),
            'all_pages_parsed':True,'sparse_text_pages':[i+1 for i,t in enumerate(pages) if len(t)<40],
            'methodology_pages':[i+1 for i,t in enumerate(pages) if re.search(r'methodology|proposed.{0,20}(?:method|framework)|architecture',t,re.I)],
            'dfdc_pages':[i+1 for i,t in enumerate(pages) if re.search(r'DFDC|Deepfake Detection Challenge',t,re.I)],
            'results_pages':[i+1 for i,t in enumerate(pages) if re.search(r'results|evaluation',t,re.I)],
            'first_page':pages[0], 'text_pages':pages}
    results.append(record)
    print(json.dumps({k:v for k,v in record.items() if k not in {'first_page','text_pages'}},ensure_ascii=False))
(WORK/'expansion/candidate_pdf_inspections.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
