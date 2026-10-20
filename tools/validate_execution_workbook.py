"""Independently reconcile saved predictions, workbook cells and checkpoints."""
import csv
import hashlib
import json
import math
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/execution_20261019'
WORK=ROOT.parent/'research/working/execution_20261019'
BOOK=ROOT.parent/'research/workbooks/DFDC_Transfer_Learning_Experiments.xlsx'
NS={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_close(actual,expected):
    assert math.isclose(actual,expected,rel_tol=1e-9,abs_tol=1e-10),(actual,expected)


def independent_metrics(rows):
    pairs=[(float(row['fake_probability']),int(row['label']=='FAKE')) for row in rows]
    assert all(math.isfinite(p) and 0<=p<=1 for p,y in pairs)
    counts=Counter((y,int(p>=.5)) for p,y in pairs)
    tn,fp,fn,tp=[counts[k] for k in [(0,0),(0,1),(1,0),(1,1)]]
    precision=tp/(tp+fp) if tp+fp else 0;recall=tp/(tp+fn) if tp+fn else 0
    by_score={}
    for p,y in pairs: by_score.setdefault(p,Counter())[y]+=1
    negatives_before=0;wins=0.
    for p,labels in sorted(by_score.items()):
        wins+=labels[1]*(negatives_before+.5*labels[0]);negatives_before+=labels[0]
    return {'accuracy':(tn+tp)/len(rows),'balanced_accuracy':(.5*(tp/(tp+fn)+tn/(tn+fp))),
            'precision':precision,'recall':recall,'f1':2*precision*recall/(precision+recall) if precision+recall else 0,
            'roc_auc':wins/((tp+fn)*(tn+fp)),'confusion_matrix_real_fake':[[tn,fp],[fn,tp]]}


def main():
    data=json.loads((RUN/'comparison.json').read_text(encoding='utf-8'))
    rows=sorted(data['experiments'],key=lambda r:data['champion']['candidate_ids'].index(r['experiment_id']))
    expected_names=['Experiment Leaderboard','Training and Validation','Attention and Augmentation','Hardware and Cost','Held-out Test Results','Champion and Limitations']
    cells={};errors=[]
    with zipfile.ZipFile(BOOK) as archive:
        shared=[]
        if 'xl/sharedStrings.xml' in archive.namelist():
            shared=[''.join(si.itertext()) for si in ET.fromstring(archive.read('xl/sharedStrings.xml'))]
        relations={r.attrib['Id']:r.attrib['Target'] for r in ET.fromstring(archive.read('xl/_rels/workbook.xml.rels'))}
        book=ET.fromstring(archive.read('xl/workbook.xml'))
        names=[]
        for sheet in book.find('m:sheets',NS):
            name=sheet.attrib['name'];names.append(name)
            target=relations[sheet.attrib['{'+NS['r']+'}id']]
            filename=target.lstrip('/') if target.startswith('/') else 'xl/'+target
            values={}
            for cell in ET.fromstring(archive.read(filename)).findall('.//m:sheetData/m:row/m:c',NS):
                typ=cell.get('t');v=cell.find('m:v',NS);formula=cell.find('m:f',NS)
                if typ=='e': errors.append((name,cell.get('r'),v.text))
                if typ=='inlineStr': value=''.join(cell.find('m:is',NS).itertext())
                elif v is None: value=None
                elif typ=='s': value=shared[int(v.text)]
                elif typ in {'str','b'}: value=v.text
                else: value=float(v.text) if v.text is not None else None
                values[cell.get('r')]={'value':value,'formula':formula.text if formula is not None else None}
            cells[name]=values
        assert names==expected_names,names
        assert not errors,errors
        chart_files=[n for n in archive.namelist() if '/charts/' in n and Path(n).name.startswith('chart') and n.endswith('.xml')]
        assert len(chart_files)==4,chart_files
        chart_ns={'c':'http://schemas.openxmlformats.org/drawingml/2006/chart','a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
        chart_colors=set()
        for chart_file in chart_files:
            tree=ET.fromstring(archive.read(chart_file))
            chart_colors.update(c.get('val','').upper() for c in tree.findall('.//a:srgbClr',chart_ns))
            if tree.find('.//c:barChart',chart_ns) is not None:
                for minimum in tree.findall('.//c:valAx/c:scaling/c:min',chart_ns): assert float(minimum.get('val'))==0
                assert tree.find('.//c:valAx/c:scaling/c:min',chart_ns) is not None,'Bar axis does not include zero'
            for series in tree.findall('.//c:lineChart/c:ser',chart_ns):
                assert series.find('c:spPr/a:ln/a:solidFill/a:srgbClr',chart_ns) is not None,'Export lost a line-series color'
        assert {'355C86','D1813C','54928F','8769A3'}.issubset(chart_colors),chart_colors
        assert any(n.startswith('xl/media/') for n in archive.namelist()),'Missing confusion matrix image'
    heldout=None
    for index,row in enumerate(rows,7):
        folder=RUN/('E00_validation' if row['experiment_id']=='E00' else row['experiment_id'])
        with (folder/'validation_predictions.csv').open(newline='',encoding='utf-8') as stream: predictions=list(csv.DictReader(stream))
        paths={r['path'] for r in predictions}
        assert len(predictions)==2000 and len(paths)==2000
        if heldout is None: heldout=paths
        assert paths==heldout,'Validation observations differ between experiments'
        independently=independent_metrics(predictions)
        for metric in ['accuracy','balanced_accuracy','precision','recall','f1','roc_auc']:
            assert_close(independently[metric],row['validation'][metric])
        assert independently['confusion_matrix_real_fake']==row['validation']['confusion_matrix_real_fake']
        first=7+(index-7)*7
        for address,expected in [(f'M{first+1}',independently['confusion_matrix_real_fake'][0][0]),(f'N{first+1}',independently['confusion_matrix_real_fake'][0][1]),
                                 (f'M{first+2}',independently['confusion_matrix_real_fake'][1][0]),(f'N{first+2}',independently['confusion_matrix_real_fake'][1][1])]:
            assert cells[expected_names[1]][address]['value']==expected
        for column,metric in [('G','accuracy'),('H','balanced_accuracy'),('I','precision'),('J','recall'),('K','f1'),('L','roc_auc')]:
            assert_close(cells[expected_names[0]][f'{column}{index}']['value'],row['validation'][metric])
        assert digest(row['checkpoint'])==row['checkpoint_sha256']
        assert Path(row['checkpoint']).stat().st_size==row['checkpoint_bytes']
    for index,row in enumerate([r for r in rows if r.get('config',{}).get('reference_experiment')],7):
        reference=next(r for r in data['experiments'] if r['experiment_id']==row['config']['reference_experiment'])
        for column,expected in [('D',100*(row['validation']['accuracy']-reference['validation']['accuracy'])),
                                ('E',row['validation']['roc_auc']-reference['validation']['roc_auc']),
                                ('F',row['validation']['f1']-reference['validation']['f1'])]:
            cell=cells[expected_names[2]][f'{column}{index}'];assert cell['formula'];assert_close(cell['value'],expected)
    with (RUN/'E00_test/predictions.csv').open(newline='',encoding='utf-8') as stream:
        predictions=[{'fake_probability':r['fake_probability'],'label':'FAKE' if int(r['target_index'])==0 else 'REAL'} for r in csv.DictReader(stream)]
    measured=independent_metrics(predictions);record=json.loads((RUN/'E00_test/metrics.json').read_text())
    assert len(predictions)==20000
    for metric in ['accuracy','balanced_accuracy','precision','recall','roc_auc']: assert_close(measured[metric],record[metric])
    assert_close(measured['f1'],record['f1_score'])
    assert measured['confusion_matrix_real_fake']==record['confusion_matrix']
    legacy=json.loads((RUN/'demo_operational/verification.json').read_text())['checkpoint_sha256']
    assert digest(ROOT/'models/best_resnet18.pth')==legacy,'Original checkpoint changed'
    assert (WORK/'DFDC_Transfer_Learning_Experiments_before_execution.xlsx').is_file(),'Original workbook backup missing'
    result={'passed':True,'experiments_reconciled':len(data['experiments']),'common_validation_images_per_experiment':len(heldout),
            'test_images_reconciled':len(predictions),'sheets':names,'native_charts':len(chart_files),'formula_errors':errors,
            'independent_auc_method':'Pairwise positive-vs-negative score ordering with half credit for ties',
            'legacy_checkpoint_unchanged':True,'original_workbook_backup_preserved':(WORK/'DFDC_Transfer_Learning_Experiments_before_execution.xlsx').is_file()}
    (WORK/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
