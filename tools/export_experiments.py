"""Build an auditable workbook input from completed official-data run predictions."""
import csv
import json
from pathlib import Path

from evaluation.dfdc import binary_metrics
from tools.acquire_dfdc import sha256
from tools.dfdc_storage import ROOT, atomic_json, check_budget
from training.experiment import read_config


def verify_predictions(path,expected,digest):
    if sha256(path)!=digest:raise ValueError('Prediction hash mismatch: '+str(path))
    with path.open(newline='',encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
    metrics=binary_metrics([int(r['label']=='FAKE') for r in rows],[float(r['fake_probability']) for r in rows])
    for key in ['accuracy','balanced_accuracy','precision','recall','f1','roc_auc','log_loss']:
        if metrics[key] is None or expected[key] is None:
            if metrics[key]!=expected[key]:raise ValueError('Undefined metric mismatch')
        elif abs(metrics[key]-expected[key])>1e-9:raise ValueError('Saved metrics differ from predictions: '+key)
    if metrics['confusion_matrix_real_fake']!=expected['confusion_matrix_real_fake']:raise ValueError('Confusion matrix mismatch')
    return rows


def export():
    records=[]
    for path in sorted((ROOT/'results/experiments').glob('*/summary.json')):
        row=json.loads(path.read_text(encoding='utf-8'))
        if row.get('kind')!='transfer_training' or not row.get('completed') or row.get('dataset_provenance')!='official_kaggle_dfdc':continue
        if row.get('evaluation_split')!='validation':raise ValueError('Leaderboard requires validation results')
        if sha256(row['checkpoint'])!=row['checkpoint_sha256']:raise ValueError('Checkpoint changed')
        for unit in ['video','frame']:
            verify_predictions(path.parent/f'validation_{unit}_predictions.csv',row['validation'][unit],row['prediction_sha256'][unit])
        row['source_file']=str(path);records.append(row)
    records.sort(key=lambda r:((r['validation']['video']['roc_auc'] or -1),(r['validation']['video']['balanced_accuracy'] or -1)),reverse=True)
    final_path=ROOT/'results/experiments/final_test/summary.json';final=None
    if final_path.exists():
        final=json.loads(final_path.read_text(encoding='utf-8'))
        if not final.get('completed') or final.get('evaluation_split')!='test':raise ValueError('Incomplete final test')
        videos=verify_predictions(final_path.parent/'test_video_predictions.csv',final['metrics'],final['prediction_sha256']['video'])
        from sklearn.metrics import roc_curve,precision_recall_curve
        y=[int(v['label']=='FAKE') for v in videos];p=[float(v['fake_probability']) for v in videos]
        if len(set(y))==2:
            fpr,tpr,_=roc_curve(y,p);precision,recall,_=precision_recall_curve(y,p)
            final['curves']={'roc':list(map(list,zip(fpr.tolist(),tpr.tolist()))),'pr':list(map(list,zip(recall.tolist(),precision.tolist())))}
        final['source_file']=str(final_path)
    access_path=ROOT/'results/dfdc_access_status.json'
    access=json.loads(access_path.read_text(encoding='utf-8')) if access_path.exists() else {'status':'not_attempted'}
    payload={'dataset':'Official DFDC labeled training sample; generated source-group splits','unit':'video, FAKE-positive, threshold 0.5; mean sampled-frame probability',
             'experiments':records,'final_test':final,'planned':[read_config(p) for p in sorted((ROOT/'configs').glob('*.json'))],
             'access':access,'storage':check_budget(),'confidence':'Uncalibrated model scores','source_directory':str(ROOT/'results/experiments')}
    atomic_json(ROOT.parent/'research/analyses/transfer_experiment_data.json',payload)
    print(json.dumps({'verified_completed_training_runs':len(records),'final_test_completed':final is not None,'status':access},indent=2))
    return payload


if __name__=='__main__':export()
