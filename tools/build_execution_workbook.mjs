import fs from 'node:fs/promises';
import {constants} from 'node:fs';
import {execFileSync} from 'node:child_process';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Workbook,SpreadsheetFile,FileBlob} from '@oai/artifact-tool';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const run=path.join(root,'results/execution_20261019');
const research=path.resolve(root,'../research');
const data=JSON.parse(await fs.readFile(path.join(run,'comparison.json'),'utf8'));
const dataset=JSON.parse(await fs.readFile(path.join(run,'dataset/manifest_info.json'),'utf8'));
const metadata=JSON.parse(await fs.readFile(path.join(run,'dfdc_metadata_validation.json'),'utf8'));
const original=JSON.parse(await fs.readFile(path.join(run,'E00_test/metrics.json'),'utf8'));
const rows=[...data.experiments].sort((a,b)=>data.champion.candidate_ids.indexOf(a.experiment_id)-data.champion.candidate_ids.indexOf(b.experiment_id));
const wb=Workbook.create();
const names=['Experiment Leaderboard','Training and Validation','Attention and Augmentation','Hardware and Cost','Held-out Test Results','Champion and Limitations'];
const sheets=new Map(names.map(n=>[n,wb.worksheets.add(n)]));
const navy='#263F5B',ink='#24364B',light='#F1F4F8',font='Arial';
const col=n=>{let result='';while(n){result=String.fromCharCode(65+(n-1)%26)+result;n=Math.floor((n-1)/26);}return result;};
function table(name,title,headers,values,widths){
  const s=sheets.get(name),last=col(headers.length),bottom=6+values.length;
  s.showGridLines=false;
  s.getRange(`A1:${last}${Math.max(bottom,10)}`).format={font:{name:font,size:11,color:ink},verticalAlignment:'center'};
  s.getRange('A2').values=[[title]];s.getRange('A2').format.font={name:font,size:16,bold:true,color:ink};
  s.getRange(`A3:${last}3`).format.borders={bottom:{style:'thin',color:'#A6B5C5'}};
  s.getRange(`A6:${last}${bottom}`).values=[headers,...values];
  s.getRange(`A6:${last}6`).format={fill:navy,font:{name:font,size:11,bold:true,color:'#FFFFFF'},wrapText:true,rowHeight:44,horizontalAlignment:'center',verticalAlignment:'center'};
  if(values.length){
    s.getRange(`A7:${last}${bottom}`).format.rowHeight=27;
    s.tables.add(`A6:${last}${bottom}`,true,name.replaceAll(/[^A-Za-z]/g,'')+'Data');
  }
  widths.forEach((width,i)=>s.getRange(`${col(i+1)}:${col(i+1)}`).format.columnWidth=width);
  s.tabColor=navy;
  return s;
}
function note(s,text,last='N'){
  s.getRange('A4').values=[[text]];
  s.getRange(`A4:${last}4`).format.font={name:font,size:11,italic:true,color:ink};
  s.getRange(`A4:${last}4`).format.rowHeight=23;
}
function chart(s,type,ranges,title,start,end,format='0.0%'){
  const c=s.charts.add(type,ranges.length===1?s.getRange(ranges[0]):ranges.map(r=>s.getRange(r)));
  c.title=title;c.titleTextStyle.typeface=font;c.titleTextStyle.fontSize=14;
  c.legend={position:'top',textStyle:{typeface:font,fontSize:11}};
  c.xAxis={axisType:'textAxis',textStyle:{typeface:font,fontSize:11}};
  c.yAxis={numberFormatCode:format,numberFormatSourceLinked:false,textStyle:{typeface:font,fontSize:11}};
  c.setPosition(start,end);
  c.series.items.forEach((series,i)=>{const color=['#355C86','#D1813C','#54928F','#8769A3'][i%4];series.fill=color;if(type==='line')series.line={fill:color,style:'solid',width:2};});
  return c;
}
const leader=table(names[0],'Transfer learning experiment results',
  ['ID','Model','Attention','Augmentation','Train images','Validation images','Accuracy','Balanced accuracy','Precision','Recall','F1','ROC-AUC','Validation loss','Best epoch'],
  rows.map(r=>[r.experiment_id.replace(/_.*/,''),r.model.backbone,r.model.attention||'none',r.config?.augmentation||'historical',
    r.training_images||100000,r.validation?.samples,r.validation.accuracy,r.validation.balanced_accuracy,r.validation.precision,r.validation.recall,r.validation.f1,r.validation.roc_auc,r.validation.loss,r.best_epoch||2]),
  [11,23,13,17,14,16,13,16,13,13,13,13,15,12]);
note(leader,'Real vs Fake Faces. Validation image metrics, FAKE positive, threshold 0.5. E00 has a larger historical training budget.');
leader.getRange(`E7:F${6+rows.length}`).setNumberFormat('#,##0');
leader.getRange(`G7:K${6+rows.length}`).setNumberFormat('0.00%');
leader.getRange(`L7:L${6+rows.length}`).setNumberFormat('0.0000');
leader.getRange(`M7:M${6+rows.length}`).setNumberFormat('0.0000');
leader.freezePanes.freezeRows(6);leader.freezePanes.freezeColumns(2);
chart(leader,'bar',[`A6:A${6+rows.length}`,`H6:H${6+rows.length}`,`L6:L${6+rows.length}`],'Validation balanced accuracy and ROC-AUC','A16','H32','0.00');
chart(leader,'bar',[`A6:A${6+rows.length}`,`I6:I${6+rows.length}`,`J6:J${6+rows.length}`,`K6:K${6+rows.length}`],'Validation precision, recall and F1','I16','P32');

const epochs=rows.flatMap(r=>(r.history||[]).map(h=>[r.experiment_id.replace(/_.*/,''),h.epoch,h.phase,h.training_loss,h.validation_loss,
  h.validation.accuracy,h.validation.balanced_accuracy,h.validation.f1,h.validation.roc_auc,h.epoch_seconds]));
const training=table(names[1],'Observed training and validation history',
  ['ID','Epoch','Phase','Training loss','Validation loss','Accuracy','Balanced accuracy','F1','ROC-AUC','Epoch time (s)'],epochs,[12,10,17,18,18,15,18,15,15,19]);
note(training,'Frozen head training precedes selective fine-tuning. BatchNorm running statistics remain fixed. Smoke tests are excluded.','J');
training.getRange(`D7:E${6+epochs.length}`).setNumberFormat('0.0000');training.getRange(`F7:H${6+epochs.length}`).setNumberFormat('0.00%');training.getRange(`I7:I${6+epochs.length}`).setNumberFormat('0.0000');
training.getRange(`J7:J${6+epochs.length}`).setNumberFormat('0.0');training.freezePanes.freezeRows(6);
const newRuns=rows.filter(r=>r.history?.length);
const curveRow=epochs.length+10;
training.getRange(`A${curveRow}:${col(1+newRuns.length)}${curveRow+3}`).values=[['Epoch',...newRuns.map(r=>r.experiment_id.replace(/_.*/,''))],
  ...[1,2,3].map(epoch=>[epoch,...newRuns.map(r=>r.history.find(h=>h.epoch===epoch)?.validation.roc_auc??null)])];
training.getRange(`A${curveRow}:${col(1+newRuns.length)}${curveRow+3}`).format.font={name:font,size:11,color:ink};
training.getRange(`B${curveRow+1}:${col(1+newRuns.length)}${curveRow+3}`).setNumberFormat('0.0000');
chart(training,'line',[`A${curveRow}:${col(1+newRuns.length)}${curveRow+3}`],'Validation ROC-AUC by epoch',`A${curveRow+6}`,`J${curveRow+22}`,'0.000');
training.getRange('L:L').format.columnWidth=31;training.getRange('M:N').format.columnWidth=15;
rows.forEach((r,i)=>{
  const first=7+i*7,matrix=r.validation.confusion_matrix_real_fake;
  training.getRange(`L${first-1}:N${first+2}`).format.font={name:font,size:11,color:ink};
  training.getRange(`L${first-1}`).values=[[r.experiment_id.replace(/_.*/,'')+' validation confusion matrix']];
  training.getRange(`L${first}:N${first+2}`).values=[['Actual / predicted','REAL','FAKE'],['REAL',matrix[0][0],matrix[0][1]],['FAKE',matrix[1][0],matrix[1][1]]];
  training.getRange(`L${first}:N${first}`).format={fill:navy,font:{name:font,size:11,bold:true,color:'#FFFFFF'},horizontalAlignment:'center'};
  training.getRange(`M${first+1}:N${first+2}`).setNumberFormat('#,##0');
  training.getRange(`M${first+1}:N${first+2}`).conditionalFormats.add('colorScale',{colors:['#F3F6FA','#537CA3'],thresholds:['min','max']});
});

const lookup=new Map(rows.map((r,i)=>[r.experiment_id,i+7]));
const variants=rows.filter(r=>r.config?.reference_experiment);
const abl=table(names[2],'Measured attention and augmentation ablations',
  ['Reference','Variant','Change','Accuracy delta (pp)','AUC delta','F1 delta','Training time delta (s)','Latency delta (ms/image)'],
  variants.map(r=>[r.config.reference_experiment.replace(/_.*/,''),r.experiment_id.replace(/_.*/,''),r.config.justification,null,null,null,null,null]),
  [15,14,68,21,18,18,26,29]);
note(abl,'Each variant retrains from ImageNet initialization with the same split, seed, epochs and optimizer as its reference.','H');
variants.forEach((r,i)=>{
  const a=lookup.get(r.experiment_id),b=lookup.get(r.config.reference_experiment),ra=rows.indexOf(r)+7,rb=rows.findIndex(x=>x.experiment_id===r.config.reference_experiment)+7;
  abl.getRange(`D${i+7}:H${i+7}`).formulas=[[
    `=('Experiment Leaderboard'!G${a}-'Experiment Leaderboard'!G${b})*100`,
    `='Experiment Leaderboard'!L${a}-'Experiment Leaderboard'!L${b}`,
    `='Experiment Leaderboard'!K${a}-'Experiment Leaderboard'!K${b}`,
    `='Hardware and Cost'!C${ra}-'Hardware and Cost'!C${rb}`,
    `='Hardware and Cost'!D${ra}-'Hardware and Cost'!D${rb}`]];
});
if(variants.length)abl.getRange(`D7:H${6+variants.length}`).setNumberFormat('0.0000');
if(variants.length){abl.getRange(`C7:C${6+variants.length}`).format.wrapText=true;abl.getRange(`A7:H${6+variants.length}`).format.rowHeight=42;}

const cost=table(names[3],'Hardware and measured experiment cost',
  ['ID','Parameters','Training time (s)','Batch 32 (ms/image)','Peak GPU VRAM (MiB)','Checkpoint (MiB)','Batch 1 latency (ms)','Training time scope'],
  rows.map(r=>[r.experiment_id.replace(/_.*/,''),r.parameters||11177538,r.training_seconds,data.common_latency_benchmark.experiments[r.experiment_id].batch32_ms_per_image,r.gpu_peak_mib,r.checkpoint_bytes/1024**2,data.common_latency_benchmark.experiments[r.experiment_id].batch1_ms_per_image,
    r.experiment_id==='E00'?'No retraining in this session':'Epochs, validation and checkpoint saves']),
  [12,19,21,25,26,22,25,58]);
note(cost,'RTX 4060 Laptop, 8 GB. FP32 inference benchmark: same resident input, 5 warm-ups, 30 timed repeats. Excludes preprocessing/XAI. E00 VRAM is evaluation only.','H');
cost.getRange(`B7:B${6+rows.length}`).setNumberFormat('#,##0');cost.getRange(`C7:G${6+rows.length}`).setNumberFormat('0.00');
cost.getRange(`G7:G${6+rows.length}`).setNumberFormat('0.00"  "');
cost.getRange('A15:B18').values=[['Historical E00 training (s)',original.training_seconds],['Additional phase storage (GiB)',data.storage.phase_bytes/1024**3],['Additional storage limit (GiB)',30],['CPU image/video fallback','Observed operational']];
cost.getRange('A15:B25').format.font={name:font,size:11,color:ink};
cost.getRange('A15:A18').format.columnWidth=30;
cost.getRange('B15:B17').setNumberFormat('0.00');
cost.getRange('A19').values=[['GPU memory is the PyTorch peak allocated value; driver and other-process allocations are excluded.']];
cost.getRange('A21:B21').values=[['ID','Training time (s)']];
const trainedRows=rows.filter(r=>r.training_seconds!==null);
cost.getRange(`A22:B${21+trainedRows.length}`).values=trainedRows.map(r=>[r.experiment_id.replace(/_.*/,''),r.training_seconds]);
chart(cost,'bar',[`A21:B${21+trainedRows.length}`],'Training duration','C21','H37','0 "s"');

const final=data.final_test, m=final.metrics;
const finalRows=[['E00 re-evaluation',original.samples,original.accuracy,original.balanced_accuracy,original.precision,original.recall,original.f1_score,original.roc_auc,'Measured during Milestone A']];
if(final.experiment_id!=='E00') finalRows.push([final.experiment_id,m.samples,m.accuracy,m.balanced_accuracy,m.precision,m.recall,m.f1,m.roc_auc,'One evaluation after validation selection']);
const test=table(names[4],'Final held-out image test results',
  ['Model','Test images','Accuracy','Balanced accuracy','Precision','Recall','F1','ROC-AUC','Evaluation'],finalRows,[23,16,16,20,16,16,16,16,57]);
note(test,'Real vs Fake Faces test split. Champion selected using validation metrics only. No DFDC or labeled-video test metrics.','I');
test.getRange(`C7:G${6+finalRows.length}`).setNumberFormat('0.00%');test.getRange(`H7:H${6+finalRows.length}`).setNumberFormat('0.0000');
test.getRange(`H7:H${6+finalRows.length}`).setNumberFormat('0.0000"  "');test.getRange(`B7:B${6+finalRows.length}`).setNumberFormat('#,##0');
const matrix=m.confusion_matrix_real_fake||m.confusion_matrix;
test.getRange('A11:C21').format.font={name:font,size:11,color:ink};
test.getRange('A11:I29').format.rowHeight=24;
test.getRange('A12:C14').values=[['Actual / predicted','REAL','FAKE'],['REAL',matrix[0][0],matrix[0][1]],['FAKE',matrix[1][0],matrix[1][1]]];
test.getRange('A11').values=[['Champion confusion matrix']];
test.getRange('A12:C12').format={fill:navy,font:{name:font,size:11,bold:true,color:'#FFFFFF'},rowHeight:32};
test.getRange('B13:C14').conditionalFormats.add('colorScale',{colors:['#F3F6FA','#537CA3'],thresholds:['min','max']});
test.getRange('B13:C14').setNumberFormat('#,##0');
test.getRange('A17:B18').values=[['Historical accuracy',0.97115],['Historical test images',20000]];
test.getRange('B17').setNumberFormat('0.000%');
test.getRange('A20:B21').values=[['Rerun difference','One image'],['Cause','Not isolated']];
const confusionFile=path.join(run,final.experiment_id==='E00'?'E00_test':'champion_test','confusion_matrix.png');
test.images.add({dataUrl:'data:image/png;base64,'+(await fs.readFile(confusionFile)).toString('base64'),anchor:{from:{row:11,col:4},extent:{widthPx:480,heightPx:370}}});

const champion=data.champion;
const limitations=[
  ['Selected model',champion.experiment_id],
  ['Selection rule',champion.selection_rule],
  ['Training dataset','Real vs Fake Faces; all experiments in this workbook use images'],
  ['Selection data','Same 2,000 validation images for E00 through E04; no test-based hyperparameter choice'],
  ['Training budget','New models: 12,000 balanced images, one frozen-head epoch then two selective epochs. E00 historically used 100,000 images.'],
  ['Test data','20,000 supplied test images. E00 test measured once in Milestone A; selected new model tested once only if selected.'],
  ['Split integrity',dataset.exact_hash_overlap_across_selected_splits?'Exact duplicates found':'No exact-content overlap across selected splits. Identity and near-duplicate leakage remain unverified.'],
  ['DFDC metadata',`${metadata.videos} entries: ${metadata.labels.REAL} REAL, ${metadata.labels.FAKE} FAKE, ${metadata.source_groups} source groups. All originals and inventory entries matched.`],
  ['DFDC media','No video downloads. Prior audit did not establish mirror compatibility with upstream terms; local image-data fallback used.'],
  ['Video metrics','Unavailable: no labeled video benchmark evaluated. No image result is labeled as DFDC or video accuracy.'],
  ['Operational video test','Decoded MP4 fixture, face detection, mean frame FAKE score, representative Grad-CAM, populated Streamlit upload and CPU fallback passed.'],
  ['Video test limitation','The MP4 fixture moves a held-out face image on a canvas. It verifies software behavior and does not measure video generalization.'],
  ['Scores and explanations','Softmax outputs are uncalibrated. Grad-CAM shows model influence and is not forensic evidence of manipulation.'],
  ['Media preprocessing','224px ImageNet-normalized RGB images. Video uses largest-face YuNet crops; missed faces use an explicit full-frame fallback.'],
  ['Checkpoint loading','Champion record includes a SHA-256 digest; loading verifies it. Original baseline checkpoint remains untouched.'],
  ['Attention convention','Attention means an added module. EfficientNet-B0 retains its native squeeze-and-excitation layers.'],
  ['Observed metric sources','Experiment summary.json and history.json records, validation_predictions.csv and saved E00_test/metrics.json.'],
  ['Reproducibility','Seed 42, deterministic CUDA algorithms, pretrained torchvision ImageNet weights, unique experiment checkpoints and split-manifest SHA-256.'],
  ['Experimental limits','One seed and a short training budget. Ablation differences have no repeated-seed confidence interval and are not comparisons with published DFDC protocols.'],
];
const limits=table(names[5],'Champion selection and material limitations',['Topic','Observed result / scope'],limitations,[30,145]);
limits.getRange(`B7:B${6+limitations.length}`).format.wrapText=true;
limits.getRange(`A7:B${6+limitations.length}`).format.rowHeight=38;
limits.freezePanes.freezeRows(6);
wb.recalculate();
const out=path.join(research,'working/execution_20261019');await fs.mkdir(out,{recursive:true});
for(const name of names){
  const range=name===names[0]?'A2:N12':name===names[1]?'A2:J18':name===names[2]?'A2:H10':name===names[3]?'A2:H19':name===names[4]?'A2:I27':`A2:B${6+limitations.length}`;
  const preview=await wb.render({sheetName:name,range,scale:1.2,format:'png'});
  await fs.writeFile(path.join(out,name.replaceAll(/[^A-Za-z]/g,'_')+'.png'),new Uint8Array(await preview.arrayBuffer()));
}
for(const [name,range,suffix] of [[names[0],'A15:P33','validation_plots'],[names[1],`A${curveRow+5}:J${curveRow+23}`,'training_plot'],[names[3],'C20:H38','cost_plot'],[names[1],'L5:N24','validation_confusion_1'],[names[1],'L26:N39','validation_confusion_2']]){
  const preview=await wb.render({sheetName:name,range,scale:1.2,format:'png'});
  await fs.writeFile(path.join(out,suffix+'.png'),new Uint8Array(await preview.arrayBuffer()));
}
const overview=await wb.inspect({kind:'table',sheetId:names[0],range:'A6:N11',include:'values,formulas',maxChars:3000,tableMaxRows:6,tableMaxCols:14});
await fs.writeFile(path.join(out,'leaderboard_inspection.ndjson'),overview.ndjson);
const scan=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},maxChars:1500});
await fs.writeFile(path.join(out,'formula_scan.ndjson'),scan.ndjson);
const target=path.join(research,'workbooks/DFDC_Transfer_Learning_Experiments.xlsx');
try {await fs.access(target);await fs.copyFile(target,path.join(out,'DFDC_Transfer_Learning_Experiments_before_execution.xlsx'),constants.COPYFILE_EXCL);}catch(error){if(error.code!=='ENOENT'&&error.code!=='EEXIST')throw error;}
const file=await SpreadsheetFile.exportXlsx(wb);await file.save(target);
// The public chart help documents no axis-bound fields. Preserve native charts
// and set their standard OpenXML bounds, then render the saved result again.
execFileSync('C:/Users/Rangarajan/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',[path.join(root,'tools/set_chart_axes.py'),target]);
const reopened=await SpreadsheetFile.importXlsx(await FileBlob.load(target));
for(const [name,range,suffix] of [[names[0],'A15:P33','validation_plots'],[names[3],'C20:H38','cost_plot']]){
  const preview=await reopened.render({sheetName:name,range,scale:1.2,format:'png'});
  await fs.writeFile(path.join(out,suffix+'.png'),new Uint8Array(await preview.arrayBuffer()));
}
console.log(JSON.stringify({output:target,experiments:rows.length,champion:champion.experiment_id,sheets:names,formulaScan:scan.ndjson}));
