// Research workbook: only verified saved official-data results become measurements.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Workbook,SpreadsheetFile} from '@oai/artifact-tool';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const research=path.resolve(root,'../research');
const data=JSON.parse(await fs.readFile(path.join(research,'analyses/transfer_experiment_data.json'),'utf8'));
const wb=Workbook.create();
const names=['Experiment Leaderboard','Ablation Comparison','Final Test Results','Research Interpretation','Charts'];
const sheets=new Map(names.map(n=>[n,wb.worksheets.add(n)]));
const font='Arial',ink='#24364B',navy='#263F5B',muted='#F1F4F8';
const col=n=>{let value='';for(;n;n=Math.floor((n-1)/26))value=String.fromCharCode(65+(n-1)%26)+value;return value;};
function setup(name,title,headers,rows,widths){
  const s=sheets.get(name);const last=col(headers.length),bottom=Math.max(7,6+rows.length);
  s.showGridLines=false;s.getRange(`A1:${last}${bottom}`).format.font={name:font,size:11,color:ink};
  s.getRange(`A1:${last}${bottom}`).format.verticalAlignment='center';
  s.getRange('A2').values=[[title]];s.getRange('A2').format.font={name:font,size:15,bold:true,color:ink};
  s.getRange(`A3:${last}3`).format.borders={bottom:{style:'thin',color:'#A6B5C5'}};
  s.getRange(`A6:${last}${bottom}`).values=[headers,...(rows.length?rows:[headers.map(()=>null)])];
  s.getRange(`A6:${last}6`).format={fill:navy,font:{name:font,size:11,color:'#FFFFFF',bold:true},wrapText:true,rowHeight:48,horizontalAlignment:'center'};
  s.getRange(`A7:${last}${bottom}`).format.rowHeight=36;
  s.getRange(`A7:${last}${bottom}`).format.wrapText=true;
  widths.forEach((w,i)=>s.getRange(`${col(i+1)}:${col(i+1)}`).format.columnWidth=w);
  if(rows.length){s.tables.add(`A6:${last}${bottom}`,true,name.replaceAll(' ','')+'Data');s.freezePanes.freezeRows(6);}
  s.tabColor=names.indexOf(name)<3?navy:'#7E91A5';
  return s;
}
const headers=['Experiment ID','Model','Pretrained source','Trainable layers / strategy','Attention mechanism','Fusion approach','Augmentation configuration','Input resolution','Frames per video','Training videos / frames','Validation videos / frames','Accuracy','Balanced accuracy','Precision','Recall','F1','ROC-AUC','Log loss','Training duration (s)','Peak VRAM (MiB)','Checkpoint size (MiB)','Inference latency (ms/video)','Experiment notes'];
const experiments=data.experiments;
const leaderRows=experiments.map(r=>{const c=r.config,m=r.validation.video;return [r.experiment_id,r.model.backbone,r.pretrained_source,c.fine_tune_epochs?`Head then last ${c.fine_tune_blocks} block(s); BN frozen`:'Frozen pretrained features; train head/added modules',
  r.model.attention+(r.model.backbone!=='resnet18'?'; native SE retained':''),r.model.fusion,c.augmentation,c.input_size,r.preprocessing.frames_per_video,
  `${r.training_videos} / ${r.training_frames}`,`${r.validation_videos} / ${r.validation_frames}`,m.accuracy,m.balanced_accuracy,m.precision,m.recall,m.f1,m.roc_auc,m.log_loss,r.training_seconds,r.gpu_peak_mib,r.checkpoint_bytes/1024**2,r.validation.timing.ms_per_video,
  'Validation video metrics. Latency excludes decoding/detection/XAI. Uncalibrated.']});
const leader=setup(names[0],'DFDC transfer learning experiments',headers,leaderRows,[30,25,40,48,35,24,28,18,18,25,25,17,20,17,17,17,17,17,23,22,25,28,70]);
leader.getRange('A4').values=[[experiments.length?`${experiments.length} completed comparable training configurations. FAKE-positive video metrics, threshold 0.5.`:'No completed DFDC training runs. Official sample metadata download returns HTTP 403 Forbidden.']];
if(!experiments.length){leader.getRange('A8').values=[['Metrics remain unavailable until official-data experiments complete.']];}
else{leader.getRange(`L7:Q${6+experiments.length}`).setNumberFormat('0.0%');leader.getRange(`R7:V${6+experiments.length}`).setNumberFormat('0.00');leader.freezePanes.freezeColumns(2);}
const index=new Map(experiments.map((r,i)=>[r.experiment_id,i+7]));
const comparisons=data.planned.filter(c=>c.reference_experiment||c.initialize_from).map(c=>({id:c.experiment_id,reference:c.reference_experiment||'A1_resnet18_frozen',change:c.comparison}));
const abl=setup(names[1],'Controlled ablation comparisons',['Reference ID','Variant ID','Controlled change','Status','Accuracy difference (pp)','AUC difference','F1 difference','Additional training (s)','Latency difference (ms/video)'],comparisons.map(c=>[c.reference,c.id,c.change,index.has(c.reference)&&index.has(c.id)?'Evaluated':'Not evaluated',null,null,null,null,null]),[30,32,78,22,25,22,22,30,34]);
comparisons.forEach((c,i)=>{if(index.has(c.id)&&index.has(c.reference)){const a=index.get(c.id),b=index.get(c.reference);abl.getRange(`E${i+7}:I${i+7}`).formulas=[[
  `=('Experiment Leaderboard'!L${a}-'Experiment Leaderboard'!L${b})*100`, `='Experiment Leaderboard'!Q${a}-'Experiment Leaderboard'!Q${b}`,`='Experiment Leaderboard'!P${a}-'Experiment Leaderboard'!P${b}`,`='Experiment Leaderboard'!S${a}-'Experiment Leaderboard'!S${b}`,`='Experiment Leaderboard'!V${a}-'Experiment Leaderboard'!V${b}`]];}});
abl.getRange(`E7:I${6+comparisons.length}`).setNumberFormat('0.000');
abl.getRange('A4').values=[['Differences are blank when either configuration has not completed. Positive latency/time differences mean higher cost.']];
const final=data.final_test;
const finalRows=final?[['Champion experiment',final.experiment_id],['Evaluation unit','Untouched held-out test videos'],['Videos',final.videos],...Object.entries(final.metrics).filter(([k,v])=>typeof v==='number'||v===null).map(([k,v])=>[k,v]),['Confusion matrix (REAL/FAKE)',JSON.stringify(final.metrics.confusion_matrix_real_fake)],['Source-group 95% intervals',JSON.stringify(final.source_bootstrap_95ci.metrics)]]:[['Status','Not evaluated'],['Champion','Not selected; no completed DFDC training experiments'],['Held-out test','Reserved for one evaluation after validation-based selection'],['Metric values',null]];
const test=setup(names[2],'Final held-out test results',['Measure','Result'],finalRows,[48,110]);
test.getRange('A4').values=[['Final test results are separate from iterative validation. No test-set-driven model selection.']];
const interpretations=[
  ['Acquisition',data.access.http_status===403?'Official Kaggle metadata download denied: HTTP 403. No DFDC videos obtained.':'See saved official acquisition status.'],
  ['Transfer learning hypothesis',experiments.length?'Assess comparable validation differences and costs below. Pretraining alone is not a guarantee.':'Not tested on DFDC. No accuracy or generalization improvement can be claimed.'],
  ['Attention','ECA and CBAM are trainable classification components. EfficientNet-B0/MobileNetV3 retain native squeeze-and-excitation.'],
  ['Explainability','Grad-CAM and the Grad-CAM++ gradient-power approximation explain scores. They do not improve classification accuracy.'],
  ['Hybrid','CNN plus 16 normalized FFT radial bands is a planned ablation. No benefit established.'],
  ['Research sources','Paper 2 motivates attention; Paper 35 motivates fusion. Papers 9/45 use FF++, while 15/16 are surveys. Their values are not local DFDC baselines.'],
  ['Preprocessing','Four deterministic frames/video; largest YuNet face and optional landmark alignment; no-face full-frame fallback. Shared 224 px RGB crops.'],
  ['Evaluation','Mean sampled-frame FAKE probability; FAKE-positive metrics. Source groups stay together. Actor disjointness is not established.'],
  ['Generalization','Small official training sample, class imbalance and actor overlap limit interpretation. No independent external dataset evaluated.'],
  ['Model selection','Validation video ROC-AUC, balanced accuracy, F1, then measured inference cost. A final test gate prevents automatic reruns.'],
  ['Confidence','Uncalibrated model probabilities. Heatmaps show influence, not forensic proof.']
];
const interpret=setup(names[3],'Research interpretation and experiment status',['Topic','Interpretation'],interpretations,[36,132]);
const planStart=interpretations.length+10;
interpret.getRange(`A${planStart}:B${planStart+data.planned.length}`).values=[['Experiment ID','Configuration and status'],...data.planned.map(c=>[c.experiment_id,`${c.model.backbone}; added attention: ${c.model.attention}; fusion: ${c.model.fusion}; augmentation: ${c.augmentation}. ${index.has(c.experiment_id)?'Completed':'Not evaluated'}.`])];
interpret.getRange(`A${planStart}:B${planStart}`).format={fill:navy,font:{name:font,size:11,color:'#FFFFFF',bold:true},rowHeight:30};
interpret.getRange(`A${planStart+1}:B${planStart+data.planned.length}`).format={font:{name:font,size:11,color:ink},wrapText:true,rowHeight:36};
const chartSheet=setup(names[4],'Measured experiment charts',['Chart','Status'],[['Validation accuracy',experiments.length?'Generated below':'Not evaluated'],['Validation F1 and ROC-AUC',experiments.length?'Generated below':'Not evaluated'],['Accuracy versus cached inference cost',experiments.length?'Generated below':'Not evaluated'],['Training time versus accuracy',experiments.length?'Generated below':'Not evaluated'],['Ablation differences',comparisons.some(c=>index.has(c.id)&&index.has(c.reference))?'See controlled comparisons':'Not evaluated'],['Champion confusion matrix / ROC / PR',final?'Generated from final predictions':'Not evaluated']],[58,84]);
if(experiments.length){
  chartSheet.getRange(`D6:J${6+experiments.length}`).values=[['Experiment','Accuracy','F1','ROC-AUC','Latency ms/video','Training s','Accuracy %'],...experiments.map(r=>[r.experiment_id,r.validation.video.accuracy,r.validation.video.f1,r.validation.video.roc_auc,r.validation.timing.ms_per_video,r.training_seconds,r.validation.video.accuracy])];
  const bindings=[['bar',['D','E'],'Validation video accuracy','A16','J33'],['bar',['D','F','G'],'Validation video F1 and ROC-AUC','L16','U33'],['scatter',['H','E'],'Accuracy versus cached inference cost','A36','J53'],['scatter',['I','J'],'Training time versus accuracy','L36','U53']];
  for(const [kind,columns,title,start,end] of bindings){const chart=chartSheet.charts.add(kind,columns.map(c=>chartSheet.getRange(`${c}6:${c}${6+experiments.length}`)));chart.title=title;chart.titleTextStyle.typeface=font;chart.setPosition(start,end);chart.yAxis={numberFormatCode:'0%',numberFormatSourceLinked:false,textStyle:{typeface:font}};}
}
if(final){
  const matrix=final.metrics.confusion_matrix_real_fake;
  test.getRange('D6:F8').values=[['Actual / predicted','REAL','FAKE'],['REAL',matrix[0][0],matrix[0][1]],['FAKE',matrix[1][0],matrix[1][1]]];
  test.getRange('D6:F8').format.font={name:font,size:11,color:ink};test.getRange('D:D').format.columnWidth=28;
  test.getRange('E:E').format.columnWidth=16;test.getRange('F:F').format.columnWidth=16;
  test.getRange('E7:F8').conditionalFormats.add('colorScale',{colors:['#F3F6FA','#537CA3'],thresholds:['min','max']});
  for(const [key,label,columns,start,end] of [['roc','Champion ROC',['AA','AB'],'A56','J74'],['pr','Champion precision–recall',['AD','AE'],'L56','U74']]){
    const points=final.curves?.[key];if(!points?.length)continue;
    chartSheet.getRange(`${columns[0]}6:${columns[1]}${6+points.length}`).values=[[key==='roc'?'False positive rate':'Recall',key==='roc'?'True positive rate':'Precision'],...points];
    const chart=chartSheet.charts.add('scatter',columns.map(c=>chartSheet.getRange(`${c}6:${c}${6+points.length}`)));
    chart.title=label;chart.titleTextStyle.typeface=font;chart.setPosition(start,end);
    chart.xAxis={numberFormatCode:'0%',numberFormatSourceLinked:false,textStyle:{typeface:font}};
    chart.yAxis={numberFormatCode:'0%',numberFormatSourceLinked:false,textStyle:{typeface:font}};
  }
}
const measuredAblations=comparisons.filter(c=>index.has(c.id)&&index.has(c.reference));
if(measuredAblations.length){
  chartSheet.getRange('AG6:AH6').values=[['Variant','Validation AUC difference']];
  measuredAblations.forEach((c,i)=>{const row=comparisons.indexOf(c)+7;chartSheet.getRange(`AG${i+7}`).values=[[c.id]];chartSheet.getRange(`AH${i+7}`).formulas=[[`='Ablation Comparison'!F${row}`]];});
  const chart=chartSheet.charts.add('bar',chartSheet.getRange(`AG6:AH${6+measuredAblations.length}`));chart.title='Validation AUC ablations';chart.titleTextStyle.typeface=font;chart.setPosition('A78','J95');
}
// Keep provenance beside the input/status records, outside the presentation output.
interpret.getRange('A4').values=[['Sources: completed run summary.json and verified predictions; configs are planned settings, not achieved results.']];
wb.recalculate();
for(const name of names){
  const inspected=await wb.inspect({kind:'region',sheetId:name,range:'A6:D10',maxChars:1200,tableMaxRows:5,tableMaxCols:4});
  await fs.mkdir(path.join(research,'working/transfer'),{recursive:true});
  await fs.writeFile(path.join(research,'working/transfer',name.replaceAll(' ','_')+'.inspect.ndjson'),inspected.ndjson);
  const range=name==='Research Interpretation'?'A2:B12':name==='Experiment Leaderboard'?'A2:D10':name==='Ablation Comparison'?'A2:D12':'A2:B13';
  const preview=await wb.render({sheetName:name,range,scale:1.25,format:'png'});
  await fs.writeFile(path.join(research,'working/transfer',name.replaceAll(' ','_')+'.png'),new Uint8Array(await preview.arrayBuffer()));
}
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:30},maxChars:1500});
await fs.writeFile(path.join(research,'working/transfer/formula_scan.ndjson'),errors.ndjson);
await fs.mkdir(path.join(research,'workbooks'),{recursive:true});
const file=path.join(research,'workbooks/DFDC_Transfer_Learning_Experiments.xlsx');
const exported=await SpreadsheetFile.exportXlsx(wb);await exported.save(file);
try{await fs.copyFile(file+'.inspect.ndjson',path.join(research,'working/transfer/workbook_export.inspect.ndjson'));await fs.unlink(file+'.inspect.ndjson');}catch(error){if(error.code!=='ENOENT')throw error;}
console.log(JSON.stringify({output:file,completedExperiments:experiments.length,finalTest:!!final,sheets:names}));
