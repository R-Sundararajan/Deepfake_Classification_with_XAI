// XLSX authoring uses the bundled public artifact-tool API; no extra installs.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {createRequire} from 'node:module';
const resolved=createRequire(import.meta.url).resolve('@oai/artifact-tool',{paths:process.env.DFDC_NODE_MODULES?[process.env.DFDC_NODE_MODULES]:undefined});
const {Workbook, SpreadsheetFile}=await import(pathToFileURL(resolved).href);
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const wb=Workbook.create();
if(process.argv.includes('--api-help')) {
  console.log(wb.help('fx.HYPERLINK',{include:'index,examples,notes',maxChars:3500}).ndjson);
  process.exit(0);
}
const literature=JSON.parse(await fs.readFile(path.join(root,'../research/analyses/dfdc_literature.json'),'utf8'));
function sheet(name,headers,rows,widths,tableName){
  const s=wb.worksheets.add(name);s.showGridLines=false;
  const end=String.fromCharCode(64+headers.length);
  const matrix=[headers,...rows];
  s.getRange(`A1:${end}${matrix.length}`).values=matrix;
  const all=s.getRange(`A1:${end}${matrix.length}`);
  all.format.font={name:'Arial',size:10,color:'#17324D'};
  all.format.wrapText=true;all.format.verticalAlignment='top';
  all.format.rowHeight=135;
  s.getRange(`A1:${end}1`).format={fill:'#153B55',font:{bold:true,color:'#FFFFFF',size:11},wrapText:true,rowHeight:42};
  widths.forEach((w,i)=>s.getRange(`${String.fromCharCode(65+i)}:${String.fromCharCode(65+i)}`).format.columnWidth=w);
  if(rows.length){const t=s.tables.add(`A1:${end}${matrix.length}`,true,tableName);t.showFilterButton=true;}
  s.freezePanes.freezeRows(1);s.freezePanes.freezeColumns(2);
  return s;
}
const main=sheet('DFDC Published Results',literature.headers,literature.rows.map(r=>literature.headers.map(h=>r[h])),
  [13,52,8,54,44,48,27,58,30,52,66,15,13,13,58,49,43,58,64,79],'DFDCPublished');
main.getRange(`L2:N${literature.rows.length+1}`).setNumberFormat('0.0000');
main.getRange(`L2:L${literature.rows.length+1}`).setNumberFormat('0.00');
main.getRange(`C2:C${literature.rows.length+1}`).setNumberFormat('0');
// Links include the original local PDF/archive and page references as visible labels.
for(let i=0;i<literature.rows.length;i++){
  const r=literature.rows[i],target=r._source_path.split('!')[0].replaceAll('\\','/');
  const label=r._key+' | '+r.Source;
  main.getRange(`P${i+2}`).formulas=[[`=HYPERLINK("${target.replaceAll('"','""')}","${label.replaceAll('"','""')}")`]];
}
const lh=['Experiment ID','Model','Training Dataset','Evaluation Dataset','Videos','Frames','Preprocessing','Frame Aggregation','Accuracy (%)','Precision','Recall','F1','ROC-AUC','GPU Peak MiB','Training Seconds','Inference Seconds','Checkpoint','Constraints / Limitations'];
let runs=[];try{
  for(const dir of await fs.readdir(path.join(root,'results/dfdc_runs'))){
    try {const r=JSON.parse(await fs.readFile(path.join(root,'results/dfdc_runs',dir,'summary.json'),'utf8'));if(r.completed&&r.dataset_provenance==='official_kaggle_dfdc')runs.push(r);}catch{}
  }
}catch{}
const local=sheet('Local DFDC Baseline',lh,runs.map(r=>[r.experiment_id,r.model,'Existing real_vs_fake image dataset; no DFDC training',r.evaluation_dataset,r.videos_evaluated,r.frames_evaluated,r.preprocessing,r.aggregation,100*r.metrics.accuracy,r.metrics.precision,r.metrics.recall,r.metrics.f1,r.metrics.roc_auc,r.gpu_peak_mib,0,r.elapsed_seconds,r.checkpoint,r.limitations]),[28,24,49,48,13,13,62,46,16,15,15,15,15,19,20,22,66,80],'LocalDFDC');
if(!runs.length){local.getRange('A3').values=[['No local DFDC experiment has completed. Access/preparation status is recorded in the phase report.']];local.getRange('A3:H4').format.rowHeight=32;}
else {local.getRange(`I2:M${runs.length+1}`).setNumberFormat('0.0000');local.getRange(`I2:I${runs.length+1}`).setNumberFormat('0.00');}
const ah=Object.keys(literature.audit[0]);
sheet('Source Audit',ah,literature.audit.map(r=>ah.map(h=>r[h])),[16,17,78,29,100,27,100,75,23],'SourceAudit');
sheet('Reading Notes',['Convention'],literature.conventions.map(x=>[x]),[160],'ReadingNotes').getRange('A2:A9').format.rowHeight=45;
wb.recalculate();
console.log((await wb.inspect({kind:'region',sheetId:main.name,range:'K1:N5',maxChars:1600,tableMaxRows:5,tableMaxCols:4})).ndjson);
const out=await SpreadsheetFile.exportXlsx(wb);await out.save(path.join(root,'../research/workbooks/DFDC_Research_Comparison.xlsx'));
const preview=await wb.render({sheetName:main.name,range:'K1:N5',scale:1.5,format:'png'});
await fs.writeFile(path.join(root,'../research/working/workbook_preview.png'),new Uint8Array(await preview.arrayBuffer()));
// Keep the renderer's verbose inspection sidecar out of the handoff directory.
try{await fs.rename(path.join(root,'../research/workbooks/DFDC_Research_Comparison.xlsx.inspect.ndjson'),path.join(root,'../research/working/workbook.inspect.ndjson'));}catch(e){if(e.code!=='ENOENT')throw e;}
console.log(JSON.stringify({publishedRows:literature.rows.length,localRuns:runs.length,output:'../research/workbooks/DFDC_Research_Comparison.xlsx'}));
