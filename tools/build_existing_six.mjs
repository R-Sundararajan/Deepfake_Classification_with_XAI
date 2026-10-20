import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {fileURLToPath,pathToFileURL} from 'node:url';
const project=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const work=path.join(project,'../research/working/existing_six');
const modulePath=createRequire(import.meta.url).resolve('@oai/artifact-tool',{paths:[process.env.DFDC_NODE_MODULES]});
const {Workbook,SpreadsheetFile}=await import(pathToFileURL(modulePath).href);
const data=JSON.parse(await fs.readFile(path.join(work,'analysis.json'),'utf8'));
const wb=Workbook.create();
const names=['Summary','DFDC Verified','Hardware Feasibility',...data.papers.map(p=>String(p.paper_id))];
const sheets=new Map(names.map(n=>[n,wb.worksheets.add(n)]));
const palette={ink:'#25374A',navy:'#243E59',line:'#CAD4DE',subtle:'#F3F6F9',amber:'#FFF1D6',red:'#FBE7E6'};
const lastCol=n=>{let x=n,s='';while(x){x--;s=String.fromCharCode(65+x%26)+s;x=Math.floor(x/26);}return s;};
function setup(s,title,widths,cols){
  s.showGridLines=false;
  s.getRange(`A1:${lastCol(cols)}4`).format.font={name:'Arial',size:11,color:palette.ink};
  s.getRange('A2').values=[[title]];
  s.getRange('A2').format.font={name:'Arial',size:15,bold:true,color:palette.navy};
  s.getRange(`A2:${lastCol(cols)}2`).format.rowHeight=27;
  widths.forEach((w,i)=>s.getRange(`${lastCol(i+1)}:${lastCol(i+1)}`).format.columnWidth=w);
}
function table(s,start,headers,rows,widths,name){
  const end=lastCol(headers.length),matrix=[headers,...rows];
  const range=s.getRange(`A${start}:${end}${start+rows.length}`);
  range.values=matrix;range.format.font={name:'Arial',size:11,color:palette.ink};
  range.format.wrapText=true;range.format.verticalAlignment='top';
  const header=s.getRange(`A${start}:${end}${start}`);
  header.format={fill:palette.navy,font:{name:'Arial',size:11,color:'#FFFFFF',bold:true},wrapText:true,rowHeight:34,verticalAlignment:'center',horizontalAlignment:'center'};
  const t=s.tables.add(`A${start}:${end}${start+rows.length}`,true,name);
  t.showFilterButton=true;
  rows.forEach((r,idx)=>{
    const lines=r.map((v,j)=>Math.max(1,...String(v??'').split('\n').map(line=>Math.ceil(line.length/Math.max(8,widths[j]*.83)))));
    const height=Math.max(30,Math.max(...lines)*15+12);
    s.getRange(`A${start+idx+1}:${end}${start+idx+1}`).format.rowHeight=height;
  });
  return t;
}
const metricOwners=new Map();
const detailMeta={};
for(const p of data.papers){
  const s=sheets.get(String(p.paper_id)),widths=[19,31,91,19,22,23,34];
  setup(s,`Paper ${p.paper_id}: ${p.title}`,widths,7);
  s.getRange('A3').values=[[`Original PDF: ${p.source}`]];
  s.getRange('A4').values=[[`Physical pages 1-${p.pages} read. SHA-256 ${p.sha256}`]];
  s.getRange('A3:G4').format.rowHeight=20;
  const rows=p.notes.map(n=>[n.section,n.item,n.text,n.value,n.unit,n.status,n.reference]);
  rows.unshift(['Scope','Research role',`${p.role}. Independent DFDC: ${p.independent_dfdc}. ${p.finding}`,null,'','Interpretation',`Complete original PDF pp. 1-${p.pages}`]);
  for(let ri=0;ri<p.performance.length;ri++){
    const r=p.performance[ri];
    for(const [key,label] of [['accuracy','Accuracy'],['precision','Precision'],['recall','Recall'],['f1','F1'],['auc','ROC-AUC']]){
      const row=7+rows.length;
      rows.push(['Performance',label,`${r.dataset}. ${r.protocol}. Model: ${r.model}.${r.other?' '+r.other:''}`,r[key]??'Not Reported','fraction [0,1]',r[key]==null?'Unknown':r.status,r.reference]);
      metricOwners.set(`${p.paper_id}:${ri}:${key}`,row);
    }
  }
  if(p.xai_attack_table){
    for(const [method,types] of Object.entries(p.xai_attack_table)){
      types.forEach((vals,ti)=>vals.forEach((v,ki)=>rows.push(['XAI evaluation','Retained attack accuracy',`${method}; FF++ ${['DF','F2F','FS','NT'][ti]}; attack top-${ki+1} SLIC regions. Lower is stronger influence in this evaluation, not clean-detector improvement.`,v,'fraction [0,1]','Reported','p. 5, Table 1'])));
    }
    for(const [method,types] of Object.entries(p.xai_sufficiency_table)){
      types.forEach((vals,ti)=>vals.forEach((v,ki)=>rows.push(['XAI evaluation','Prediction-drop sufficiency',`${method}; FF++ ${['DF','F2F','FS','NT'][ti]}; top-${ki+1} regions. Higher is a larger prediction drop under this framework.`,v,'fraction [0,1]','Reported','p. 6, Table 2'])));
    }
  }
  table(s,6,['Topic','Component / metric','Technical analysis and experiment context','Numeric value','Unit','Evidence type','Original PDF page / table'],rows,widths,`Paper${p.paper_id}Details`);
  s.getRange(`D7:D${6+rows.length}`).setNumberFormat('0.0000');
  rows.forEach((r,idx)=>{if(r[4]&&!r[4].startsWith('fraction'))s.getRange(`D${idx+7}`).setNumberFormat('#,##0');});
  s.freezePanes.freezeRows(6);s.freezePanes.freezeColumns(2);
  detailMeta[p.paper_id]={rows:rows.length,lastRow:6+rows.length,performanceRows:p.performance.length};
}
const summary=sheets.get('Summary');
const sw=[9,48,32,33,57,48,56,31];
setup(summary,'Technical comparison of six existing papers',sw,8);
summary.getRange('A3').values=[['Only Papers 2, 9, 15, 16, 35 and 45. Original-PDF evidence; no experiments rerun. Original literature workbook unchanged.']];
table(summary,5,['Paper ID','Existing publication','Research role / DFDC status','Actual experimental data','Reported architecture','Explainability and its role','Main finding / qualification','Original PDF evidence'],data.papers.map(p=>[
  p.paper_id,p.title,`${p.role}. ${p.independent_dfdc}`,p.datasets,p.architecture,p.xai,p.finding,
  p.paper_id===2?'pp. 5-11, Tables 2-4':p.paper_id===9?'pp. 3-12, Fig. 6':p.paper_id===15?'pp. 1-7, Table I':p.paper_id===16?'pp. 1-7, Table I':p.paper_id===35?'pp. 3-9, Tables III-IX':'pp. 3-6, Tables 1-4'
]),sw,'SixPapersSummary');
summary.getRange('A14').values=[['Evidence labels: Reported, Reported (review), Calculated, Estimated, Unknown, Interpretation. Full PDF paths and page-specific claims are on the numbered sheets.']];
summary.getRange('A16').values=[['DFDC Verified contains only Paper 2 and Paper 35 results. Their dataset splits and metric units are not matched for ranking.']];
summary.getRange('A18').values=[['All 63 physical PDF pages read. Missing hyperparameters remain unknown; hardware estimates are assumptions, not published performance.']];
summary.tabColor=palette.navy;

const ds=sheets.get('DFDC Verified'),dw=[9,42,56,18,18,18,18,18,61,38,50];
setup(ds,'DFDC-only source-verified reported results',dw,11);
ds.getRange('A3').values=[['Two papers, eight within-study experiment rows. Different test protocols; no cross-paper leaderboard or rerun claim.']];
const drows=[];
for(const p of data.papers)for(let ri=0;ri<p.performance.length;ri++){
  const r=p.performance[ri];if(!r.dfdc)continue;
  const row=6+drows.length;
  drows.push([p.paper_id,r.model,r.dataset,...['accuracy','precision','recall','f1','auc'].map(k=>r[k]??'Not Reported'),r.protocol,r.reference,
    r.proposed?'Proposed result; protocol/reporting caveats on paper sheet':'Within-paper comparison; not another publication']);
  for(const [i,k] of ['accuracy','precision','recall','f1','auc'].entries()){
    // Direct links to the corresponding source facts in the detailed paper sheet.
    const cell=`${lastCol(i+4)}${row}`;
    metricOwners.set(`dfdc:${cell}`,`='${p.paper_id}'!D${metricOwners.get(`${p.paper_id}:${ri}:${k}`)}`);
  }
}
table(ds,5,['Paper ID','Model','DFDC release / subset','Accuracy','Precision','Recall','F1','ROC-AUC','Reported evaluation protocol','Original PDF evidence','Comparison limit'],drows,dw,'VerifiedDFDCResults');
for(const [k,f] of metricOwners)if(k.startsWith('dfdc:'))ds.getRange(k.slice(5)).formulas=[[f]];
ds.getRange('D6:D13').setNumberFormat('0.00%');ds.getRange('E6:H13').setNumberFormat('0.0000');
ds.freezePanes.freezeRows(5);ds.freezePanes.freezeColumns(2);

const hs=sheets.get('Hardware Feasibility'),hw=[10,36,34,24,53,82,59,35];
setup(hs,'Hardware feasibility on an RTX 4060 Laptop GPU',hw,8);
hs.getRange('A3').values=[['User context: 8 GB VRAM, about 16 GB RAM, 30 GiB additional storage on D:. Feasibility is estimated; no workload was executed.']];
table(hs,5,['Paper ID','Published GPU','Epoch information','Batch size','Published parameters / time','Estimated 8 GB feasibility','Constraint / unknown','Original PDF evidence'],data.hardware.map(h=>[
 h.paper_id,h.published_gpu,h.epochs,h.batch,`${h.parameters}. ${h.published_time}`,h.estimate,h.bottleneck,h.ref
]),hw,'SixPaperHardware');
hs.getRange('A14').values=[['Transparent resource and workload calculations']];
hs.getRange('A14').format.font={name:'Arial',size:14,bold:true,color:palette.navy};
const calculations=[
 [9,'Latent height',16,'positions','Reported','P9 p. 4, encoder dimensions'],
 [9,'Latent width',16,'positions','Reported','P9 p. 4'],
 [9,'Latent channels',256,'values','Reported','P9 p. 4'],
 [9,'FP32 bytes per value',4,'bytes','Calculated','Representation assumption; not a paper storage measurement'],
 [9,'Hypothetical cached sequences',100000,'sequences','Estimated','Illustration only; not the published dataset size'],
 [9,'Latent bytes per sequence',null,'bytes','Calculated','Height x width x channels x bytes/value'],
 [9,'Full latent cache',null,'GiB','Calculated','Raw embeddings only; excludes file/object overhead'],
 [9,'Default total prototypes',40,'prototypes','Reported','P9 pp. 4 and 9: 20 per class'],
 [9,'Maximum-score bytes per sequence',null,'bytes','Calculated','40 maximum similarities x 4 bytes'],
 [9,'Maximum-score cache',null,'MiB','Calculated','No full latent grids; same hypothetical count'],
 [45,'Fake manipulation classes',4,'classes','Reported','P45 p. 5, DF/F2F/FS/NT'],
 [45,'XAI videos per fake class',127,'videos','Reported','P45 p. 5; test sample selection'],
 [45,'XAI frames per video',10,'frames','Reported','P45 p. 5; not detector training sampling'],
 [45,'XAI image total',null,'images','Calculated','4 x 127 x 10'],
 [45,'Explanation methods',5,'methods','Reported','P45 pp. 3-5'],
 [45,'Top-k attack settings',3,'settings','Reported','P45 pp. 4-5'],
 [45,'NES noise samples / pairs',40,'samples','Reported','P45 p. 5'],
 [45,'Maximum attack iterations',50,'iterations','Reported','P45 p. 5'],
 [45,'Two-sided queries per sample',2,'queries','Estimated','Intended NES interpretation; printed Algorithm 1 has identical plus signs'],
 [45,'Worst-case attack evaluations',null,'forward evaluations','Calculated','Before early stopping, batching/caching and explanation queries'],
 [45,'Illustrative throughput',100,'evaluations / s','Estimated','Editable scenario; not measured on RTX 4060'],
 [45,'Illustrative compute duration',null,'hours','Calculated','Worst-case evaluations / assumed rate / 3600'],
 [35,'Pilot clip duration',10,'seconds','Reported','P35 p. 8'],
 [35,'Pilot sampling rate',8,'frames / s','Reported','P35 p. 8, Table IX protocol'],
 [35,'Reported per-clip latency',2.17,'seconds','Reported','P35 p. 8, Table IX'],
 [35,'Sampled frames per clip',null,'frames','Calculated','10 x 8'],
 [35,'Implied sampled throughput',null,'sampled frames / s','Calculated','80 / 2.17; does not equal reported 120 fps'],
 [35,'Implied clips per minute',null,'clips / min','Calculated','60 / 2.17; does not equal reported 11.5'],
 [35,'Reported average throughput',120,'fps','Reported','P35 p. 8, Table IX; unit reconciliation unresolved'],
 [35,'Reported clips per minute',11.5,'clips / min','Reported','P35 p. 8, Table IX; conflicts with latency'],
];
table(hs,16,['Paper ID','Calculation input / output','Value','Unit','Evidence type','Basis'],calculations,hw.slice(0,6),'HardwareCalculations');
const formulas={C22:'=C17*C18*C19*C20',C23:'=C22*C21/1024^3',C25:'=C24*C20',C26:'=C25*C21/1024^2',C30:'=C27*C28*C29',C36:'=C30*C31*C32*C33*C34*C35',C38:'=C36/C37/3600',C42:'=C39*C40',C43:'=C42/C41',C44:'=60/C41'};
for(const [cell,f] of Object.entries(formulas))hs.getRange(cell).formulas=[[f]];
hs.getRange('C17:C46').setNumberFormat('#,##0.00');
for(const row of [21,35,37])hs.getRange(`C${row}`).format.fill=palette.amber;
hs.getRange('C37').dataValidation={rule:{type:'decimal',operator:'greaterThan',formula1:0}};
hs.freezePanes.freezeRows(5);hs.freezePanes.freezeColumns(1);

wb.recalculate();
const baseHours=Number(hs.getRange('C38').values[0][0]);
if(Math.abs(baseHours-846.6666666667)>1e-6)throw Error(`Unexpected workload calculation ${baseHours}`);
hs.getRange('C37').values=[[200]];wb.recalculate();
const changedHours=Number(hs.getRange('C38').values[0][0]);
if(Math.abs(changedHours-baseHours/2)>1e-6)throw Error('Throughput scenario does not recalculate');
hs.getRange('C37').values=[[100]];wb.recalculate();
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:30},maxChars:1500});
console.log(errors.ndjson);
console.log((await wb.inspect({kind:'region',sheetId:'DFDC Verified',range:'A6:H13',maxChars:2500,tableMaxRows:8,tableMaxCols:8})).ndjson);
console.log((await wb.inspect({kind:'region',sheetId:'Hardware Feasibility',range:'B30:D38',maxChars:1600,tableMaxRows:9,tableMaxCols:3})).ndjson);
const previews=[['Summary','A1:H11'],['DFDC Verified','A1:H9'],['Hardware Feasibility','A1:F9'],...data.papers.map(p=>[String(p.paper_id),'A6:G13'])];
for(const [name,range] of previews){
 const blob=await wb.render({sheetName:name,range,scale:1,format:'png'});
 await fs.writeFile(path.join(work,`sheet_${name.replaceAll(' ','_')}.png`),new Uint8Array(await blob.arrayBuffer()));
}
const output=path.join(project,'../research/workbooks/Existing_Six_Papers_Detailed_Analysis.xlsx');
const book=await SpreadsheetFile.exportXlsx(wb);await book.save(output);
try{await fs.rename(output+'.inspect.ndjson',path.join(work,'analysis_workbook.inspect.ndjson'));}catch(e){if(e.code!=='ENOENT')throw e;}
await fs.writeFile(path.join(work,'build_checks.json'),JSON.stringify({sheets:names,detailMeta,dfdcRows:drows.length,baseHours,changedHours,scenarioRestored:true},null,2));
console.log(JSON.stringify({output,sheets:names,dfdcRows:drows.length,baseHours,changedHours}));
