// Read-only previews of the verified backup; never exports or edits the XLSX.
import fs from 'node:fs/promises';
import path from 'node:path';
import {createRequire} from 'node:module';
import {pathToFileURL,fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const out=path.join(root,'../research/working/expansion');
const resolve=createRequire(import.meta.url).resolve('@oai/artifact-tool',{paths:[process.env.DFDC_NODE_MODULES]});
const {FileBlob,SpreadsheetFile}=await import(pathToFileURL(resolve).href);
const info=JSON.parse(await fs.readFile(path.join(out,'original_inventory.json'),'utf8'));
const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(info.source));
console.log((await wb.inspect({kind:'sheet',include:'id,name',maxChars:2000})).ndjson);
const ranges={
 'Master Literature Database':'A1:H5','Summary Statistics':'A1:F22',
 'Dashboard':'A1:AF34','Dataset Analysis':'A1:C12','Model Analysis':'A1:H12',
 'Explainable AI Analysis':'A1:F10','Research Gap Analysis':'A1:C12',
 'Performance Comparison':'A1:J10','Insights':'A1:C8'};
for(const [name,range] of Object.entries(ranges)){
 const image=await wb.render({sheetName:name,range,scale:1,format:'png'});
 await fs.writeFile(path.join(out,name.replaceAll(' ','_')+'.png'),new Uint8Array(await image.arrayBuffer()));
 console.log('Rendered '+name);
}
