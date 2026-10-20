"""Set standard OpenXML bar-axis bounds missing from the documented chart API."""
import json
import math
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
C='http://schemas.openxmlformats.org/drawingml/2006/chart'
A='http://schemas.openxmlformats.org/drawingml/2006/main'
ET.register_namespace('c',C);ET.register_namespace('a',A)
ET.register_namespace('r','http://schemas.openxmlformats.org/officeDocument/2006/relationships')


def fix(path):
    path=Path(path).resolve()
    if not path.is_relative_to((ROOT.parent/'research').resolve()): raise ValueError('Workbook must stay under the research workspace')
    temporary=path.with_suffix('.axes.pending.xlsx')
    if temporary.exists(): raise FileExistsError('Previous unfinished chart-axis output preserved')
    comparison=json.loads((ROOT/'results/execution_20261019/comparison.json').read_text())
    time_max=math.ceil(max(r['training_seconds'] or 0 for r in comparison['experiments'])/50)*50
    changes=[]
    with zipfile.ZipFile(path) as source,zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED) as destination:
        for info in source.infolist():
            content=source.read(info.filename)
            if '/charts/' in info.filename and info.filename.endswith('.xml'):
                tree=ET.fromstring(content)
                if tree.find('.//{'+C+'}barChart') is not None:
                    title=' '.join(t.text or '' for t in tree.findall('.//{'+C+'}title//{'+A+'}t'))
                    for axis in tree.findall('.//{'+C+'}valAx'):
                        scaling=axis.find('{'+C+'}scaling')
                        if scaling is None: scaling=ET.SubElement(axis,'{'+C+'}scaling')
                        for child in list(scaling):
                            if child.tag in {'{'+C+'}min','{'+C+'}max'}: scaling.remove(child)
                        ET.SubElement(scaling,'{'+C+'}max',{'val':str(time_max if 'Training duration' in title else 1)})
                        ET.SubElement(scaling,'{'+C+'}min',{'val':'0'})
                    content=ET.tostring(tree,encoding='utf-8',xml_declaration=True)
                    changes.append(info.filename)
            destination.writestr(info,content)
    if len(changes)!=3: raise ValueError('Expected exactly three bar charts; unfinished output preserved')
    temporary.replace(path)
    print(json.dumps({'zero_based_bar_axes':changes}))


if __name__=='__main__': fix(sys.argv[1])
