"""Independently check the preserved literature workbook; never write an XLSX.

Uses the read-only inventory and values recalculated by native Excel.  Category
counts are rebuilt from the master fields, not copied from the workbook helpers.
Research gaps are checked against their explicit ID lists; their semantic coding
is not re-created or silently changed.
"""
import hashlib
import json
import math
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from xml.etree import ElementTree as ET

PROJECT = Path(__file__).resolve().parents[1]
WORK = PROJECT / '../research/working/expansion'
load = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
inventory = load(WORK / 'original_inventory.json')
native = load(WORK / 'native_excel_validation.json')
master = load(WORK / 'original_master.json')
by_id = {r['Paper No']: r for r in master}
ids = set(by_id)
checks = []


def check(name, condition, details=None):
    checks.append({'check': name, 'passed': bool(condition), 'details': details})


def same(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10)
    return a == b


def value(sheet, cell):
    return inventory['sheets'][sheet]['cells'].get(cell, {}).get('value')


def calculated(sheet, cell):
    return native['Sheets'][sheet]['Cells'].get(cell)


def numeric(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def norm(title):
    return re.sub(r'[^a-z0-9]', '', unicodedata.normalize('NFKD', title).lower())


def explicit_categories(field):
    result = defaultdict(set)
    for row in master:
        for term in row[field].split(','):
            term = term.strip()
            if term and term != 'Not Reported':
                result[term].add(row['Paper No'])
    return result


expected_names = [
    'Master Literature Database', 'Summary Statistics', 'Dashboard',
    'Dataset Analysis', 'Model Analysis', 'Explainable AI Analysis',
    'Research Gap Analysis', 'Performance Comparison', 'Insights',
]
check('Original nine names and order', list(inventory['sheets']) == expected_names)
check('Native Excel opened a read-only backup', native['ReadOnly'] and native['Worksheets'] == 9)
check('58 unique stable paper IDs', len(master) == 58 and len(by_id) == 58 and ids == set(range(1, 59)))
check('38 master columns and table extent', inventory['sheets'][expected_names[0]]['columns'] == 38
      and inventory['sheets'][expected_names[0]]['tables'] == {'MasterLiteratureTable': 'A1:AL59'})
backup_record = inventory['backup_record']
source = Path(backup_record['Original'])
backup = Path(backup_record['Backup'])
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest().upper()
hashes = {'source': sha(source), 'backup': sha(backup), 'recorded': backup_record['SHA256']}
check('Authoritative file equals the post-close timestamped backup', len(set(hashes.values())) == 1, hashes)
check('Workbook byte count unchanged', source.stat().st_size == backup.stat().st_size == backup_record['Bytes'])

formula_cells = []
cached_mismatches = []
broken_formula_literals = []
for sheet, data in inventory['sheets'].items():
    for address, cell in data['cells'].items():
        if cell['type'] == 'f':
            formula_cells.append((sheet, address))
            if '#REF!' in cell['value']:
                broken_formula_literals.append((sheet, address))
            if not same(cell['cached'], calculated(sheet, address)):
                cached_mismatches.append((sheet, address, cell['cached'], calculated(sheet, address)))
check('No formula errors returned by native Excel rebuild', not native['FormulaErrors'], native['FormulaErrors'])
check('All cached formula values agree with native recalculation', not cached_mismatches,
      {'formula_cells': len(formula_cells), 'mismatches': cached_mismatches})
check('No broken reference literals', not broken_formula_literals, broken_formula_literals)

years = Counter(r['Year'] for r in master)
types = Counter(r['Paper Type'] for r in master)
media = Counter(r['Media Type'] for r in master)
for start, end, label_col, count_col, field_counts in [
    (26, 30, 'A', 'B', years), (26, 32, 'E', 'F', types),
    (82, 86, 'A', 'B', years), (82, 88, 'E', 'F', types),
    (82, 86, 'I', 'J', media),
]:
    for r in range(start, end + 1):
        label = value('Summary Statistics', f'{label_col}{r}')
        check(f'Summary {count_col}{r}: {label}', same(field_counts[label], calculated('Summary Statistics', f'{count_col}{r}')))

accuracy = [r['Accuracy'] for r in master if numeric(r['Accuracy'])]
f1 = [r['F1 Score'] for r in master if numeric(r['F1 Score'])]
auc = [r['ROC-AUC'] for r in master if numeric(r['ROC-AUC'])]
expected_summary = {
    'B4': 58, 'B5': types['Research Article'] + types['Research / Review Article'],
    'B6': types['Review Article'] + types['Systematic Review'] + types['Research / Review Article'],
    'B7': types['Conference Paper'], 'B8': types['Survey Paper'], 'B9': types['Thesis'],
    'B10': mean(accuracy), 'B11': max(accuracy), 'B12': min(accuracy),
    'B13': mean(f1), 'B14': mean(auc), 'B19': media['Video'],
    'B20': media['Image'], 'B21': media['Audio'], 'B22': media['Multimodal'],
}
for c, expected in expected_summary.items():
    check(f'Independent summary {c}', same(expected, calculated('Summary Statistics', c)))
for row in range(26, 31):
    year = value('Summary Statistics', f'A{row}')
    vals = [r['Accuracy'] for r in master if r['Year'] == year and numeric(r['Accuracy'])]
    check(f'Mean accuracy by year {year}', same(mean(vals) if vals else 'Not Reported', calculated('Summary Statistics', f'C{row}')))

category_results = {}
for sheet, field, paper_col, incidence_cols, summary_cols in [
    ('Dataset Analysis', 'Dataset(s)', 'G', ('AA', 'AB', 'AC'), ('H', 'I')),
    ('Model Analysis', 'Prediction Model(s)', 'J', ('AE', 'AF', 'AG'), ('K', 'L')),
    ('Explainable AI Analysis', 'Explainability Method(s)', 'F', ('AI', 'AJ'), ('N', 'O')),
]:
    categories = explicit_categories(field)
    known = set()
    results = {}
    for row in range(2, inventory['sheets'][sheet]['rows'] + 1):
        label = value(sheet, f'A{row}')
        known.add(label)
        assoc = value(sheet, f'{paper_col}{row}')
        associated_ids = set(map(int, re.findall(r'\d+', assoc))) if assoc != 'Not Reported' else set()
        expected = categories[label]
        check(f'{sheet}: {label} unique-paper count', same(len(expected), calculated(sheet, f'B{row}')))
        check(f'{sheet}: {label} paper associations', associated_ids == expected)
        if len(incidence_cols) == 3:
            vals = [by_id[x]['Accuracy'] for x in expected if numeric(by_id[x]['Accuracy'])]
            check(f'{sheet}: {label} stored headline mean', same(mean(vals) if vals else 'Not Reported', calculated(sheet, f'H{row}')))
        results[label] = {'count': len(expected), 'paper_ids': sorted(expected)}
    check(f'{sheet}: no uncategorized explicit terms', not (set(categories) - known), sorted(set(categories) - known))
    actual_pairs = []
    for row in range(120, inventory['sheets']['Summary Statistics']['rows'] + 1):
        paper = value('Summary Statistics', f'{incidence_cols[0]}{row}')
        label = value('Summary Statistics', f'{incidence_cols[1]}{row}')
        if paper is None and label is None:
            continue
        actual_pairs.append((paper, label))
        if len(incidence_cols) == 3:
            score = value('Summary Statistics', f'{incidence_cols[2]}{row}')
            check(f'{sheet}: incidence metric for {paper}/{label}', same(score, by_id.get(paper, {}).get('Accuracy')))
    expected_pairs = {(paper, label) for label, papers in categories.items() for paper in papers}
    check(f'{sheet}: helper incidence equals master terms', set(actual_pairs) == expected_pairs and len(actual_pairs) == len(expected_pairs))
    for row in range(26, 26 + len(known)):
        label = value('Summary Statistics', f'{summary_cols[0]}{row}')
        check(f'{sheet}: summary category {label}', same(len(categories[label]), calculated('Summary Statistics', f'{summary_cols[1]}{row}')))
    category_results[sheet] = results

for cell, expected in [('B15', len(category_results['Dataset Analysis'])), ('B16', len(category_results['Model Analysis'])),
                       ('B17', sum(v['count'] > 0 for v in category_results['Explainable AI Analysis'].values())),
                       ('B18', sum(value('Dataset Analysis', f'I{r}') == 'Explicitly described as a benchmark dataset' for r in range(2, 23)))]:
    check(f'Independent summary {cell}', same(expected, calculated('Summary Statistics', cell)))

gaps = {}
for row in range(2, 18):
    label = value('Research Gap Analysis', f'A{row}')
    assoc = value('Research Gap Analysis', f'C{row}')
    papers = list(map(int, re.findall(r'\d+', assoc))) if assoc != 'Not Reported' else []
    check(f'Research gap {label}: count and valid unique IDs', value('Research Gap Analysis', f'B{row}') == len(set(papers))
          and len(papers) == len(set(papers)) and set(papers) <= ids)
    examples = value('Research Gap Analysis', f'D{row}')
    if examples and examples != 'Not Reported':
        example_ids = list(map(int, re.findall(r'(?:^|; )([0-9]+):', examples)))
        check(f'Research gap {label}: examples use associated IDs', set(example_ids) <= set(papers))
    gaps[label] = {'count': len(papers), 'paper_ids': papers}

performance_ids = []
for row in range(2, 60):
    label = value('Performance Comparison', f'A{row}')
    paper = int(re.match(r'Paper (\d+):', label).group(1))
    performance_ids.append(paper)
    record = by_id[paper]
    check(f'Performance {paper}: title', label == f"Paper {paper}: {record['Title']}")
    for col, field in [('B', 'Best Model'), ('C', 'Dataset(s)'), ('D', 'Accuracy'), ('E', 'Precision'),
                       ('F', 'Recall'), ('G', 'F1 Score'), ('H', 'ROC-AUC'), ('I', 'Explainability Method(s)')]:
        check(f'Performance {paper}: {field}', same(value('Performance Comparison', f'{col}{row}'), record[field]))
    score = record['Accuracy']
    expected_rank = 1 + sum(v > score for v in accuracy) if numeric(score) else 'Not Reported'
    check(f'Performance {paper}: competition rank', same(expected_rank, calculated('Performance Comparison', f'J{row}')))
check('Performance contains all 58 papers once', len(set(performance_ids)) == 58 and set(performance_ids) == ids)
performance_numeric = [by_id[x]['Accuracy'] for x in performance_ids if numeric(by_id[x]['Accuracy'])]
check('Numeric performance rows sorted descending', performance_numeric == sorted(accuracy, reverse=True))

insight_basis = {
    'numeric_accuracy_count': len(accuracy),
    'accuracy_range_count': sum(isinstance(r['Accuracy'], str) and r['Accuracy'] != 'Not Reported' for r in master),
    'generalization_yes': sum(r['Generalization Study'] == 'Yes' for r in master),
    'cross_dataset_yes': sum(r['Cross Dataset Evaluation'] == 'Yes' for r in master),
    'adversarial_yes': sum(r['Adversarial Robustness'] == 'Yes' for r in master),
    'positive_gap_categories': sum(v['count'] > 0 for v in gaps.values()),
    'highest_accuracy_ids': sorted(r['Paper No'] for r in master if r['Accuracy'] == max(accuracy)),
}
check('Insights count basis', insight_basis == {
    'numeric_accuracy_count': 36, 'accuracy_range_count': 0,
    'generalization_yes': 11, 'cross_dataset_yes': 5, 'adversarial_yes': 8,
    'positive_gap_categories': 15, 'highest_accuracy_ids': [32, 40],
}, insight_basis)
content_findings = [{
    'sheet': 'Insights', 'cell': 'B7',
    'preserved_claim': value('Insights', 'B7'),
    'independent_evidence': 'Master Accuracy has 36 numeric fractions and 22 Not Reported cells; no additional range-valued accuracy cell. Paper 27 has an older range in Other Metrics but already has a numeric headline.',
    'proposed_correction': 'Remove the unsupported additional range-only paper statement. Preserve the workbook unchanged under the zero-accepted instruction.',
}]

dashboard_expected = {
    'A4': 58, 'E4': mean(accuracy), 'I4': max(accuracy),
    'M4': 'DeepFake Detection Challenge', 'Q4': 'CNN', 'U4': 'Attention',
    'Y4': expected_summary.get('B18', calculated('Summary Statistics', 'B18')),
}
for cell, expected in dashboard_expected.items():
    check(f'Dashboard KPI {cell}', same(expected, calculated('Dashboard', cell)))

numeric_issues = []
for record in master:
    for field in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC', 'Specificity', 'MCC']:
        v = record[field]
        if numeric(v) and not (-1 if field == 'MCC' else 0) <= v <= 1:
            numeric_issues.append((record['Paper No'], field, v))
check('Numeric metrics retain fraction scale', not numeric_issues, numeric_issues)

title_groups = defaultdict(list)
for record in master:
    title_groups[norm(record['Title'])].append(record['Paper No'])
existing_duplicate_groups = sorted((sorted(v) for v in title_groups.values() if len(v) > 1))
candidate_pdfs = load(WORK / 'candidate_pdf_inspections.json')
for candidate in candidate_pdfs:
    # Source-audit titles are supplemented in the separate decision inventory.
    check(f"{candidate['key']}: complete supplied PDF parsed", candidate['all_pages_parsed'] and bool(candidate['methodology_pages']) and bool(candidate['results_pages']))

ns = {'c': 'http://schemas.openxmlformats.org/drawingml/2006/chart',
      's': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
chart_references = []
with zipfile.ZipFile(source) as original_zip, zipfile.ZipFile(backup) as backup_zip:
    check('XLSX archive CRCs valid', original_zip.testzip() is None and backup_zip.testzip() is None)
    check('Every original package part preserved byte for byte', original_zip.namelist() == backup_zip.namelist()
          and all(original_zip.read(name) == backup_zip.read(name) for name in original_zip.namelist()))
    charts = [name for name in original_zip.namelist() if re.fullmatch(r'xl/charts/chart\d+\.xml', name)]
    for chart in charts:
        root = ET.fromstring(original_zip.read(chart))
        for element in root.findall('.//c:f', ns):
            ref = element.text
            if not ref:
                continue
            match = re.fullmatch(r"(?:'([^']+)'|([^'!]+))!\$?([A-Z]+)\$?(\d+)(?::\$?([A-Z]+)\$?(\d+))?", ref)
            resolved = bool(match and (match.group(1) or match.group(2)) in expected_names)
            chart_references.append({'chart': chart, 'reference': ref, 'resolved': resolved})
    check('15 native charts and resolvable chart data references', len(charts) == 15 and all(x['resolved'] for x in chart_references))
    tables = [name for name in original_zip.namelist() if re.fullmatch(r'xl/tables/table\d+\.xml', name)]
    table_ranges = {}
    for part in tables:
        root = ET.fromstring(original_zip.read(part))
        table_ranges[root.attrib['name']] = {'ref': root.attrib['ref'], 'filter': root.find('s:autoFilter', ns).attrib.get('ref') if root.find('s:autoFilter', ns) is not None else None}
    check('Master table and its filter cover all 58 rows', table_ranges['MasterLiteratureTable'] == {'ref': 'A1:AL59', 'filter': 'A1:AL59'})

arithmetic_passed = all(x['passed'] for x in checks)
result = {
    'status': ('PASS_WITH_EXISTING_CONTENT_WARNINGS' if content_findings else 'PASS') if arithmetic_passed else 'FAIL',
    'structural_arithmetic_status': 'PASS' if arithmetic_passed else 'FAIL',
    'content_findings': content_findings,
    'scope': 'Preservation, native recalculation, and consistency with existing master classifications; not a full scientific re-audit of all 58 papers.',
    'source': str(source), 'backup': str(backup), 'sha256': hashes,
    'source_bytes': source.stat().st_size, 'sheets_changed': [], 'new_paper_ids': [],
    'paper_count': len(master), 'unique_normalized_titles': len(title_groups),
    'existing_duplicate_title_groups': existing_duplicate_groups,
    'formula_count': len(formula_cells), 'chart_count': len(charts),
    'chart_references': chart_references, 'table_ranges': table_ranges,
    'merged_range_count': sum(len(s['merges']) for s in inventory['sheets'].values()),
    'package_parts': len(inventory['package_parts']), 'native_excel_version': native['ExcelVersion'],
    'years': dict(sorted(years.items())), 'paper_types': dict(types), 'media_types': dict(media),
    'numeric_metric_counts': {'Accuracy': len(accuracy), 'F1': len(f1), 'AUC': len(auc)},
    'insight_count_basis': insight_basis,
    'summary': expected_summary, 'categories': category_results, 'research_gaps': gaps,
    'gap_validation_limit': 'Frequencies rebuilt from unique listed paper IDs; original limitations taxonomy preserved, not independently re-coded for every source.',
    'checks_passed': sum(x['passed'] for x in checks), 'checks_total': len(checks), 'checks': checks,
}
destination = PROJECT / '../research/analyses/DFDC_Literature_Expansion_Validation.json'
destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps({key: result[key] for key in ['status', 'checks_passed', 'checks_total', 'formula_count', 'chart_count', 'paper_count', 'unique_normalized_titles', 'existing_duplicate_title_groups']}, indent=2))
for item in checks:
    if not item['passed']:
        print('FAILED', item)
raise SystemExit(0 if arithmetic_passed else 1)
