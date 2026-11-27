#!/usr/bin/env python3
"""Create a local, filterable review index without modifying evaluation records."""
import argparse
import collections
import csv
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evaluation.report_store import case_files, legacy_report


def make_review(directory, legacy=None):
    directory = Path(directory)
    meta = json.loads((directory / 'report.json').read_text(encoding='utf-8'))
    if legacy is None and meta.get('resumed_from'):
        candidate = Path(meta['resumed_from'])
        if candidate.is_file():
            legacy = candidate
    imported = set()
    if legacy is not None:
        for kind, value in legacy_report(legacy):
            if kind == 'metadata':
                if value.get('dataset_id') != meta.get('dataset_id'):
                    raise ValueError('旧报告和目标报告的数据集不匹配')
            else:
                imported.add(value['id'])
    rows = []
    seen = set()
    for path in case_files(directory):
        with path.open(encoding='utf-8') as source:
            record = json.load(source)
        ident = record['id']
        if ident in seen:
            raise ValueError(f'案例 {ident} 有多个结果文件，请人工检查')
        seen.add(ident)
        relative = path.relative_to(directory)
        if relative.parts[0] == 'runs':
            origin = relative.parts[1]
        elif legacy is not None:
            origin = '旧报告导入' if ident in imported else '旧版续跑新增（具体批次未知）'
        else:
            origin = '旧版混合记录（来源未知）'
        turns = record.get('turns', [])
        calls = [call for turn in turns for task in (turn.get('workflow') or {}).get('tasks', [])
                 for call in task.get('calls', []) if call.get('name')]
        failures = [c for c in calls if c.get('status') == 'failed']
        errors = [str(c.get('error') or c.get('result') or '') for c in failures]
        errors += [str(t.get('error')) for t in turns if t.get('error')]
        if record.get('error'):
            errors.append(str(record['error']))
        target = record.get('target_workflow') or {}
        rows.append({'案例ID': ident, '类别': record.get('category', ''), '来源批次': origin,
                     '目标轮状态': target.get('status', '未到达或未知'),
                     '已执行轮数': len(turns), '工具调用数': len(calls), '工具失败数': len(failures),
                     '失败摘要': ' | '.join(errors)[:1500],
                     '最后用户问题': str(turns[-1].get('question', ''))[:500] if turns else '',
                     '结果文件': relative.as_posix()})
    rows.sort(key=lambda x: x['案例ID'])
    csv_path = directory / 'review.csv'
    fields = ['案例ID', '类别', '来源批次', '目标轮状态', '已执行轮数', '工具调用数', '工具失败数', '失败摘要', '最后用户问题', '结果文件']
    with csv_path.open('w', encoding='utf-8-sig', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    groups = collections.Counter(r['类别'] for r in rows)
    origins = collections.Counter(r['来源批次'] for r in rows)
    escape = lambda value: html.escape(str(value), quote=True)
    document = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>测评案例审阅</title>',
                '<style>body{font:14px sans-serif;margin:24px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:7px;vertical-align:top}th{position:sticky;top:0;background:#eee}td{max-width:460px;overflow-wrap:anywhere}select,input{margin:6px;padding:6px}</style>',
                f'<h1>测评案例审阅：{len(rows)} 例</h1><p>本页只索引现有记录，目标轮完成不代表答案正确；错误摘要仅供定位，请查看原始 JSON。</p>',
                '<p>按类别：' + '；'.join(f'{escape(k)}：{v}' for k,v in sorted(groups.items())) + '</p>',
                '<p>按来源：' + '；'.join(f'{escape(k)}：{v}' for k,v in sorted(origins.items())) + '</p>',
                '<p>旧版混合记录如未提供旧报告，无法区分来源；旧报告恢复后新增记录的具体批次无法从旧文件反推。</p>',
                '<label>类别 <select id="category"><option value="">全部</option>' + ''.join(f'<option>{escape(v)}</option>' for v in sorted(groups)) + '</select></label>',
                '<label>批次 <select id="origin"><option value="">全部</option>' + ''.join(f'<option>{escape(v)}</option>' for v in sorted(origins)) + '</select></label>',
                '<label>状态 <select id="status"><option value="">全部</option>' + ''.join(f'<option>{escape(v)}</option>' for v in sorted({r['目标轮状态'] for r in rows})) + '</select></label>',
                '<label>搜索 <input id="search" placeholder="案例 ID、问题、错误"></label><table><thead><tr>' + ''.join(f'<th>{escape(v)}</th>' for v in fields[:-1]) + '</tr></thead><tbody>']
    from urllib.parse import quote
    for row in rows:
        attrs = ' '.join(f'data-{name}="{escape(row[key])}"' for name,key in [('category','类别'),('origin','来源批次'),('status','目标轮状态')])
        cells = [f'<td><a href="{quote(row["结果文件"], safe="/")}">{escape(row["案例ID"])}</a></td>']
        cells.extend(f'<td>{escape(row[key])}</td>' for key in fields[1:-1])
        document.append(f'<tr {attrs}>' + ''.join(cells) + '</tr>')
    document.append('''</tbody></table><script>
const inputs=['category','origin','status','search'].map(x=>document.getElementById(x));
function filter(){const [c,o,s,q]=inputs.map(x=>x.value.toLowerCase());document.querySelectorAll('tbody tr').forEach(r=>{r.hidden=!!((c&&r.dataset.category.toLowerCase()!==c)||(o&&r.dataset.origin.toLowerCase()!==o)||(s&&r.dataset.status.toLowerCase()!==s)||(q&&!r.textContent.toLowerCase().includes(q)))})}inputs.forEach(x=>x.addEventListener('input',filter));</script></html>''')
    (directory / 'review.html').write_text('\n'.join(document), encoding='utf-8')
    return len(rows), csv_path, directory / 'review.html'


def main():
    parser = argparse.ArgumentParser(description='为测评报告生成可筛选的本地审阅索引，不重新执行案例')
    parser.add_argument('report_dir', type=Path)
    parser.add_argument('--legacy', type=Path, help='用于区分旧报告导入案例的原始 JSON 文件')
    args = parser.parse_args()
    count, csv_path, html_path = make_review(args.report_dir, args.legacy)
    print(f'已索引 {count} 个案例；未重新调用模型。')
    print(f'浏览器审阅：{html_path}')
    print(f'表格审阅：{csv_path}')


if __name__ == '__main__':
    main()
