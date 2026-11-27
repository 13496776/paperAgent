#!/usr/bin/env python3
"""Create a local, filterable review index without modifying evaluation records."""
import argparse
import collections
import csv
import html
import json
import sys
from pathlib import Path

import ijson

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from evaluation.report_store import case_files


def summarize_case(path):
    """Read only review fields, without materializing the workflow JSON tree."""
    output = {'id': None, 'category': '', 'status': '未到达或未知',
              'turns': 0, 'calls': 0, 'failures': 0, 'errors': [], 'question': '',
              'judge_success': None, 'judge_reason': '', 'judge_error': '',
              'compression_required': False, 'compression_verified': False}
    call = None
    call_prefix = 'turns.item.workflow.tasks.item.calls.item'
    def add_error(value):
        if value is not None and sum(map(len, output['errors'])) < 1500:
            output['errors'].append(str(value)[:1500])
    with Path(path).open('rb') as source:
        for prefix, event, value in ijson.parse(source):
            if prefix == 'id' and event == 'string':
                output['id'] = value
            elif prefix == 'category' and event == 'string':
                output['category'] = value
            elif prefix == 'target_workflow.status' and event == 'string':
                output['status'] = value
            elif prefix == 'judge.success' and event in ('boolean', 'null'):
                output['judge_success'] = value
            elif prefix == 'judge.reason' and event == 'string':
                output['judge_reason'] = value[:1500]
            elif prefix == 'judge.error' and event == 'string':
                output['judge_error'] = value[:1500]
            elif prefix in ('compression_required', 'compression_verified') and event == 'boolean':
                output[prefix] = value
            elif prefix == 'turns.item' and event == 'start_map':
                output['turns'] += 1
            elif prefix == 'turns.item.question' and event == 'string':
                output['question'] = value[:500]
            elif prefix in ('error', 'turns.item.error') and event == 'string':
                add_error(value)
            elif prefix == call_prefix and event == 'start_map':
                call = {'name': None, 'status': None, 'error': None, 'result': None}
            elif call is not None and prefix == call_prefix and event == 'end_map':
                if call['name']:
                    output['calls'] += 1
                    if call['status'] == 'failed':
                        output['failures'] += 1
                        add_error(call['error'] or call['result'])
                call = None
            elif call is not None and prefix in {call_prefix + '.name', call_prefix + '.status',
                                                   call_prefix + '.error', call_prefix + '.result'}:
                if event in ('string', 'number', 'boolean'):
                    call[prefix.rsplit('.', 1)[-1]] = str(value)[:1500]
            elif call is not None and prefix == call_prefix + '.result.error' and event == 'string':
                call['result'] = value[:1500]
    if not output['id']:
        raise ValueError(f'案例文件缺少 ID: {path}')
    return output


def make_review(directory, legacy=None):
    directory = Path(directory)
    meta = json.loads((directory / 'report.json').read_text(encoding='utf-8'))
    if legacy is None and meta.get('resumed_from'):
        candidate = Path(meta['resumed_from'])
        if candidate.is_file():
            legacy = candidate
    imported = set()
    if legacy is not None:
        with Path(legacy).open('rb') as source:
            legacy_ids = set(ijson.items(source, 'cases.item.id'))
        with Path(legacy).open('rb') as source:
            legacy_dataset = next(ijson.items(source, 'dataset_id'), None)
        if legacy_dataset != meta.get('dataset_id'):
            raise ValueError('旧报告和目标报告的数据集不匹配')
        imported = legacy_ids
    rows = []
    seen = set()
    for path in case_files(directory):
        record = summarize_case(path)
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
        judged = '通过' if record['judge_success'] is True else '未通过' if record['judge_success'] is False else '未评分'
        compression = ('初步确认' if record['compression_verified'] else '未确认') if record['compression_required'] else '不要求'
        rows.append({'案例ID': ident, '类别': record['category'], '来源批次': origin,
                     '目标轮状态': record['status'], '已执行轮数': record['turns'],
                     '裁判结果': judged, '裁判说明': record['judge_reason'] or record['judge_error'],
                     '压缩检查': compression,
                     '工具调用数': record['calls'], '工具失败数': record['failures'],
                     '失败摘要': ' | '.join(record['errors'])[:1500],
                     '最后用户问题': record['question'], '结果文件': relative.as_posix()})
    rows.sort(key=lambda x: x['案例ID'])
    csv_path = directory / 'review.csv'
    fields = ['案例ID', '类别', '来源批次', '目标轮状态', '已执行轮数', '裁判结果', '裁判说明', '压缩检查', '工具调用数', '工具失败数', '失败摘要', '最后用户问题', '结果文件']
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
                '<label>裁判 <select id="judge"><option value="">全部</option>' + ''.join(f'<option>{escape(v)}</option>' for v in sorted({r['裁判结果'] for r in rows})) + '</select></label>',
                '<label>搜索 <input id="search" placeholder="案例 ID、问题、错误"></label><table><thead><tr>' + ''.join(f'<th>{escape(v)}</th>' for v in fields[:-1]) + '</tr></thead><tbody>']
    from urllib.parse import quote
    for row in rows:
        attrs = ' '.join(f'data-{name}="{escape(row[key])}"' for name,key in [('category','类别'),('origin','来源批次'),('status','目标轮状态'),('judge','裁判结果')])
        cells = [f'<td><a href="{quote(row["结果文件"], safe="/")}">{escape(row["案例ID"])}</a></td>']
        cells.extend(f'<td>{escape(row[key])}</td>' for key in fields[1:-1])
        document.append(f'<tr {attrs}>' + ''.join(cells) + '</tr>')
    document.append('''</tbody></table><script>
const inputs=['category','origin','status','judge','search'].map(x=>document.getElementById(x));
function filter(){const [c,o,s,j,q]=inputs.map(x=>x.value.toLowerCase());document.querySelectorAll('tbody tr').forEach(r=>{r.hidden=!!((c&&r.dataset.category.toLowerCase()!==c)||(o&&r.dataset.origin.toLowerCase()!==o)||(s&&r.dataset.status.toLowerCase()!==s)||(j&&r.dataset.judge.toLowerCase()!==j)||(q&&!r.textContent.toLowerCase().includes(q)))})}inputs.forEach(x=>x.addEventListener('input',filter));</script></html>''')
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
