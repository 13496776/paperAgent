"""Bounded-memory report storage and legacy report import for pilot runs."""
import json
import os
import re
import tempfile
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.report-', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as output:
            json.dump(value, output, ensure_ascii=False, indent=2)
            output.write('\n')
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def case_path(directory, case_id):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', case_id):
        raise ValueError(f'无效的案例 ID: {case_id!r}')
    return Path(directory) / 'cases' / (case_id + '.json')


def brief(record):
    """Retain only data needed by metrics, leaving large traces on disk."""
    def call(c):
        return {'name': c.get('name'), 'status': c.get('status')}

    turns = []
    for t in record.get('turns', []):
        w = t.get('workflow') or {}
        turns.append({'status': t.get('status'), 'workflow': {'tasks': [
            {'calls': [call(c) for c in task.get('calls', [])]} for task in w.get('tasks', [])]}})
    observation = record.get('observation')
    if observation is not None:
        observation = {**{k: observation[k] for k in ('tool_selection_names_ok', 'labeled_evidence_hits',
                       'labeled_evidence_total', 'groups_hit', 'groups_total')},
                       'calls': [call(c) for c in observation.get('calls', [])]}
    result = {'id': record['id'], 'turns': turns, 'judge': record.get('judge', {}),
              'abstention_required': record.get('abstention_required', False),
              'compression_required': record.get('compression_required', False),
              'compression_verified': record.get('compression_verified', False),
              'target_workflow': {'status': record['target_workflow'].get('status')}
                  if record.get('target_workflow') else {}}
    if observation is not None:
        result['observation'] = observation
    return result


def legacy_report(path):
    """Read the old indented JSON report one case at a time, without loading it all."""
    with open(path, encoding='utf-8') as src:
        header = []
        for line in src:
            if re.match(r'^\s*"cases"\s*:\s*\[', line):
                prefix = ''.join(header).rstrip()
                meta = json.loads(prefix + ' "cases": []}')
                break
            header.append(line)
            if sum(map(len, header)) > 1024 * 1024:
                raise ValueError('无法识别旧报告的 cases 字段')
        else:
            raise ValueError('旧报告中没有 cases 字段')
        yield ('metadata', meta)
        decoder = json.JSONDecoder()
        buffer = ''
        eof = False
        while True:
            buffer = buffer.lstrip(' \r\n\t,')
            if buffer.startswith(']'):
                return
            if not buffer and not eof:
                buffer = src.read(65536)
                eof = not buffer
                continue
            try:
                record, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError as exc:
                if eof:
                    raise ValueError('旧报告不完整，请选择上一次成功保存的 .json 文件') from exc
                more = src.read(max(65536, len(buffer)))
                eof = not more
                buffer += more
                continue
            if not isinstance(record, dict) or 'id' not in record:
                raise ValueError('旧报告案例格式无效')
            yield ('case', record)
            buffer = buffer[end:]
