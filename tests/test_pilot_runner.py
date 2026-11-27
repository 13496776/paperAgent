import json
import time

import pytest

pytest.importorskip('fastapi')
fakeredis = pytest.importorskip('fakeredis')
jwt = pytest.importorskip('jwt')
from fastapi.testclient import TestClient

from corecoder import web
from corecoder.llm import LLMResponse, ToolCall
from corecoder.persistent import SessionStore
from corecoder.workflow.engine import ResearchTools, Supervisor
from evaluation.run_pilot import judge_case, metrics, observe, run_case


class Retriever:
    def retrieve(self, query):
        return [{'id': 'chunk-1', 'paper_id': 'paper-1', 'text': 'evidence'}]


class Model:
    def chat(self, messages, **kwargs):
        if 'research supervisor' in messages[0]['content']:
            return LLMResponse(content=json.dumps({'tasks': [{'id': 'a', 'worker': 'retrieval',
                                       'goal': 'search', 'depends_on': []}]}))
        if messages[-1]['role'] == 'tool':
            return LLMResponse(content='Answer with evidence')
        return LLMResponse(tool_calls=[ToolCall('call-1', 'retrieve_chunks', {'query': 'test'})])


def test_runner_replays_real_http_routes_and_scores(tmp_path, monkeypatch):
    key = 'test-key-that-is-long-enough-for-hs256'
    for name, value in [('CORECODER_JWT_KEY', key), ('CORECODER_JWT_ISSUER', 'issuer'),
                        ('CORECODER_JWT_AUDIENCE', 'app'), ('PAPER_EVALUATION_LOCAL_ONLY', '1')]:
        monkeypatch.setenv(name, value)
    monkeypatch.delenv('PAPER_CORPUS_PATH', raising=False)
    store = SessionStore(fakeredis.FakeRedis(), tmp_path / 'sessions.sqlite')
    monkeypatch.setattr(web, 'get_store', lambda: store)
    monkeypatch.setattr(web, 'supervisor', lambda provider, user: Supervisor(Model(), store, ResearchTools(Retriever())))
    now = int(time.time())
    token = jwt.encode({'sub': 'local-user', 'iss': 'issuer', 'aud': 'app', 'iat': now, 'exp': now + 60}, key, algorithm='HS256')
    case = {'id': 'x', 'category': '检索', 'user_turns': ['one', 'two'], 'success_criteria': ['answer'],
            'relevant_chunk_ids': ['chunk-1'], 'evidence_groups': [{'any_chunk_ids': ['chunk-1']}],
            'expected_tools': {'required_all': ['retrieve_chunks'], 'forbidden': ['search_arxiv']}}
    with TestClient(web.app, headers={'Authorization': 'Bearer ' + token}) as client:
        assert client.get('/api/settings').json()['evaluation_local_only']
        record = run_case(client, case, 'deepseek', None, {})
    assert len(record['turns']) == 2 and record['target_workflow']['status'] == 'completed'
    assert record['observation']['tool_selection_names_ok']
    assert record['observation']['labeled_evidence_hits'] == 1
    summary = metrics([record])
    assert summary['target_workflow_completed_rate']['rate'] == 1
    assert summary['tool_execution_success_rate_target_turn']['rate'] == 1
    assert summary['tool_execution_success_rate_all_turns']['denominator'] == 2
    assert summary['semantic_task_success_rate_judged_only']['rate'] is None


def test_tool_selection_zero_denominator_and_judge_validation():
    case = {'user_turns': ['q'], 'success_criteria': ['correct'], 'expected_tools': {'required_all': ['retrieve_chunks']},
            'relevant_chunk_ids': ['a'], 'evidence_groups': []}
    observed = observe(case, {'tasks': []})
    assert not observed['tool_selection_names_ok']
    class Judge:
        def chat(self, **kwargs):
            return LLMResponse(content='{"criteria":[true],"memory_checks":[],"reason":"ok"}')
    record = {'target_workflow': {'status': 'completed'}, 'turns': []}
    assert judge_case(Judge(), case, record, {})['success'] is True
    case['compression_required'] = True
    assert judge_case(Judge(), case, record, {}) is None
    assert metrics([])['tool_execution_success_rate_target_turn']['rate'] is None


def test_evaluation_mode_disables_external_tools(monkeypatch):
    monkeypatch.setenv('PAPER_EVALUATION_LOCAL_ONLY', '1')
    tools = web.get_tools('local-user')
    assert 'search_arxiv' not in [t['function']['name'] for t in tools.schemas('retrieval')]
    assert 'search_online' not in [t['function']['name'] for t in tools.schemas('retrieval')]
    assert 'search_paperlist' not in [t['function']['name'] for t in tools.schemas('retrieval')]
    assert tools.schemas('ingestion') == []


def test_streamed_report_resume_imports_legacy_and_preserves_metrics(tmp_path, monkeypatch, capsys):
    import sys

    from evaluation import run_pilot
    from evaluation.report_store import brief

    old_record = {'id': 'PC-001', 'category': 'x', 'turns': [], 'judge': {},
                  'target_workflow': {'status': 'completed', 'tasks': []},
                  'observation': {'calls': [], 'tool_selection_names_ok': True,
                                  'labeled_evidence_hits': 0, 'labeled_evidence_total': 0,
                                  'groups_hit': 0, 'groups_total': 0}}
    assert metrics([brief(old_record)]) == metrics([old_record])
    legacy = tmp_path / 'old.json'
    legacy.write_text(json.dumps({'dataset_id': 'd', 'provider': 'deepseek', 'started_at': 1,
                                 'judge_enabled': False, 'cases': [old_record]}, indent=2), encoding='utf-8')
    ds = tmp_path / 'data.json'
    ds.write_text(json.dumps({'dataset_id': 'd', 'corpus': {'paper_count': 0}, 'cases': [
        {'id': 'PC-001', 'category': 'x'}, {'id': 'PC-002', 'category': 'x'}]}), encoding='utf-8')
    evidence = tmp_path / 'evidence.json'
    evidence.write_text('{}', encoding='utf-8')
    class Client:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setattr(run_pilot.httpx, 'Client', Client)
    monkeypatch.setattr(run_pilot, 'api', lambda client, method, path: {'evaluation_local_only': True} if path == '/api/settings' else [])
    attempted = []
    def fake_run(client, case, provider, judge, evidence):
        attempted.append(case['id'])
        return {'id': case['id'], 'turns': [], 'judge': {}}
    monkeypatch.setattr(run_pilot, 'run_case', fake_run)
    monkeypatch.setenv('PAPER_EVAL_TOKEN', 'test')
    output = tmp_path / 'out'
    options = ['run_pilot', '--dataset', str(ds), '--evidence', str(evidence), '--output', str(output), '--no-judge']
    monkeypatch.setattr(sys, 'argv', options + ['--resume', str(legacy)])
    run_pilot.main()
    assert attempted == ['PC-002']
    report_dir = next(output.iterdir())
    report = json.loads((report_dir / 'report.json').read_text(encoding='utf-8'))
    assert report['metrics']['cases_attempted'] == 2
    assert json.loads((report_dir / 'cases' / 'PC-001.json').read_text(encoding='utf-8')) == old_record
    assert report['case_ids'] == ['PC-001', 'PC-002']
    assert (report_dir / 'runs' / report['runs'][0] / 'cases' / 'PC-002.json').is_file()
    assert (report_dir / 'review.html').is_file()
    monkeypatch.setattr(sys, 'argv', options + ['--resume', str(report_dir)])
    run_pilot.main()
    assert attempted == ['PC-002']


def test_resume_sparse_ids_keeps_old_results_and_separates_new_run(tmp_path, monkeypatch):
    import sys

    from evaluation import run_pilot
    from evaluation.report_store import atomic_json
    from evaluation.review_report import make_review

    directory = tmp_path / 'report'
    (directory / 'cases').mkdir(parents=True)
    atomic_json(directory / 'report.json', {'dataset_id': 'd', 'provider': 'deepseek', 'judge_enabled': False})
    atomic_json(directory / 'cases' / 'PC-002.json', {'id': 'PC-002', 'category': 'x', 'turns': [], 'judge': {}})
    data = tmp_path / 'dataset.json'
    data.write_text(json.dumps({'dataset_id': 'd', 'corpus': {'paper_count': 0}, 'cases': [
        {'id': f'PC-00{i}', 'category': 'x'} for i in range(1, 4)]}), encoding='utf-8')
    evidence = tmp_path / 'evidence.json'
    evidence.write_text('{}', encoding='utf-8')
    class Client:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setattr(run_pilot.httpx, 'Client', Client)
    monkeypatch.setattr(run_pilot, 'api', lambda c, m, p: {'evaluation_local_only': True} if p == '/api/settings' else [])
    called = []
    def fake_case(client, case, provider, judge, evidence):
        called.append(case['id'])
        return {'id': case['id'], 'category': case['category'], 'turns': [], 'judge': {}}
    monkeypatch.setattr(run_pilot, 'run_case', fake_case)
    monkeypatch.setenv('PAPER_EVAL_TOKEN', 'test')
    monkeypatch.setattr(sys, 'argv', ['run_pilot', '--dataset', str(data), '--evidence', str(evidence),
                                     '--no-judge', '--resume', str(directory)])
    run_pilot.main()
    assert called == ['PC-001', 'PC-003']
    report = json.loads((directory / 'report.json').read_text(encoding='utf-8'))
    run = directory / 'runs' / report['runs'][0]
    manifest = json.loads((run / 'run.json').read_text(encoding='utf-8'))
    assert manifest['skipped_existing_ids'] == ['PC-002']
    assert manifest['planned_case_ids'] == ['PC-001', 'PC-003']
    assert manifest['new_case_ids'] == ['PC-001', 'PC-003']
    assert not (run / 'cases' / 'PC-002.json').exists()
    assert make_review(directory)[0] == 3
    assert 'PC-001' in (directory / 'review.csv').read_text(encoding='utf-8-sig')


def test_windows_replace_retries_only_expected_errors(tmp_path, monkeypatch):
    from corecoder import persistent
    source, target = tmp_path / 'new', tmp_path / 'target'
    source.write_text('new')
    original = persistent.os.replace
    attempts = []
    def flaky(src, dst):
        attempts.append(1)
        if len(attempts) < 3:
            exc = PermissionError('locked')
            exc.winerror = 5
            raise exc
        original(src, dst)
    monkeypatch.setattr(persistent.os, 'replace', flaky)
    monkeypatch.setattr(persistent.time, 'sleep', lambda _: None)
    persistent._replace_workflow_file(source, target)
    assert len(attempts) == 3 and target.read_text() == 'new'
    def permanent(src, dst):
        raise PermissionError('ordinary permission failure')
    monkeypatch.setattr(persistent.os, 'replace', permanent)
    with pytest.raises(PermissionError):
        persistent._replace_workflow_file(source, target)


def test_legacy_resume_with_first_163_does_not_replay_them(tmp_path, monkeypatch, capsys):
    import sys

    from evaluation import run_pilot
    records = [{'id': f'PC-{i:03}', 'category': 'test', 'turns': [], 'judge': {}} for i in range(1, 164)]
    old = tmp_path / 'old.json'
    old.write_text(json.dumps({'dataset_id': 'd', 'provider': 'deepseek', 'judge_enabled': False,
                               'started_at': 1, 'cases': records}, indent=2), encoding='utf-8')
    dataset = tmp_path / 'data.json'
    dataset.write_text(json.dumps({'dataset_id': 'd', 'corpus': {'paper_count': 0}, 'cases': [
        {'id': f'PC-{i:03}', 'category': 'test'} for i in range(1, 165)]}), encoding='utf-8')
    evidence = tmp_path / 'evidence.json'
    evidence.write_text('{}', encoding='utf-8')
    class Client:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
    monkeypatch.setattr(run_pilot.httpx, 'Client', Client)
    monkeypatch.setattr(run_pilot, 'api', lambda c, m, p: {'evaluation_local_only': True} if p == '/api/settings' else [])
    calls = []
    def fake_case(client, case, provider, judge, evidence):
        calls.append(case['id'])
        return {'id': case['id'], 'category': 'test', 'turns': [], 'judge': {}}
    monkeypatch.setattr(run_pilot, 'run_case', fake_case)
    monkeypatch.setenv('PAPER_EVAL_TOKEN', 'test')
    monkeypatch.setattr(sys, 'argv', ['run_pilot', '--dataset', str(dataset), '--evidence', str(evidence),
                                     '--output', str(tmp_path / 'out'), '--no-judge', '--resume', str(old)])
    run_pilot.main()
    assert calls == ['PC-164']
    output = capsys.readouterr().out
    assert '已有 163 个案例' in output
    assert '运行 PC-164' in output
    assert '运行 PC-001' not in output


def test_api_reports_http_status_and_bounded_response_detail():
    import httpx

    from evaluation.run_pilot import api
    transport = httpx.MockTransport(lambda request: httpx.Response(400, json={'detail': 'dependencies must be a list of task IDs'}))
    with httpx.Client(base_url='http://test', transport=transport) as client, pytest.raises(
            RuntimeError, match='HTTP 400.*dependencies must be a list'):
        api(client, 'POST', '/plan', {})


def test_structured_retrieval_results_are_included_in_evaluation_hits():
    from evaluation.run_pilot import returned_chunks
    result = {'hits': [{'id': 'chunk-1', 'text': 'evidence'}], 'coverage': {'pending_count': 2}}
    calls = [{'name': 'retrieve_chunks', 'status': 'completed', 'result': result}]
    assert returned_chunks(calls) == result['hits']
