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
    assert tools.schemas('ingestion') == []
