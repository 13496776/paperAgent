"""Validate the focused evaluation set's provenance and task design."""
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'evaluation' / 'paperagent_pilot'


def test_focused_dialogues_have_distinct_questions_and_real_targets():
    data = json.loads((ROOT / 'pilot_dialogues.json').read_text(encoding='utf-8'))
    evidence = json.loads((ROOT / 'evidence_index.json').read_text(encoding='utf-8'))
    cases = data['cases']
    assert data['dataset_id'] == 'paperagent-focused-100-v1'
    assert len(cases) == len({case['id'] for case in cases}) == 100
    assert collections.Counter(case['category'] for case in cases) == {
        '混合检索': 34, '指代消解': 33, '会话压缩一致性': 33}
    turns = [turn for case in cases for turn in case['user_turns']]
    assert len(turns) == len(set(turns)), '对话中不应重复同一句用户消息'
    for case in cases:
        assert len(case['user_turns']) >= 2
        assert case['scored_turn'] == len(case['user_turns'])
        assert case['success_criteria'] and case['target_papers']
        targets = {paper['id'] for paper in case['target_papers']}
        assert set(case['required_ready_paper_ids']) == targets
        assert set(case['required_paper_ids']) <= targets
        assert set(case['relevant_chunk_ids']) == {
            chunk for group in case['evidence_groups'] for chunk in group['any_chunk_ids']}
        for group in case['evidence_groups']:
            for chunk_id in group['any_chunk_ids']:
                assert evidence[chunk_id]['paper_id'] == group['paper_id']
        if case['category'] == '混合检索':
            assert 'retrieve_chunks' in case['expected_tools']['required_all']
        if case['category'] == '指代消解':
            assert case['expected_rewrite']
        if case['category'] == '会话压缩一致性':
            assert len(case['user_turns']) == 8
            assert case['compression_required'] and len(case['memory_checks']) >= 4
    for chunk in evidence.values():
        assert chunk['verified_anchor_quote'].casefold() in chunk['excerpt'].casefold()
