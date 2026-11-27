"""JSON Schema contracts for the research workflow's model and tool boundaries."""
import json

from jsonschema import Draft202012Validator


class ContractError(ValueError):
    """A response failed its declared JSON contract."""


def check(schema, value, label):
    try:
        json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ContractError(f'{label}: result is not valid JSON data ({type(exc).__name__})') from exc
    errors = list(Draft202012Validator(schema).iter_errors(value))
    if errors:
        error = errors[0]
        path = '.'.join(map(str, error.absolute_path)) or '$'
        raise ContractError(f'{label}: {path}: {error.message}')
    return value


PLAN = {'type': 'object', 'required': ['tasks'], 'properties': {'tasks': {
    'type': 'array', 'minItems': 1, 'maxItems': 12, 'items': {
        'type': 'object', 'required': ['id', 'worker', 'goal', 'depends_on'],
        'properties': {'id': {'type': 'string', 'minLength': 1},
                       'worker': {'enum': ['retrieval', 'ingestion', 'analysis', 'delivery', 'evaluation']},
                       'goal': {'type': 'string', 'minLength': 1},
                       'depends_on': {'type': 'array', 'items': {'type': 'string', 'minLength': 1}}}}}}}

MODEL = {'type': 'object', 'required': ['content', 'tool_calls', 'prompt_tokens', 'completion_tokens', 'reasoning_content'], 'properties': {
    'content': {'type': 'string'}, 'reasoning_content': {'type': ['string', 'null']},
    'prompt_tokens': {'type': 'integer', 'minimum': 0}, 'completion_tokens': {'type': 'integer', 'minimum': 0},
    'tool_calls': {'type': 'array', 'items': {
        'type': 'object', 'required': ['id', 'name', 'arguments'],
        'properties': {'id': {'type': 'string', 'minLength': 1}, 'name': {'type': 'string', 'minLength': 1},
                       'arguments': {'type': 'object'}}}}}}

RESULTS = {
    'retrieve_chunks': {'type': 'array', 'items': {'type': 'object', 'required': ['id', 'text'], 'properties': {'id': {'type': 'string', 'minLength': 1}, 'text': {'type': 'string'}}}},
    'list_corpus_papers': {'type': 'object', 'required': ['count', 'papers'], 'properties': {'count': {'type': 'integer', 'minimum': 0}, 'papers': {'type': 'array', 'items': {'type': 'object'}}}},
    'search_arxiv': {'type': 'array', 'items': {'type': 'object'}},
    'search_paperlist': {'type': 'object', 'required': ['papers'], 'properties': {'papers': {'type': 'array', 'items': {'type': 'object'}}}},
    'search_online': {'type': 'object', 'required': ['papers', 'source'], 'properties': {'papers': {'type': 'array', 'items': {'type': 'object'}}, 'source': {'enum': ['paperlist', 'arxiv']}}},
    'list_paperlist_search_results': {'type': 'object', 'required': ['papers', 'total', 'next_offset'], 'properties': {'papers': {'type': 'array', 'items': {'type': 'object'}}, 'total': {'type': 'integer', 'minimum': 0}, 'next_offset': {'type': ['integer', 'null']}}},
    'import_arxiv': {'type': 'object', 'required': ['id', 'created', 'status'], 'properties': {'id': {'type': 'string', 'minLength': 1}, 'created': {'type': 'boolean'}, 'status': {'type': 'string'}}},
    'import_paperlist': {'type': 'object', 'required': ['id', 'created', 'status'], 'properties': {'id': {'type': 'string', 'minLength': 1}, 'created': {'type': 'boolean'}, 'status': {'type': 'string'}}},
    'evaluate_metrics': {'type': 'object', 'required': ['intent_accuracy', 'rewrite_exact_match', 'retrieval_recall', 'examples', 'recall_examples'], 'properties': {'intent_accuracy': {'type': 'number'}, 'rewrite_exact_match': {'type': 'number'}, 'retrieval_recall': {'type': ['number', 'null']}, 'examples': {'type': 'integer', 'minimum': 0}, 'recall_examples': {'type': 'integer', 'minimum': 0}}},
}


def model_response(response, tools=None, *, allow_calls=True):
    try:
        calls = [{'id': c.id, 'name': c.name, 'arguments': c.arguments} for c in response.tool_calls]
        payload = {'content': response.content, 'tool_calls': calls,
                   'prompt_tokens': response.prompt_tokens, 'completion_tokens': response.completion_tokens,
                   'reasoning_content': response.reasoning_content}
    except (AttributeError, TypeError) as exc:
        raise ContractError('LLM response has invalid structure') from exc
    check(MODEL, payload, 'LLM response')
    if len({c['id'] for c in calls}) != len(calls):
        raise ContractError('LLM response contains duplicate tool call IDs')
    if not calls and not payload['content'].strip():
        raise ContractError('LLM response contains neither text nor tool calls')
    if calls and not allow_calls:
        raise ContractError('LLM tool calls are not allowed for this response')
    if tools is not None:
        schemas = {item['function']['name']: item['function']['parameters'] for item in tools}
        for call in calls:
            if call['name'] not in schemas:
                raise ContractError(f"tool {call['name']} is not available")
            check(schemas[call['name']], call['arguments'], f"tool {call['name']} arguments")
    return response


def tool_result(name, result):
    if name not in RESULTS:
        raise ContractError(f'no result schema for tool {name}')
    return check(RESULTS[name], result, f'tool {name} result')
