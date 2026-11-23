import { test } from 'node:test'
import assert from 'node:assert/strict'
import { createWorkflowSync } from '../src/workflowSync.mjs'

test('newer backend revisions win even when network responses arrive late', () => {
  const sync = createWorkflowSync()
  const old = sync.begin('one')
  const current = sync.begin('one')
  assert.equal(sync.accept('one', { revision: 4 }, current), true)
  assert.equal(sync.accept('one', { revision: 3 }, old), false)
  assert.equal(sync.accept('one', { revision: 4 }, old), false)
  assert.equal(sync.accept('one', { revision: 5 }, old), true)
})

test('old sessions cannot overwrite the newly selected session', () => {
  const sync = createWorkflowSync()
  const previous = sync.begin('one')
  const current = sync.begin('two')
  assert.equal(sync.accept('two', { revision: 1 }, current), true)
  assert.equal(sync.accept('one', { revision: 20 }, previous), false)
})
