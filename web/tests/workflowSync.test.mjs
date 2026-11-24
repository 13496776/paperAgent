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

test('explicit reload accepts an archived result after a newer live workflow expired', () => {
  const sync = createWorkflowSync()
  const live = sync.begin('one')
  assert.equal(sync.accept('one', { revision: 9 }, live), true)
  const delayed = sync.begin('one')
  const reload = sync.begin('one', true)
  assert.equal(sync.accept('one', { revision: 5 }, reload), true)
  assert.equal(sync.accept('one', { revision: 9 }, delayed), false)
})
