import { test } from 'node:test'
import assert from 'node:assert/strict'
import { uploadPapers } from '../src/batchUpload.mjs'

test('uploads files sequentially and retains individual successes, duplicates and failures', async () => {
  const files = [{ name: 'one.pdf' }, { name: 'bad.pdf' }, { name: 'duplicate.pdf' }]
  const progress = []
  let active = 0
  const results = await uploadPapers(files, async file => {
    assert.equal(++active, 1)
    await Promise.resolve()
    active--
    if (file.name === 'bad.pdf') throw new Error('无效 PDF')
    return { created: file.name !== 'duplicate.pdf', title: file.name, chunks: 2 }
  }, (index, status) => progress.push([index, status.status]))
  assert.deepEqual(results.map(item => item.status), ['created', 'failed', 'duplicate'])
  assert.equal(results[1].error, '无效 PDF')
  assert.deepEqual(progress, [[0, 'uploading'], [0, 'created'], [1, 'uploading'], [1, 'failed'],
    [2, 'uploading'], [2, 'duplicate']])
  const failed = files.filter((_, index) => results[index].status === 'failed')
  assert.deepEqual(failed.map(file => file.name), ['bad.pdf'])
})
