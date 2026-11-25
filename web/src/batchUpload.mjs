export async function uploadPapers(files, upload, onProgress) {
  const results = []
  for (const [index, file] of files.entries()) {
    onProgress(index, { status: 'uploading' })
    try {
      const result = await upload(file)
      results.push({ status: result.created ? 'created' : 'duplicate', result })
    } catch (error) {
      results.push({ status: 'failed', error: error instanceof Error ? error.message : '上传失败' })
    }
    onProgress(index, results[index])
  }
  return results
}
