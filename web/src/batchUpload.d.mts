export type UploadResult = { created: boolean; title: string; chunks: number }
export type UploadStatus =
  | { status: 'uploading' }
  | { status: 'created' | 'duplicate'; result: UploadResult }
  | { status: 'failed'; error: string }

export function uploadPapers(
  files: File[],
  upload: (file: File) => Promise<UploadResult>,
  onProgress: (index: number, status: UploadStatus) => void,
): Promise<Exclude<UploadStatus, { status: 'uploading' }>[]>
