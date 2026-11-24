export function createWorkflowSync(): {
  begin(sessionId: string, reload?: boolean): number
  accept(sessionId: string, workflow: { revision?: number } | null, requestId: number): boolean
}
