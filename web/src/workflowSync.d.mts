export function createWorkflowSync(): {
  begin(sessionId: string): number
  accept(sessionId: string, workflow: { revision?: number } | null, requestId: number): boolean
}
