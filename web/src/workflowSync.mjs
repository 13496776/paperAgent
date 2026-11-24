/** Track server revisions so delayed HTTP responses cannot restore old state. */
export function createWorkflowSync() {
  let session = ''
  let revision = -1
  let request = 0
  let appliedRequest = 0
  let firstRequest = 0
  return {
    begin(sessionId, reload = false) {
      if (session !== sessionId || reload) {
        session = sessionId
        revision = -1
        appliedRequest = 0
        firstRequest = request + 1
      }
      return ++request
    },
    accept(sessionId, workflow, requestId) {
      if (session !== sessionId || requestId < firstRequest) return false
      const nextRevision = typeof workflow?.revision === 'number' ? workflow.revision : -1
      if (nextRevision < revision || (nextRevision === revision && requestId < appliedRequest)) return false
      revision = nextRevision
      appliedRequest = requestId
      return true
    },
  }
}
