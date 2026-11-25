<script setup lang="ts">
import { onMounted, onUnmounted, ref, watch } from 'vue'
import { createWorkflowSync } from './workflowSync.mjs'
import { uploadPapers, type UploadStatus } from './batchUpload.mjs'
type Activity = { stage: string; detail: string; status: string; started_at: number; duration_ms?: number; elapsed_ms?: number }
type Task = { id: string; worker: string; goal: string; depends_on: string[]; status: string; result?: string; error?: string; activities?: Activity[]; calls?: {id?: string; name?: string; status: string; error?: string}[] }
type Workflow = { question: string; status: string; tasks: Task[]; revision?: number; result?: string; error?: string; trace?: unknown[]; control_requested?: string }
const token = ref('')
function initialSession() {
  try { return sessionStorage.getItem('paperagent_session') || crypto.randomUUID() }
  catch { return crypto.randomUUID() }
}
const session = ref(initialSession())
const sync = createWorkflowSync()
const provider = ref('deepseek')
const question = ref('')
const workflow = ref<Workflow | null>(null)
const error = ref('')
const busy = ref(false)
const now = ref(Date.now())
const controlBusy = ref(false)
const confirming = ref(false)
const syncError = ref('')
let pollInFlight = false
let pollTimer: ReturnType<typeof setInterval> | undefined
const corpusBusy = ref(false)
const pdfs = ref<File[]>([])
const pdfInput = ref<HTMLInputElement | null>(null)
const batchSelection = ref(false)
const uploadResults = ref<{ name: string; status: 'pending' | UploadStatus['status']; detail: string }[]>([])
const ccfRank = ref('')
const paperTitle = ref('')
const paperAuthors = ref('')
const paperDoi = ref('')
const paperVenue = ref('')
const paperYear = ref('')
const paperMonth = ref('')
const ccfSource = ref('')
const ccfVersion = ref('')
const ccfVerified = ref(false)
const searchQuery = ref('')
const searchProvider = ref<'arxiv' | 'paperlist'>('arxiv')
const searchProviderSelected = ref(false)
const paperlistCursor = ref<string | null>(null)
const papers = ref<{id: string; title: string; authors: string | string[]; doi: string; source: string; year?: number | null; month?: number | null; date_source?: string; page_count?: number | null; ccf_rank?: string | null; status?: string; error?: string | null}[]>([])
const corpusLoaded = ref(false)
const results = ref<{id: string; title: string; authors: string[]; abstract: string; provider: 'arxiv' | 'paperlist'; url?: string}[]>([])
const notice = ref('')
watch([searchQuery, searchProvider], () => { results.value = []; paperlistCursor.value = null })
onMounted(async () => {
  try {
    token.value = sessionStorage.getItem('paperagent_token') || ''
  } catch {
    // The token can still be entered manually if browser storage is unavailable.
  }
  const params = new URLSearchParams(window.location.hash.slice(1))
  const code = params.get('local-session')
  if (code) {
    token.value = ''
    try { sessionStorage.removeItem('paperagent_token') } catch { /* Storage is optional. */ }
    history.replaceState(null, '', window.location.pathname + window.location.search)
    try {
      const response = await fetch('/api/local-session', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ code }),
      })
      if (!response.ok) throw new Error('本地登录链接已失效，请重新启动程序或手动粘贴终端中的令牌。')
      token.value = (await response.json()).token
      try { sessionStorage.setItem('paperagent_token', token.value) } catch { /* Manual entry remains available. */ }
    } catch (e) {
      error.value = e instanceof Error ? e.message : '本地登录失败，请检查后端是否启动。'
    }
  }
  if (token.value) void refreshWorkflow()
  pollTimer = setInterval(() => {
    now.value = Date.now()
    if (token.value && (busy.value || ['planning', 'running'].includes(workflow.value?.status || ''))) {
      void refreshWorkflow()
    }
  }, 1500)
  try {
    const response = await fetch('/api/settings')
    if (response.ok) {
      const settings = await response.json()
      provider.value = settings.default_provider
      if (settings.paperlist_enabled && !searchProviderSelected.value && !corpusBusy.value && !searchQuery.value) searchProvider.value = 'paperlist'
    }
  } catch {
    // The model selector remains available if the backend is temporarily unavailable.
  }
})
async function corpusRequest(path: string, body?: BodyInit, contentType?: string, method?: string) {
  const headers: Record<string, string> = { Authorization: `Bearer ${token.value}` }
  if (contentType) headers['Content-Type'] = contentType
  const response = await fetch(`/api/corpus${path}`, { method: method || (body ? 'POST' : 'GET'), headers, body })
  const data = await response.json().catch(() => ({ detail: `服务请求失败（HTTP ${response.status}）` }))
  if (!response.ok) throw new Error(response.status === 401 ? '访问令牌无效或已过期，请重新启动程序或粘贴有效令牌。' : data.detail || '请求失败')
  return data
}
async function corpusAction(action: 'load' | 'upload' | 'search' | 'search_more' | 'import' | 'delete', id = '') {
  corpusBusy.value = true; error.value = ''; notice.value = ''
  try {
    if (action === 'upload' && pdfs.value.length) {
      const selected = [...pdfs.value]
      const isBatch = batchSelection.value
      const metadata = {
        ccf_rank: ccfRank.value, title: paperTitle.value, authors: paperAuthors.value, doi: paperDoi.value,
        venue: paperVenue.value, year: paperYear.value.trim(), month: paperMonth.value.trim(),
        ccf_source: ccfSource.value, ccf_version: ccfVersion.value, ccf_verified: String(ccfVerified.value),
      }
      uploadResults.value = selected.map(file => ({ name: file.name, status: 'pending', detail: '等待上传' }))
      const outcomes = await uploadPapers(selected, async file => {
        const form = new FormData()
        form.append('file', file)
        if (!isBatch) {
          for (const [key, value] of Object.entries(metadata)) {
            if (value || (key !== 'year' && key !== 'month')) form.append(key, value)
          }
        }
        return corpusRequest('/upload', form)
      }, (index, status) => {
        let detail = '正在上传和建库'
        if (status.status === 'created') detail = `已导入，生成 ${status.result.chunks} 个分块`
        if (status.status === 'duplicate') detail = '该论文已经导入'
        if (status.status === 'failed') detail = `失败：${status.error}`
        uploadResults.value[index] = { name: selected[index].name, status: status.status, detail }
      })
      const failed = selected.filter((_, index) => outcomes[index].status === 'failed')
      pdfs.value = failed
      if (pdfInput.value) pdfInput.value.value = ''
      notice.value = `处理完成，共 ${selected.length} 个文件，${failed.length} 个失败${failed.length ? '，可重试失败文件' : ''}`
      if (!failed.length) {
        batchSelection.value = false
        ccfRank.value = ''; paperTitle.value = ''; paperAuthors.value = ''; paperDoi.value = ''
        paperVenue.value = ''; paperYear.value = ''; paperMonth.value = ''; ccfSource.value = ''; ccfVersion.value = ''; ccfVerified.value = false
      }
    }
    if (action === 'delete') {
      await corpusRequest(`/${encodeURIComponent(id)}`, undefined, undefined, 'DELETE')
      notice.value = '论文及其向量已删除'
    }
    if (action === 'search' || action === 'search_more') {
      if (searchProvider.value === 'paperlist') {
        const body = { query: searchQuery.value, ...(action === 'search_more' ? { cursor: paperlistCursor.value } : {}) }
        const response = await corpusRequest('/paperlist/search', JSON.stringify(body), 'application/json')
        const next = response.papers.map((item: typeof results.value[number]) => ({ ...item, provider: 'paperlist' as const }))
        results.value = action === 'search_more' ? [...results.value, ...next] : next
        paperlistCursor.value = response.next_cursor
        if (response.note) notice.value = response.note
      } else {
        const response = await corpusRequest('/search', JSON.stringify({query: searchQuery.value}), 'application/json')
        results.value = response.map((item: typeof results.value[number]) => ({ ...item, provider: 'arxiv' as const }))
      }
      if (!results.value.length) notice.value = '没有找到匹配的论文'
    }
    if (action === 'import') {
      const result = await corpusRequest('/import', JSON.stringify({id}), 'application/json')
      notice.value = result.created ? `已导入 ${result.title}，生成 ${result.chunks} 个分块` : '该论文已经导入'
    }
    if (action !== 'search' && action !== 'search_more') {
      papers.value = await corpusRequest('')
      corpusLoaded.value = true
    }
  } catch (e) { error.value = e instanceof Error ? e.message : '语料库请求失败' }
  finally { corpusBusy.value = false }
}
function deletePaper(id: string, title: string) {
  if (window.confirm(`确认删除论文《${title}》及其检索索引？`)) void corpusAction('delete', id)
}
function choosePdf(event: Event) {
  pdfs.value = Array.from((event.target as HTMLInputElement).files || [])
  batchSelection.value = pdfs.value.length > 1
  uploadResults.value = []
}
function elapsed(activity: Activity) {
  const milliseconds = activity.duration_ms ?? activity.elapsed_ms ?? Math.max(0, now.value - activity.started_at * 1000)
  return (milliseconds / 1000).toFixed(1) + ' 秒'
}
const names: Record<string,string> = { ingestion: '数据接入', analysis: '推理分析', retrieval: '文献检索', delivery: '生成交付', evaluation: '测评运维' }
const labels: Record<string,string> = { pending: '等待中', running: '执行中', completed: '已完成', failed: '失败', awaiting_approval: '等待确认', cancelled: '已取消', paused: '已暂停', planning: '规划中', interrupted: '已中断' }
async function request(path: string, body?: object, reload = false) {
  const sessionId = session.value
  const requestId = sync.begin(sessionId, reload)
  const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}${path}`, {
    method: body ? 'POST' : 'GET', headers: { Authorization: `Bearer ${token.value}`, 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
    signal: body ? undefined : AbortSignal.timeout(10000),
  })
  const data = await response.json().catch(() => ({ detail: `服务请求失败（HTTP ${response.status}）` }))
  if (!response.ok) throw new Error(response.status === 401 ? '访问令牌无效或已过期，请重新启动程序或粘贴有效令牌。' : data.detail || '请求失败')
  return { data, sessionId, requestId }
}
function applyWorkflow(result: { sessionId: string; requestId: number }, next: Workflow | null) {
  if (session.value !== result.sessionId || !sync.accept(result.sessionId, next, result.requestId)) return
  workflow.value = next
  if (next && next.status !== 'awaiting_approval') confirming.value = false
}
watch(session, (id) => {
  sync.begin(id)
  workflow.value = null
  syncError.value = ''
  confirming.value = false
  try { sessionStorage.setItem('paperagent_session', id) } catch { /* Session ID can be entered manually. */ }
}, { flush: 'sync', immediate: true })
onUnmounted(() => { if (pollTimer) clearInterval(pollTimer) })
async function refreshWorkflow() {
  if (pollInFlight) return
  pollInFlight = true
  const expectedSession = session.value
  try {
    const result = await request('')
    if (session.value !== result.sessionId) return
    if (result.data.workflow) {
      applyWorkflow(result, result.data.workflow)
      syncError.value = ''
    } else if (workflow.value) {
      syncError.value = '后端未找到该会话的任务状态，请检查会话 ID 或会话是否已过期。'
    }
  } catch (e) {
    if (session.value === expectedSession) {
      syncError.value = `无法同步后端状态：${e instanceof Error ? e.message : '网络连接失败'}`
    }
  } finally {
    pollInFlight = false
  }
}
async function control(action: 'pause' | 'cancel') {
  controlBusy.value = true; error.value = ''
  try {
    const result = await request(`/${action}`, {})
    applyWorkflow(result, result.data)
  } catch (e) {
    error.value = e instanceof Error ? e.message : '控制请求失败'
    await refreshWorkflow()
  } finally {
    controlBusy.value = false
  }
}
async function act(action: 'plan' | 'resume' | 'load') {
  busy.value = true; error.value = ''; syncError.value = ''
  if (action === 'resume') confirming.value = true
  try {
    if (action === 'plan') {
      const result = await request('/plan', { question: question.value, provider: provider.value })
      applyWorkflow(result, result.data)
      question.value = ''
    } else if (action === 'resume') {
      const result = await request('/resume', { provider: provider.value })
      applyWorkflow(result, result.data)
    }
    const result = await request('', undefined, action === 'load')
    applyWorkflow(result, result.data.workflow || null)
  } catch (e) {
    error.value = e instanceof Error ? e.message : '请求失败'
    await refreshWorkflow()
  } finally {
    confirming.value = false
    busy.value = false
  }
}
</script>
<template>
  <div class="layout"><aside class="sidebar"><div class="brand"><span class="mark">P</span><div>Paper Lab<small>RESEARCH WORKSPACE</small></div></div><div class="side-label">工作空间</div><div class="nav">◫ &nbsp; 研究工作台</div><div class="side-bottom">● &nbsp; 本地研究环境</div></aside>
  <main><header><div><div class="eyebrow">RESEARCH WORKSPACE</div><h1>研究工作台</h1><p>规划研究任务，追踪证据与执行进度。</p></div><span class="header-tag">SUPERVISOR / WORKERS</span></header>
  <div class="workspace"><section class="primary">
    <div class="card"><div class="card-head"><div><h2>会话连接</h2><p>使用一键启动时会自动连接；也可手动输入访问令牌。</p></div></div><div class="access-fields"><label>访问令牌<input v-model="token" type="password" placeholder="JWT Bearer token" autocomplete="off" :disabled="corpusBusy"></label><label>会话 ID<input v-model="session" :disabled="busy || ['planning', 'running'].includes(workflow?.status || '')"></label><button class="secondary" :disabled="busy || !token || !session" @click="act('load')">加载会话</button></div></div>
    <div class="card"><div class="card-head"><div><h2>论文语料库</h2><p>上传 PDF 自动提取与分块，或检索并导入 arXiv 论文。</p></div><button class="secondary" :disabled="corpusBusy || !token" @click="corpusAction('load')">查看已导入</button></div>
      <div class="corpus-row"><input ref="pdfInput" type="file" accept="application/pdf,.pdf" multiple :disabled="corpusBusy" @change="choosePdf"><select v-if="!batchSelection" v-model="ccfRank" aria-label="CCF 等级"><option value="">CCF 未核验</option><option value="A">CCF-A</option><option value="B">CCF-B</option><option value="C">CCF-C</option><option value="无类别">无类别</option></select><button :disabled="corpusBusy || !token || !pdfs.length" @click="corpusAction('upload')">{{ uploadResults.some(item => item.status === 'failed') ? `重试失败文件（${pdfs.length}）` : '上传并建库' }}</button></div>
      <p v-if="batchSelection" class="corpus-notice">批量上传将从各 PDF 提取元数据。单篇手工填写的元数据不会应用到批量文件。</p>
      <details v-if="!batchSelection"><summary>补充论文元数据（可选）</summary><div class="metadata-fields"><label>标题<input v-model="paperTitle"></label><label>作者<input v-model="paperAuthors" placeholder="多位作者用逗号分隔"></label><label>DOI<input v-model="paperDoi"></label><label>期刊或会议<input v-model="paperVenue"></label><label>发表年份<input v-model="paperYear" inputmode="numeric" placeholder="例如 2024"></label><label>发表月份<input v-model="paperMonth" inputmode="numeric" placeholder="1–12，需填写年份"></label><label>CCF 来源<input v-model="ccfSource"></label><label>CCF 版本<input v-model="ccfVersion"></label><label class="checkbox"><input v-model="ccfVerified" type="checkbox"> 已根据来源和版本核验 CCF 等级</label></div></details>
      <ul v-if="uploadResults.length" class="upload-results" aria-live="polite"><li v-for="(item, index) in uploadResults" :key="index" :class="{ 'upload-failed': item.status === 'failed' }"><span>{{ item.name }}</span><span>{{ item.detail }}</span></li></ul>
      <div class="corpus-row"><select v-model="searchProvider" aria-label="在线搜索来源" :disabled="corpusBusy" @change="searchProviderSelected = true"><option value="arxiv">arXiv</option><option value="paperlist">Paperlist</option></select><input v-model="searchQuery" placeholder="搜索在线论文" :disabled="corpusBusy"><button class="secondary" :disabled="corpusBusy || !token || !searchQuery.trim()" @click="corpusAction('search')">在线检索</button></div>
      <p v-if="searchProvider === 'paperlist'" class="corpus-notice">Paperlist 搜索会议论文，需要配置 Paperlist API Key。</p>
      <p v-if="notice" class="corpus-notice">{{ notice }}</p><p v-if="corpusLoaded" class="corpus-notice">当前个人语料已导入 {{ papers.length }} 篇论文。</p>
      <div v-for="item in results" :key="item.id" class="corpus-item"><div><strong>{{ item.title }}</strong><small>{{ item.authors.join('、') }} · {{ item.id }}</small><p>{{ item.abstract }}</p><a v-if="item.provider === 'paperlist' && item.url" :href="item.url" target="_blank" rel="noopener noreferrer">查看论文</a></div><button v-if="item.provider === 'arxiv'" class="secondary" :disabled="corpusBusy" @click="corpusAction('import', item.id)">导入</button></div>
      <button v-if="searchProvider === 'paperlist' && paperlistCursor" class="secondary" :disabled="corpusBusy" @click="corpusAction('search_more')">加载更多</button>
      <details v-if="papers.length"><summary>已导入 {{ papers.length }} 篇论文</summary><div v-for="paper in papers" :key="paper.id" class="corpus-item"><div><strong>{{ paper.title }}</strong><small>{{ (Array.isArray(paper.authors) ? paper.authors.join('、') : paper.authors) || '作者未知' }} · {{ paper.doi || 'DOI 未知' }}</small><small>{{ paper.year ? (paper.date_source === 'arxiv_published' ? 'arXiv 首次公开 ' : '发表 ') + paper.year + (paper.month ? '-' + String(paper.month).padStart(2, '0') : '') : '日期未知' }} · {{ paper.source }} · {{ paper.page_count ? paper.page_count + ' 页' : '页数未知' }} · {{ paper.ccf_rank ? 'CCF ' + paper.ccf_rank : 'CCF 未核验' }} · {{ paper.status || '可用' }}</small><small v-if="paper.error">{{ paper.error }}</small></div><button class="secondary" :disabled="corpusBusy" @click="deletePaper(paper.id, paper.title)">删除</button></div></details>
    </div>
    <div class="card"><div class="card-head"><div><h2>开始研究</h2><p>描述你的研究问题，先生成可审阅的任务计划。</p></div><select v-model="provider"><option value="deepseek">DeepSeek</option><option value="openai">OpenAI</option></select></div><textarea v-model="question" placeholder="例如：比较混合检索与重排序的作用，并整理证据来源。"></textarea><div class="prompt-footer"><span>计划生成后需要确认才会执行</span><button :disabled="busy || !token || !question.trim()" @click="act('plan')">{{ busy ? '处理中…' : '生成计划 →' }}</button></div></div>
    <div v-if="error" class="error">{{ error }}</div><div v-if="syncError" class="error">{{ syncError }}</div>
    <div v-if="workflow" class="card"><div class="card-head"><div><div class="eyebrow">WORKFLOW</div><h2>任务计划</h2><p>{{ workflow.question }}</p></div><span class="status">{{ workflow.control_requested === 'cancel' ? '正在取消' : workflow.control_requested === 'pause' ? '正在暂停' : confirming && workflow.status === 'awaiting_approval' ? '正在提交执行请求' : labels[workflow.status] || workflow.status }}</span></div><div v-for="task in workflow.tasks" :key="task.id" class="task"><div class="task-indicator" :class="task.status"></div><div class="task-body"><div class="task-top"><strong>{{ names[task.worker] || task.worker }}</strong><span>{{ labels[task.status] || task.status }}</span></div><p>{{ task.goal }}</p><small v-if="task.depends_on.length">依赖 {{ task.depends_on.join('、') }}</small><p v-if="task.status === 'running' && task.activities?.length" class="activity-current">{{ task.activities[task.activities.length - 1].detail }} · {{ elapsed(task.activities[task.activities.length - 1]) }}</p><details v-if="task.activities?.length"><summary>查看步骤耗时</summary><div v-for="(activity, index) in task.activities" :key="index" class="activity-row">{{ activity.detail }} · {{ elapsed(activity) }} · {{ labels[activity.status] || activity.status }}</div></details><p v-if="task.error" class="error">子任务失败原因：{{ task.error }}</p><div v-for="(call, index) in (task.calls || []).filter(item => item.status === 'failed')" :key="call.id || index" class="error">工具 {{ call.name || '未知工具' }} 调用失败：{{ call.error || '未提供错误详情' }}</div><details v-if="task.result"><summary>查看结果</summary><pre>{{ task.result }}</pre></details></div></div><details v-if="workflow.trace?.length"><summary>查看执行追踪</summary><pre>{{ JSON.stringify(workflow.trace, null, 2) }}</pre></details><div v-if="workflow.error" class="error" role="alert">任务停止原因：{{ workflow.error }}</div><div v-else-if="workflow.status === 'failed'" class="error" role="alert">任务执行失败，后端没有提供详细原因，请检查服务日志。</div><p v-if="workflow.status === 'awaiting_approval' && !confirming" class="workflow-hint">计划已生成，请确认执行或取消计划。</p><p v-if="confirming && workflow.status === 'awaiting_approval'" class="workflow-hint">正在向后端提交执行请求，等待状态更新。</p><p v-if="workflow.control_requested" class="workflow-hint">请求已收到，正在等待当前模型或工具调用结束。</p><div v-if="workflow.status !== 'completed' && workflow.status !== 'cancelled'" class="actions"><button class="secondary" :disabled="controlBusy || !!workflow.control_requested" @click="control('cancel')">{{ workflow.status === 'running' ? '取消任务' : '取消计划' }}</button><button v-if="workflow.status === 'running'" class="secondary" :disabled="controlBusy || !!workflow.control_requested" @click="control('pause')">暂停</button><button v-if="['awaiting_approval', 'paused', 'failed'].includes(workflow.status) && workflow.tasks.length" :disabled="busy || controlBusy" @click="act('resume')">{{ workflow.status === 'awaiting_approval' ? '确认并执行 →' : '继续执行 →' }}</button></div><div v-if="workflow.result" class="result"><div class="eyebrow">RESEARCH OUTPUT</div><h3>研究结果</h3><pre>{{ workflow.result }}</pre></div></div>
  </section><aside class="right"><div class="card"><div class="eyebrow">AGENT TEAM</div><h2>研究团队</h2><p class="muted">每个角色负责不同的研究环节。</p><div v-for="(name, key) in names" :key="key" class="worker"><span class="worker-icon">✳</span><div><strong>{{ name }}</strong><small>{{ key }}</small></div></div></div><div class="note"><strong>会话与追踪</strong><p>执行计划和阶段结果保存在当前会话中。加载相同会话可继续查看。</p></div></aside></div></main></div>
</template>
