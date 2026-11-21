<script setup lang="ts">
import { onMounted, ref } from 'vue'
type Task = { id: string; worker: string; goal: string; depends_on: string[]; status: string; result?: string }
type Workflow = { question: string; status: string; tasks: Task[]; result?: string; error?: string; trace?: unknown[] }
const token = ref('')
const session = ref(crypto.randomUUID())
const provider = ref('openai')
onMounted(async () => {
  try {
    const response = await fetch('/api/settings')
    if (response.ok) provider.value = (await response.json()).default_provider
  } catch {
    // The model selector remains available if the backend is temporarily unavailable.
  }
})
const question = ref('')
const workflow = ref<Workflow | null>(null)
const error = ref('')
const busy = ref(false)
const corpusBusy = ref(false)
const pdf = ref<File | null>(null)
const searchQuery = ref('')
const papers = ref<{id: string; title: string; source: string}[]>([])
const results = ref<{id: string; title: string; authors: string[]; abstract: string}[]>([])
const notice = ref('')
async function corpusRequest(path: string, body?: BodyInit, contentType?: string) {
  const headers: Record<string, string> = { Authorization: `Bearer ${token.value}` }
  if (contentType) headers['Content-Type'] = contentType
  const response = await fetch(`/api/corpus${path}`, { method: body ? 'POST' : 'GET', headers, body })
  const data = await response.json()
  if (!response.ok) throw new Error(data.detail || '请求失败')
  return data
}
async function corpusAction(action: 'load' | 'upload' | 'search' | 'import', id = '') {
  corpusBusy.value = true; error.value = ''; notice.value = ''
  try {
    if (action === 'upload' && pdf.value) {
      const form = new FormData(); form.append('file', pdf.value)
      const result = await corpusRequest('/upload', form)
      notice.value = result.created ? `已导入 ${result.title}，生成 ${result.chunks} 个分块` : '该论文已经导入'
      pdf.value = null
    }
    if (action === 'search') results.value = await corpusRequest('/search', JSON.stringify({query: searchQuery.value}), 'application/json')
    if (action === 'import') {
      const result = await corpusRequest('/import', JSON.stringify({id}), 'application/json')
      notice.value = result.created ? `已导入 ${result.title}，生成 ${result.chunks} 个分块` : '该论文已经导入'
    }
    if (action !== 'search') papers.value = await corpusRequest('')
  } catch (e) { error.value = e instanceof Error ? e.message : '语料库请求失败' }
  finally { corpusBusy.value = false }
}
function choosePdf(event: Event) {
  pdf.value = (event.target as HTMLInputElement).files?.[0] || null
}
const names: Record<string,string> = { ingestion: '数据接入', analysis: '推理分析', retrieval: '文献检索', delivery: '生成交付', evaluation: '测评运维' }
const labels: Record<string,string> = { pending: '等待中', running: '执行中', completed: '已完成', failed: '失败', awaiting_approval: '等待确认', cancelled: '已取消', paused: '已暂停', planning: '规划中' }
async function request(path: string, body?: object) {
  const response = await fetch(`/api/sessions/${encodeURIComponent(session.value)}${path}`, {
    method: body ? 'POST' : 'GET', headers: { Authorization: `Bearer ${token.value}`, 'Content-Type': 'application/json' }, body: body ? JSON.stringify(body) : undefined,
  })
  const data = await response.json()
  if (!response.ok) throw new Error(data.detail || '请求失败')
  return data
}
async function act(action: 'plan' | 'resume' | 'cancel' | 'load') {
  busy.value = true; error.value = ''
  try {
    if (action === 'plan') { workflow.value = await request('/plan', { question: question.value, provider: provider.value }); question.value = '' }
    else if (action === 'resume') workflow.value = await request('/resume', { provider: provider.value })
    else if (action === 'cancel') workflow.value = await request('/cancel', {})
    const data = await request(''); workflow.value = data.workflow || null
  } catch (e) { error.value = e instanceof Error ? e.message : '请求失败' }
  finally { busy.value = false }
}
</script>
<template>
  <div class="layout"><aside class="sidebar"><div class="brand"><span class="mark">P</span><div>Paper Lab<small>RESEARCH WORKSPACE</small></div></div><div class="side-label">工作空间</div><div class="nav">◫ &nbsp; 研究工作台</div><div class="side-bottom">● &nbsp; 本地研究环境</div></aside>
  <main><header><div><div class="eyebrow">RESEARCH WORKSPACE</div><h1>研究工作台</h1><p>规划研究任务，追踪证据与执行进度。</p></div><span class="header-tag">SUPERVISOR / WORKERS</span></header>
  <div class="workspace"><section class="primary">
    <div class="card"><div class="card-head"><div><h2>会话连接</h2><p>使用访问令牌加载或创建研究会话。</p></div></div><div class="access-fields"><label>访问令牌<input v-model="token" type="password" placeholder="JWT Bearer token" autocomplete="off"></label><label>会话 ID<input v-model="session"></label><button class="secondary" :disabled="busy || !token || !session" @click="act('load')">加载会话</button></div></div>
    <div class="card"><div class="card-head"><div><h2>论文语料库</h2><p>上传 PDF 自动提取与分块，或检索并导入 arXiv 论文。</p></div><button class="secondary" :disabled="corpusBusy || !token" @click="corpusAction('load')">查看已导入</button></div>
      <div class="corpus-row"><input type="file" accept="application/pdf,.pdf" @change="choosePdf"><button :disabled="corpusBusy || !token || !pdf" @click="corpusAction('upload')">上传并建库</button></div>
      <div class="corpus-row"><input v-model="searchQuery" placeholder="搜索 arXiv 论文"><button class="secondary" :disabled="corpusBusy || !token || !searchQuery.trim()" @click="corpusAction('search')">在线检索</button></div>
      <p v-if="notice" class="corpus-notice">{{ notice }}</p>
      <div v-for="item in results" :key="item.id" class="corpus-item"><div><strong>{{ item.title }}</strong><small>{{ item.authors.join('、') }} · {{ item.id }}</small><p>{{ item.abstract }}</p></div><button class="secondary" :disabled="corpusBusy" @click="corpusAction('import', item.id)">导入</button></div>
      <details v-if="papers.length"><summary>已导入 {{ papers.length }} 篇论文</summary><div v-for="paper in papers" :key="paper.id" class="corpus-item"><div><strong>{{ paper.title }}</strong><small>{{ paper.source }}</small></div></div></details>
    </div>
    <div class="card"><div class="card-head"><div><h2>开始研究</h2><p>描述你的研究问题，先生成可审阅的任务计划。</p></div><select v-model="provider"><option value="openai">OpenAI</option><option value="deepseek">DeepSeek</option></select></div><textarea v-model="question" placeholder="例如：比较混合检索与重排序的作用，并整理证据来源。"></textarea><div class="prompt-footer"><span>计划生成后需要确认才会执行</span><button :disabled="busy || !token || !question.trim()" @click="act('plan')">{{ busy ? '处理中…' : '生成计划 →' }}</button></div></div>
    <div v-if="error" class="error">{{ error }}</div>
    <div v-if="workflow" class="card"><div class="card-head"><div><div class="eyebrow">WORKFLOW</div><h2>任务计划</h2><p>{{ workflow.question }}</p></div><span class="status">{{ labels[workflow.status] || workflow.status }}</span></div><div v-for="task in workflow.tasks" :key="task.id" class="task"><div class="task-indicator" :class="task.status"></div><div class="task-body"><div class="task-top"><strong>{{ names[task.worker] || task.worker }}</strong><span>{{ labels[task.status] || task.status }}</span></div><p>{{ task.goal }}</p><small v-if="task.depends_on.length">依赖 {{ task.depends_on.join('、') }}</small><details v-if="task.result"><summary>查看结果</summary><pre>{{ task.result }}</pre></details></div></div><details v-if="workflow.trace?.length"><summary>查看执行追踪</summary><pre>{{ JSON.stringify(workflow.trace, null, 2) }}</pre></details><div v-if="workflow.error" class="error">{{ workflow.error }}</div><div v-if="workflow.status !== 'completed' && workflow.status !== 'cancelled'" class="actions"><button class="secondary" :disabled="busy" @click="act('cancel')">取消计划</button><button :disabled="busy" @click="act('resume')">{{ busy ? '执行中…' : '确认并执行 →' }}</button></div><div v-if="workflow.result" class="result"><div class="eyebrow">RESEARCH OUTPUT</div><h3>研究结果</h3><pre>{{ workflow.result }}</pre></div></div>
  </section><aside class="right"><div class="card"><div class="eyebrow">AGENT TEAM</div><h2>研究团队</h2><p class="muted">每个角色负责不同的研究环节。</p><div v-for="(name, key) in names" :key="key" class="worker"><span class="worker-icon">✳</span><div><strong>{{ name }}</strong><small>{{ key }}</small></div></div></div><div class="note"><strong>会话与追踪</strong><p>执行计划和阶段结果保存在当前会话中。加载相同会话可继续查看。</p></div></aside></div></main></div>
</template>
