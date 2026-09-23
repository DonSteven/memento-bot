export type RunStatus = 'running' | 'completed' | 'error' | 'stopped' | 'cancelled'

export interface Run {
  run_id: string
  session_key: string
  channel: string
  model: string
  started_at: string
  finished_at: string | null
  duration_ms: number | null
  status: RunStatus
  stop_reason: string | null
  error: string | null
  prompt_tokens: number
  completion_tokens: number
  usage_reported: number
}

export interface Preview {
  text: string
  truncated: boolean
}

export interface MemoryData {
  core_count?: number
  retrieved_count?: number
  error?: Preview
}

export interface ModelData {
  usage: { prompt_tokens: number; completion_tokens: number }
  usage_reported: boolean
  stop_reason: string | null
  tool_call_count: number
  content_preview: Preview
}

export interface ToolCallData {
  call_id: string
  name: string
  arguments: Preview
  status: 'ok' | 'error'
  result_preview: Preview
}

export interface ToolsData {
  calls: ToolCallData[]
}

export interface ErrorData {
  error: Preview
}

export interface TraceEvent {
  event_id: number
  run_id: string
  kind: 'memory' | 'model' | 'tools' | 'error'
  iteration: number | null
  started_at: string
  duration_ms: number | null
  data: MemoryData | ModelData | ToolsData | ErrorData
}

export interface RunDetail {
  run: Run
  events: TraceEvent[]
}

export interface RunPage {
  items: Run[]
  next_cursor: string | null
}

export interface HourBucket {
  hour: string
  runs: number
  avg_latency_ms: number | null
}

export interface OverviewData {
  window_start: string
  window_end: string
  runs: number
  success_rate: number | null
  avg_latency_ms: number | null
  p95_latency_ms: number | null
  tool_errors: number
  enabled_tasks: number
  hours: HourBucket[]
  recent_runs: Run[]
}

export type MemoryClass = 'personal_profile' | 'preferences' | 'constraints' | 'projects' | 'daily_life' | 'plans_commitments'

export interface MemoryRecord {
  memory_id: string
  main_class: MemoryClass
  sub_class: string
  text: string
}

export interface MemorySnapshot {
  revision: number
  counts: Record<MemoryClass, number>
  records: MemoryRecord[]
}

export interface DynamicMemoryHit {
  record: MemoryRecord
  sources: string[]
  rrf_score: number
  fts_rank: number | null
  vector_rank: number | null
  vector_similarity: number | null
}

export interface MemorySearchResult {
  core: MemoryRecord[]
  dynamic: DynamicMemoryHit[]
}

export type KnowledgeStatus = 'sufficient' | 'insufficient' | 'retrieval_error' | 'assessment_error' | 'online_error'

export interface ParentEvidence {
  parent_id: number
  page_id: number
  title: string
  url: string
  text: string
  partial: boolean
  sources: string[]
  rerank_score: number
  rerank_rank: number
}

export interface ChildEvidence extends ParentEvidence {
  child_id: number
  fts_rank: number | null
  vector_rank: number | null
  vector_similarity: number | null
}

export interface KnowledgeResult {
  query: string
  status: KnowledgeStatus
  sufficient: boolean | null
  reason: string
  missing_points: string[]
  parents: ParentEvidence[]
  children: ChildEvidence[]
  online_attempted: boolean
  fetched_urls: string[]
  ingested_urls: string[]
  online_errors: string[]
}
