import { AlertCircle, Brain, Braces, Hammer, MessageSquareText } from 'lucide-react'

import { formatDuration } from '../lib/format'
import type {
  ErrorData, MemoryData, ModelData, Preview, RunDetail, ToolCallData,
  ToolsData, TraceEvent,
} from '../lib/types'

export function TraceTree({ detail }: { detail: RunDetail }) {
  const memories = detail.events.filter(event => event.kind === 'memory')
  const iterations = new Map<number, TraceEvent[]>()
  const rootErrors: TraceEvent[] = []
  for (const event of detail.events) {
    if (event.kind === 'memory') continue
    if (event.iteration === null) rootErrors.push(event)
    else iterations.set(event.iteration, [...(iterations.get(event.iteration) ?? []), event])
  }

  return <section aria-labelledby="trace-heading">
    <div className="trace-header"><h3 id="trace-heading">Execution trace</h3>
      <span className="trace-caption">MODEL STEP AND TOOL BATCH TIMING</span></div>
    <div className="trace">
      {detail.events.length === 0 ? <p className="detail-placeholder">No events recorded for this run.</p> : <>
        {memories.map(event => <MemoryNode key={event.event_id} event={event} />)}
        {[...iterations.entries()].sort(([a], [b]) => a - b).map(([iteration, events]) =>
          <details className="trace-node" key={iteration} open>
            <summary><Braces size={14} aria-hidden="true" /><strong>Iteration {iteration + 1}</strong>
              <span className="node-right">{events.length} {events.length === 1 ? 'EVENT' : 'EVENTS'}</span></summary>
            <div className="trace-node-body">
              {events.map(event => <EventNode key={event.event_id} event={event} />)}
            </div>
          </details>)}
        {rootErrors.map(event => <ErrorNode key={event.event_id} event={event} />)}
      </>}
    </div>
  </section>
}

function EventNode({ event }: { event: TraceEvent }) {
  if (event.kind === 'model') return <ModelNode event={event} />
  if (event.kind === 'tools') return <ToolsNode event={event} />
  if (event.kind === 'error') return <ErrorNode event={event} />
  return null
}

function MemoryNode({ event }: { event: TraceEvent }) {
  const data = event.data as MemoryData
  return <details className="trace-node" open>
    <summary><Brain size={14} aria-hidden="true" /><strong>Memory</strong>
      <span className="node-right">{formatDuration(event.duration_ms)}</span></summary>
    <div className="trace-node-body">
      {data.error ? <PreviewBlock label="Memory error" preview={data.error} /> :
        <div className="trace-data"><span>Core records <strong>{data.core_count ?? '—'}</strong></span>
          <span>Retrieved records <strong>{data.retrieved_count ?? '—'}</strong></span></div>}
    </div>
  </details>
}

function ModelNode({ event }: { event: TraceEvent }) {
  const data = event.data as ModelData
  return <details className="trace-node" open>
    <summary><MessageSquareText size={14} aria-hidden="true" /><strong>Model step</strong>
      <span className="node-right">{formatDuration(event.duration_ms)}</span></summary>
    <div className="trace-node-body">
      <div className="trace-data">
        <span>Input tokens <strong>{data.usage_reported ? data.usage.prompt_tokens : '—'}</strong></span>
        <span>Output tokens <strong>{data.usage_reported ? data.usage.completion_tokens : '—'}</strong></span>
        <span>Tool calls <strong>{data.tool_call_count}</strong></span>
        <span>Stop reason <strong>{data.stop_reason || '—'}</strong></span>
      </div>
      <PreviewBlock label="Text preview" preview={data.content_preview} />
    </div>
  </details>
}

function ToolsNode({ event }: { event: TraceEvent }) {
  const data = event.data as ToolsData
  return <details className="trace-node" open>
    <summary><Hammer size={14} aria-hidden="true" /><strong>Tools batch</strong>
      <span className="node-right">{data.calls.length} {data.calls.length === 1 ? 'CALL' : 'CALLS'} · {formatDuration(event.duration_ms)}</span></summary>
    <div className="trace-node-body">
      {data.calls.map((call, index) => <ToolCallNode key={`${call.call_id}-${index}`} call={call} />)}
    </div>
  </details>
}

function ToolCallNode({ call }: { call: ToolCallData }) {
  return <details className="trace-node">
    <summary><strong>{call.name}</strong><span className="muted mono">{call.call_id}</span>
      <span className="node-right"><span className={`trace-call-status${call.status === 'error' ? ' error' : ''}`}>
        {call.status === 'error' ? 'Error' : 'OK'}</span></span></summary>
    <div className="trace-node-body">
      <PreviewBlock label="Arguments" preview={call.arguments} />
      <PreviewBlock label="Result preview" preview={call.result_preview} />
    </div>
  </details>
}

function ErrorNode({ event }: { event: TraceEvent }) {
  const data = event.data as ErrorData
  return <details className="trace-node" open>
    <summary><AlertCircle size={14} aria-hidden="true" /><strong>Error</strong></summary>
    <div className="trace-node-body"><PreviewBlock label="Error summary" preview={data.error} /></div>
  </details>
}

function PreviewBlock({ label, preview }: { label: string; preview?: Preview }) {
  if (!preview || !preview.text) return null
  return <div><span className="trace-label">{label}{preview.truncated &&
    <span className="truncated">TRUNCATED</span>}</span>
    <pre className="trace-pre">{preview.text}</pre></div>
}
