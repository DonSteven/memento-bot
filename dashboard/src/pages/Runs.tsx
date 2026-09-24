import { useEffect, useRef, useState } from 'react'
import { RefreshCw } from 'lucide-react'
import { Link, useParams } from 'react-router-dom'

import { EmptyState, ErrorState, LoadingState } from '../components/States'
import { useLiveUpdates } from '../components/LiveUpdates'
import { StatusBadge } from '../components/StatusBadge'
import { TraceTree } from '../components/TraceTree'
import { Button } from '../components/ui/button'
import { dashboardApi, ApiError } from '../lib/api'
import { formatDuration, formatTime, shortId } from '../lib/format'
import type { Run, RunDetail } from '../lib/types'

export function Runs() {
  const { versions } = useLiveUpdates()
  const { runId } = useParams<{ runId: string }>()
  const [items, setItems] = useState<Run[]>([])
  const [cursor, setCursor] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [loadingMore, setLoadingMore] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [reload, setReload] = useState(0)
  const moreController = useRef<AbortController | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    moreController.current?.abort()
    setLoadingMore(false)
    setLoading(true)
    setError(null)
    dashboardApi.runs(controller.signal).then(result => {
      if (controller.signal.aborted) return
      setItems(result.items)
      setCursor(result.next_cursor)
    }).catch((cause: unknown) => {
      if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : 'Unknown error')
    }).finally(() => {
      if (!controller.signal.aborted) setLoading(false)
    })
    return () => { controller.abort(); moreController.current?.abort() }
  }, [reload, versions.runs])

  async function loadMore() {
    if (!cursor || loadingMore) return
    const controller = new AbortController()
    moreController.current = controller
    setLoadingMore(true)
    setError(null)
    try {
      const page = await dashboardApi.runs(controller.signal, cursor)
      if (controller.signal.aborted) return
      setItems(previous => {
        const seen = new Set(previous.map(run => run.run_id))
        return [...previous, ...page.items.filter(run => !seen.has(run.run_id))]
      })
      setCursor(page.next_cursor)
    } catch (cause) {
      if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : 'Unknown error')
    } finally {
      if (!controller.signal.aborted) setLoadingMore(false)
    }
  }

  return <>
    <div className="page-heading">
      <div><span className="eyebrow">02 / EXECUTION HISTORY</span><h1>Runs</h1>
        <p className="page-subtitle">Inspect each Agent turn, model step, and tools batch.</p></div>
      <div className="heading-actions"><Button variant="outline" onClick={() => setReload(value => value + 1)} disabled={loading}>
        <RefreshCw size={14} aria-hidden="true" /> Refresh</Button></div>
    </div>
    <div className="runs-grid">
      <section className="panel run-list" aria-label="Run list">
        <div className="run-list-head"><strong>Recent runs</strong><small>{items.length} LOADED</small></div>
        {loading && items.length === 0 ? <LoadingState label="Loading runs…" /> :
          error && items.length === 0 ? <ErrorState message={error} onRetry={() => setReload(value => value + 1)} /> :
          items.length === 0 ? <EmptyState title="No runs yet" description="Start an Agent turn, then refresh." /> :
          <div className="run-list-items">{items.map(run =>
            <Link to={`/runs/${run.run_id}`} className={`run-item${run.run_id === runId ? ' active' : ''}`}
              key={run.run_id} aria-current={run.run_id === runId ? 'page' : undefined}>
              <span className="run-item-top"><span className="run-item-id">{shortId(run.run_id)}</span>
                <StatusBadge status={run.status} /></span>
              <span className="run-item-meta"><span>{formatTime(run.started_at)}</span>
                <span>{formatDuration(run.duration_ms)}</span></span>
            </Link>)}</div>}
        {error && items.length > 0 && <div className="detail-error" role="alert">{error}</div>}
        {cursor && <div className="run-list-foot"><Button variant="outline" size="sm" onClick={loadMore} disabled={loading || loadingMore}>
          {loadingMore ? 'Loading…' : 'Load more'}</Button></div>}
      </section>
      <section className="panel run-detail" aria-label="Run details">
        {runId ? <RunDetailPanel key={runId} runId={runId} reload={reload + versions.runs} /> :
          <div className="detail-placeholder">Select a run to inspect its trace.</div>}
      </section>
    </div>
  </>
}

function RunDetailPanel({ runId, reload }: { runId: string; reload: number }) {
  const [detail, setDetail] = useState<RunDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<ApiError | null>(null)
  const [retry, setRetry] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    setDetail(null)
    setLoading(true)
    setError(null)
    dashboardApi.run(runId, controller.signal).then(result => {
      if (!controller.signal.aborted) setDetail(result)
    }).catch((cause: unknown) => {
      if (!controller.signal.aborted) setError(cause instanceof ApiError ? cause :
        new ApiError(cause instanceof Error ? cause.message : 'Unknown error', 0))
    }).finally(() => {
      if (!controller.signal.aborted) setLoading(false)
    })
    return () => controller.abort()
  }, [runId, reload, retry])

  if (loading) return <LoadingState label="Loading trace…" />
  if (error) return <ErrorState message={error.message} notFound={error.status === 404}
    onRetry={error.status === 404 ? undefined : () => setRetry(value => value + 1)} />
  if (!detail) return null

  const { run } = detail
  return <>
    <div className="detail-head">
      <div className="detail-head-main"><h2>{run.run_id}</h2><StatusBadge status={run.status} /></div>
      <div className="detail-meta">
        <span>SESSION<strong>{run.session_key}</strong></span>
        <span>CHANNEL<strong>{run.channel}</strong></span>
        <span>MODEL<strong>{run.model}</strong></span>
        <span>STARTED<strong>{formatTime(run.started_at)}</strong></span>
        <span>DURATION<strong>{formatDuration(run.duration_ms)}</strong></span>
        <span>STOP REASON<strong>{run.stop_reason || '—'}</strong></span>
        <span>TOKENS<strong>{run.usage_reported ? `${run.prompt_tokens} in / ${run.completion_tokens} out` : 'Not reported'}</strong></span>
      </div>
      {run.error && <div className="detail-error" role="alert">{run.error}</div>}
    </div>
    <TraceTree detail={detail} />
  </>
}
