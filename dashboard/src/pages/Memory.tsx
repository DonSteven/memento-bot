import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { RefreshCw, Search } from 'lucide-react'

import { EmptyState, ErrorState, LoadingState } from '../components/States'
import { useLiveUpdates } from '../components/LiveUpdates'
import { Button } from '../components/ui/button'
import { dashboardApi, ApiError } from '../lib/api'
import type { DynamicMemoryHit, MemoryClass, MemoryRecord, MemorySearchResult, MemorySnapshot } from '../lib/types'

const classes: MemoryClass[] = [
  'personal_profile', 'preferences', 'constraints', 'projects', 'daily_life', 'plans_commitments',
]
const labels: Record<MemoryClass, string> = {
  personal_profile: 'Personal profile', preferences: 'Preferences', constraints: 'Constraints',
  projects: 'Projects', daily_life: 'Daily life', plans_commitments: 'Plans & commitments',
}

function RecordCard({ record }: { record: MemoryRecord }) {
  return <article className="memory-record">
    <div className="memory-record-meta"><strong>{labels[record.main_class]}</strong>
      <span>{record.sub_class}</span></div>
    <p>{record.text}</p>
    <small className="mono">{record.memory_id}</small>
  </article>
}

function HitCard({ hit }: { hit: DynamicMemoryHit }) {
  return <div className="memory-hit"><RecordCard record={hit.record} />
    <div className="memory-hit-meta">
      <span>RRF <strong className="mono">{hit.rrf_score}</strong></span>
      <span>FTS rank <strong className="mono">{hit.fts_rank ?? '—'}</strong></span>
      <span>Vector rank <strong className="mono">{hit.vector_rank ?? '—'}</strong></span>
      <span>Similarity <strong className="mono">{hit.vector_similarity ?? '—'}</strong></span>
      <span className="memory-sources">{hit.sources.map(source =>
        <span className="memory-source" key={source}>{source}</span>)}</span>
    </div>
  </div>
}

export function Memory() {
  const { versions } = useLiveUpdates()
  const [snapshot, setSnapshot] = useState<MemorySnapshot | null>(null)
  const [category, setCategory] = useState<MemoryClass | 'all'>('all')
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [reload, setReload] = useState(0)
  const [query, setQuery] = useState('')
  const [limit, setLimit] = useState(5)
  const [result, setResult] = useState<MemorySearchResult | null>(null)
  const [searching, setSearching] = useState(false)
  const [searchError, setSearchError] = useState<string | null>(null)
  const searchController = useRef<AbortController | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    setLoadError(null)
    dashboardApi.memory(controller.signal).then(data => {
      if (!controller.signal.aborted) setSnapshot(data)
    }).catch((cause: unknown) => {
      if (!controller.signal.aborted) setLoadError(cause instanceof Error ? cause.message : 'Unknown error')
    }).finally(() => {
      if (!controller.signal.aborted) setLoading(false)
    })
    return () => controller.abort()
  }, [reload, versions.memory])

  useEffect(() => () => searchController.current?.abort(), [])

  async function search(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!query.trim() || searching) return
    const controller = new AbortController()
    searchController.current = controller
    setSearching(true)
    setSearchError(null)
    setResult(null)
    try {
      const data = await dashboardApi.searchMemory(query, limit, controller.signal)
      if (!controller.signal.aborted) {
        setResult(data)
        setReload(value => value + 1)
      }
    } catch (cause) {
      if (!controller.signal.aborted) {
        const prefix = cause instanceof ApiError && cause.code === 'invalid_markdown' ? 'Invalid MEMORY.md: '
          : cause instanceof ApiError && cause.status === 409 ? 'Memory conflict: '
          : cause instanceof ApiError && cause.status === 502 ? 'Embedding failure: '
            : cause instanceof ApiError && cause.status === 503 ? 'Memory unavailable: ' : ''
        setSearchError(prefix + (cause instanceof Error ? cause.message : 'Unknown error'))
      }
    } finally {
      if (!controller.signal.aborted) setSearching(false)
    }
  }

  const records = snapshot?.records.filter(record => category === 'all' || record.main_class === category) ?? []
  return <>
    <div className="page-heading"><div><span className="eyebrow">03 / PERSISTENT MEMORY</span>
      <h1>Memory</h1><p className="page-subtitle">Committed memory and on-demand recall candidates.</p></div>
      <div className="heading-actions"><Button variant="outline" onClick={() => setReload(value => value + 1)} disabled={loading}>
        <RefreshCw size={14} aria-hidden="true" /> Refresh snapshot</Button></div></div>
    <section className="panel memory-panel" aria-label="Committed memory">
      <div className="panel-header"><h2 className="panel-title">Committed snapshot</h2>
        <span className="panel-note">REVISION {snapshot?.revision ?? '—'}</span></div>
      {loading && !snapshot ? <LoadingState label="Loading memory…" /> :
        loadError && !snapshot ? <ErrorState message={loadError} onRetry={() => setReload(value => value + 1)} /> : <>
          {loadError && <div className="detail-error" role="alert">{loadError}</div>}
          <div className="memory-categories" role="group" aria-label="Memory category">
            <button type="button" className={category === 'all' ? 'selected' : ''}
              aria-pressed={category === 'all'} onClick={() => setCategory('all')}>
              All <span>{snapshot?.records.length ?? 0}</span></button>
            {classes.map(mainClass => <button type="button" key={mainClass}
              className={category === mainClass ? 'selected' : ''}
              aria-pressed={category === mainClass} onClick={() => setCategory(mainClass)}>
              {labels[mainClass]} <span>{snapshot?.counts[mainClass] ?? 0}</span></button>)}
          </div>
          {records.length ? <div className="memory-records">{records.map(record =>
            <RecordCard record={record} key={record.memory_id} />)}</div> :
            <EmptyState title="No records" description="There are no committed records in this category." />}
        </>}
    </section>
    <section className="panel memory-panel" aria-label="Retrieval debug">
      <div className="panel-header"><h2 className="panel-title">Retrieval debug</h2><span className="panel-note">ON DEMAND</span></div>
      <form className="memory-search" onSubmit={search}>
        <label htmlFor="memory-query">Query</label>
        <div className="memory-search-controls"><input id="memory-query" value={query}
          onChange={event => setQuery(event.target.value)} maxLength={512} required
          placeholder="Search current memory…" />
          <label htmlFor="memory-limit">Limit</label><input id="memory-limit" type="number" min={1} max={100}
            value={limit} onChange={event => setLimit(Number(event.target.value))} required />
          <Button type="submit" disabled={searching || !query.trim()}><Search size={14} aria-hidden="true" />
            {searching ? 'Searching…' : 'Search'}</Button></div>
        <p className="memory-help">Search first syncs MEMORY.md with the database. It may write memory and call the embedding service. Dynamic results are recall candidates; later context budgets may trim them. The service’s dynamic top K may further cap the requested limit.</p>
      </form>
      {searchError && <div className="detail-error" role="alert">{searchError}</div>}
      {searching && <LoadingState label="Syncing and searching memory…" />}
      {result && <div className="memory-results">
        <div><h3>Core memory <span>{result.core.length}</span></h3>
          {result.core.length ? result.core.map(record => <RecordCard record={record} key={record.memory_id} />) :
            <p className="muted">No core memory.</p>}</div>
        <div><h3>Dynamic candidates <span>{result.dynamic.length}</span></h3>
          {result.dynamic.length ? result.dynamic.map(hit => <HitCard hit={hit} key={hit.record.memory_id} />) :
            <p className="muted">No dynamic candidates matched.</p>}</div>
      </div>}
    </section>
  </>
}
