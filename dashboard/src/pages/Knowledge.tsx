import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { Search } from 'lucide-react'

import { Button } from '../components/ui/button'
import { LoadingState } from '../components/States'
import { dashboardApi, ApiError } from '../lib/api'
import type { ChildEvidence, KnowledgeResult, ParentEvidence } from '../lib/types'

const statusLabels: Record<KnowledgeResult['status'], string> = {
  sufficient: 'Sufficient', insufficient: 'Insufficient',
  retrieval_error: 'Retrieval error', assessment_error: 'Assessment error',
  online_error: 'Online error',
}

function EvidenceMeta({ evidence }: { evidence: ParentEvidence | ChildEvidence }) {
  return <div className="knowledge-evidence-meta">
    <span>Rerank score <strong className="mono">{evidence.rerank_score}</strong></span>
    <span>Rank <strong className="mono">{evidence.rerank_rank}</strong></span>
    {'child_id' in evidence && <>
      <span>FTS rank <strong className="mono">{evidence.fts_rank ?? '—'}</strong></span>
      <span>Vector rank <strong className="mono">{evidence.vector_rank ?? '—'}</strong></span>
      <span>Similarity <strong className="mono">{evidence.vector_similarity ?? '—'}</strong></span>
    </>}
    {evidence.partial && <span className="knowledge-partial">Partial</span>}
    {evidence.sources.map(source => <span className="memory-source" key={source}>{source}</span>)}
  </div>
}

function EvidenceCard({ parent, children }: { parent: ParentEvidence; children: ChildEvidence[] }) {
  return <article className="knowledge-evidence">
    <div className="knowledge-evidence-head"><strong>{parent.title || 'Untitled source'}</strong>
      <span className="mono">Parent {parent.parent_id} · Page {parent.page_id}</span></div>
    <div className="knowledge-url">{parent.url}</div>
    <EvidenceMeta evidence={parent} />
    <p className="knowledge-text">{parent.text}</p>
    <div className="knowledge-children"><h4>Selected passages · {children.length}</h4>
      {children.map(child => <div className="knowledge-child" key={child.child_id}>
        <div className="knowledge-evidence-head"><strong>Passage {child.child_id}</strong>
          <span className="mono">Parent {child.parent_id}</span></div>
        <EvidenceMeta evidence={child} />
        <p className="knowledge-text">{child.text}</p>
      </div>)}</div>
  </article>
}

function UrlList({ label, urls, empty }: { label: string; urls: string[]; empty: string }) {
  return <div className="knowledge-online-list"><h4>{label} <span>{urls.length}</span></h4>
    {urls.length ? <ul>{urls.map((url, index) => <li key={`${url}-${index}`} className="mono">{url}</li>)}</ul> :
      <p className="muted">{empty}</p>}</div>
}

export function Knowledge() {
  const [query, setQuery] = useState('')
  const [docLimit, setDocLimit] = useState(10)
  const [evidenceLimit, setEvidenceLimit] = useState(5)
  const [result, setResult] = useState<KnowledgeResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [running, setRunning] = useState(false)
  const inFlight = useRef<AbortController | null>(null)

  useEffect(() => () => inFlight.current?.abort(), [])

  async function runRetrieval(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!query.trim() || inFlight.current) return
    const controller = new AbortController()
    inFlight.current = controller
    setRunning(true)
    setResult(null)
    setError(null)
    try {
      const data = await dashboardApi.searchKnowledge(query, docLimit, evidenceLimit, controller.signal)
      if (!controller.signal.aborted) setResult(data)
    } catch (cause) {
      if (!controller.signal.aborted) {
        const prefix = cause instanceof ApiError && cause.status === 503 ? 'Knowledge unavailable: '
          : cause instanceof ApiError && cause.status === 502 ? 'Upstream failure: ' : ''
        setError(prefix + (cause instanceof Error ? cause.message : 'Unknown error'))
      }
    } finally {
      if (inFlight.current === controller) inFlight.current = null
      if (!controller.signal.aborted) setRunning(false)
    }
  }

  return <>
    <div className="page-heading"><div><span className="eyebrow">04 / KNOWLEDGE RETRIEVAL</span>
      <h1>Knowledge</h1><p className="page-subtitle">Run the current retrieval workflow and inspect selected evidence.</p></div></div>
    <section className="panel knowledge-panel" aria-label="Knowledge query">
      <div className="panel-header"><h2 className="panel-title">Retrieval query</h2><span className="panel-note">ON DEMAND</span></div>
      <form className="knowledge-form" onSubmit={runRetrieval}>
        <label htmlFor="knowledge-query">Question or query</label>
        <input id="knowledge-query" value={query} onChange={event => setQuery(event.target.value)}
          maxLength={512} required placeholder="What should the knowledge base answer?" />
        <div className="knowledge-form-actions">
          <label htmlFor="knowledge-doc-limit">Document limit</label>
          <input id="knowledge-doc-limit" type="number" min={1} max={100} required value={docLimit}
            onChange={event => setDocLimit(Number(event.target.value))} />
          <label htmlFor="knowledge-evidence-limit">Evidence limit</label>
          <input id="knowledge-evidence-limit" type="number" min={1} max={100} required value={evidenceLimit}
            onChange={event => setEvidenceLimit(Number(event.target.value))} />
          <Button type="submit" disabled={running || !query.trim()}><Search size={14} aria-hidden="true" />
            {running ? 'Running…' : 'Run retrieval'}</Button>
        </div>
        <p className="memory-help">This may call model, embedding, rerank, and online services, and may write fetched content to the knowledge base. Results show selected evidence and final assessment; fetched URLs record attempts, not successful fetches.</p>
      </form>
    </section>
    {error && <div className="panel knowledge-error" role="alert">{error}</div>}
    {running && <div className="panel"><LoadingState label="Retrieval in progress…" /></div>}
    {result && <>
      <section className="panel knowledge-panel" aria-label="Retrieval conclusion">
        <div className="panel-header"><h2 className="panel-title">Conclusion</h2>
          <span className={`knowledge-status knowledge-status-${result.status}`}>{statusLabels[result.status]}</span></div>
        <div className="knowledge-conclusion">
          <div className="knowledge-facts"><span>Coverage <strong>{result.sufficient === null ? 'Not assessed' : result.sufficient ? 'Sufficient' : 'Insufficient'}</strong></span>
            <span>Query <strong>{result.query}</strong></span></div>
          <p>{result.reason}</p>
          <h3>Missing points</h3>
          {result.missing_points.length ? <ul>{result.missing_points.map((point, index) =>
            <li key={index}>{point}</li>)}</ul> : <p className="muted">None reported.</p>}
        </div>
      </section>
      <section className="panel knowledge-panel" aria-label="Selected evidence">
        <div className="panel-header"><h2 className="panel-title">Selected evidence</h2>
          <span className="panel-note">{result.parents.length} PARENTS · {result.children.length} PASSAGES</span></div>
        {result.parents.length ? <div className="knowledge-evidence-list">{result.parents.map(parent =>
          <EvidenceCard key={parent.parent_id} parent={parent}
            children={result.children.filter(child => child.parent_id === parent.parent_id)} />)}</div> :
          <p className="knowledge-empty">No selected evidence.</p>}
      </section>
      <section className="panel knowledge-panel" aria-label="Online supplementation">
        <div className="panel-header"><h2 className="panel-title">Online supplementation</h2>
          <span className="panel-note">{result.online_attempted ? 'ATTEMPTED' : 'NOT ATTEMPTED'}</span></div>
        <div className="knowledge-online">
          <UrlList label="Fetch attempts" urls={result.fetched_urls} empty="No URL fetch attempts recorded." />
          <UrlList label="Ingested URLs" urls={result.ingested_urls} empty="No URLs ingested." />
          <div className="knowledge-online-list"><h4>Online errors <span>{result.online_errors.length}</span></h4>
            {result.online_errors.length ? <ul>{result.online_errors.map((message, index) =>
              <li key={index}>{message}</li>)}</ul> : <p className="muted">None reported.</p>}</div>
        </div>
      </section>
    </>}
  </>
}
