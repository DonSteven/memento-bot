import { useEffect, useMemo, useState } from 'react'
import { ArrowRight, RefreshCw } from 'lucide-react'
import { Link } from 'react-router-dom'
import {
  Bar, BarChart, CartesianGrid, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from 'recharts'

import { EmptyState, ErrorState, LoadingState } from '../components/States'
import { useLiveUpdates } from '../components/LiveUpdates'
import { StatusBadge } from '../components/StatusBadge'
import { Button } from '../components/ui/button'
import { dashboardApi } from '../lib/api'
import { formatDuration, formatHour, formatNumber, formatTime, shortId } from '../lib/format'
import type { OverviewData } from '../lib/types'

type Range = 6 | 12 | 24

export function Overview() {
  const { versions } = useLiveUpdates()
  const [data, setData] = useState<OverviewData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [reload, setReload] = useState(0)
  const [range, setRange] = useState<Range>(24)

  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    setError(null)
    dashboardApi.overview(controller.signal).then(result => {
      if (!controller.signal.aborted) setData(result)
    }).catch((cause: unknown) => {
      if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : 'Unknown error')
    }).finally(() => {
      if (!controller.signal.aborted) setLoading(false)
    })
    return () => controller.abort()
  }, [reload, versions.overview])

  const hours = useMemo(() => data?.hours.slice(-range) ?? [], [data, range])
  const hasRuns = hours.some(hour => hour.runs > 0)
  const hasLatency = hours.some(hour => hour.avg_latency_ms !== null)
  const unavailable = error && data === null

  return <>
    <div className="page-heading">
      <div>
        <span className="eyebrow">01 / WORKSPACE HEALTH</span>
        <h1>Overview</h1>
        <p className="page-subtitle">A quick view of Agent runs started in the last 24 hours.</p>
      </div>
      <div className="heading-actions">
        <Button variant="outline" onClick={() => setReload(value => value + 1)} disabled={loading}>
          <RefreshCw size={14} aria-hidden="true" /> Refresh
        </Button>
      </div>
    </div>

    {unavailable ? <ErrorState message={error} onRetry={() => setReload(value => value + 1)} /> :
      loading && !data ? <LoadingState label="Loading overview…" /> : data && <>
        {error && <div className="detail-error" role="alert">{error} — showing the last loaded data.</div>}
        <section className="metrics" aria-label="Last 24 hour metrics">
          <Metric label="Runs" value={formatNumber(data.runs)} accent />
          <Metric label="Success rate" value={data.success_rate === null ? '—' : `${Math.round(data.success_rate * 100)}%`} />
          <Metric label="Average latency" value={formatDuration(data.avg_latency_ms)} />
          <Metric label="P95 latency" value={formatDuration(data.p95_latency_ms)} />
          <Metric label="Tool errors" value={formatNumber(data.tool_errors)} />
          <Metric label="Enabled tasks" value={formatNumber(data.enabled_tasks)} />
        </section>

        <section className="charts" aria-label="Hourly trends">
          <div className="panel chart-panel">
            <div className="panel-header">
              <div><h2 className="panel-title">Runs per hour</h2><span className="panel-note">UTC buckets · local labels</span></div>
              <RangeSelect value={range} onChange={setRange} />
            </div>
            {hasRuns ? <div className="chart-wrap">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={hours} margin={{ top: 9, right: 8, bottom: 0, left: 7 }}>
                  <CartesianGrid vertical={false} stroke="#edf1ef" />
                  <XAxis dataKey="hour" tickFormatter={formatHour} tickLine={false} axisLine={false} fontSize={10} tickMargin={10} minTickGap={18} />
                  <YAxis width={42} allowDecimals={false} tickLine={false} axisLine={false} fontSize={10} />
                  <Tooltip labelFormatter={value => formatTime(String(value))} formatter={value => [value, 'Runs']} contentStyle={{ borderRadius: 7, borderColor: '#dce5e3', fontSize: 11 }} />
                  <Bar dataKey="runs" fill="#168d80" radius={[3, 3, 0, 0]} maxBarSize={23} />
                </BarChart>
              </ResponsiveContainer>
            </div> : <EmptyState title="No runs in this range" description="New runs will appear after an Agent turn." />}
          </div>
          <div className="panel chart-panel">
            <div className="panel-header">
              <div><h2 className="panel-title">Latency per hour</h2><span className="panel-note">Average of finished runs · ms</span></div>
              <RangeSelect value={range} onChange={setRange} />
            </div>
            {hasLatency ? <div className="chart-wrap">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={hours} margin={{ top: 9, right: 8, bottom: 0, left: 7 }}>
                  <CartesianGrid vertical={false} stroke="#edf1ef" />
                  <XAxis dataKey="hour" tickFormatter={formatHour} tickLine={false} axisLine={false} fontSize={10} tickMargin={10} minTickGap={18} />
                  <YAxis width={52} tickLine={false} axisLine={false} fontSize={10} tickFormatter={value => `${Math.round(value)}ms`} />
                  <Tooltip labelFormatter={value => formatTime(String(value))} formatter={value => [formatDuration(Number(value)), 'Average latency']} contentStyle={{ borderRadius: 7, borderColor: '#dce5e3', fontSize: 11 }} />
                  <Bar dataKey="avg_latency_ms" fill="#62aea3" radius={[3, 3, 0, 0]} maxBarSize={23} />
                </BarChart>
              </ResponsiveContainer>
            </div> : <EmptyState title="No latency samples" description="Latency appears when a run finishes." />}
          </div>
        </section>

        <section className="panel" aria-labelledby="recent-heading">
          <div className="panel-header">
            <h2 id="recent-heading" className="panel-title">Recent runs</h2>
            <Link to="/runs" className="button button-ghost button-small">View all <ArrowRight size={13} /></Link>
          </div>
          {data.recent_runs.length === 0 ? <EmptyState title="No runs yet" description="Agent activity will appear here." /> :
            <div className="recent-table-wrap"><table className="recent-table">
              <thead><tr><th>Run</th><th>Status</th><th>Started</th><th>Duration</th><th>Model</th></tr></thead>
              <tbody>{data.recent_runs.map(run => <tr key={run.run_id}>
                <td><Link to={`/runs/${run.run_id}`} aria-label={`Open run ${run.run_id}`}>{shortId(run.run_id)}</Link></td>
                <td><StatusBadge status={run.status} /></td>
                <td>{formatTime(run.started_at)}</td>
                <td className="mono">{formatDuration(run.duration_ms)}</td>
                <td className="mono">{run.model}</td>
              </tr>)}</tbody>
            </table></div>}
        </section>
      </>}
  </>
}

function Metric({ label, value, accent = false }: { label: string; value: string; accent?: boolean }) {
  return <div className="panel metric"><span className="metric-label">{label}</span>
    <strong className={`metric-value${accent ? ' accent' : ''}`}>{value}</strong></div>
}

function RangeSelect({ value, onChange }: { value: Range; onChange: (value: Range) => void }) {
  return <select className="time-range" aria-label="Chart time range" value={value}
    onChange={event => onChange(Number(event.target.value) as Range)}>
    <option value={6}>Last 6h</option><option value={12}>Last 12h</option><option value={24}>Last 24h</option>
  </select>
}
