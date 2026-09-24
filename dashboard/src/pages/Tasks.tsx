import { useEffect, useRef, useState } from 'react'
import { RefreshCw, Trash2 } from 'lucide-react'

import { EmptyState, ErrorState, LoadingState } from '../components/States'
import { Button } from '../components/ui/button'
import { dashboardApi } from '../lib/api'
import { formatDuration, formatTime } from '../lib/format'
import type { Task } from '../lib/types'

function localTime(ms: number | null): string {
  return ms === null ? '—' : formatTime(new Date(ms).toISOString())
}

function scheduleLabel(task: Task): string {
  const { schedule } = task
  if (schedule.kind === 'at') return `Once · ${localTime(schedule.at_ms)}`
  if (schedule.kind === 'every') {
    const interval = schedule.every_ms
    if (interval === null) return 'Interval · —'
    if (interval % 86400000 === 0) return `Every ${interval / 86400000} d`
    if (interval % 3600000 === 0) return `Every ${interval / 3600000} h`
    if (interval % 60000 === 0) return `Every ${interval / 60000} min`
    return `Every ${interval / 1000} s`
  }
  return `Cron · ${schedule.expr || '—'} · ${schedule.tz || 'system timezone'}`
}

export function Tasks() {
  const [items, setItems] = useState<Task[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [busyId, setBusyId] = useState<string | null>(null)
  const [reload, setReload] = useState(0)
  const mutation = useRef<AbortController | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    setLoading(true)
    setError(null)
    dashboardApi.tasks(controller.signal).then(data => {
      if (!controller.signal.aborted) setItems(data.items)
    }).catch((cause: unknown) => {
      if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : 'Unknown error')
    }).finally(() => {
      if (!controller.signal.aborted) setLoading(false)
    })
    return () => controller.abort()
  }, [reload])

  useEffect(() => () => mutation.current?.abort(), [])

  async function change(task: Task, action: 'toggle' | 'delete') {
    if (mutation.current) return
    if (action === 'delete' && !window.confirm(`Delete task "${task.name}"? This removes its schedule and history.`)) return
    const controller = new AbortController()
    mutation.current = controller
    setBusyId(task.id)
    setError(null)
    try {
      if (action === 'toggle') await dashboardApi.setTaskEnabled(task.id, !task.enabled, controller.signal)
      else await dashboardApi.deleteTask(task.id, controller.signal)
      const data = await dashboardApi.tasks(controller.signal)
      if (!controller.signal.aborted) setItems(data.items)
    } catch (cause) {
      if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : 'Unknown error')
    } finally {
      if (mutation.current === controller) mutation.current = null
      if (!controller.signal.aborted) setBusyId(null)
    }
  }

  return <>
    <div className="page-heading"><div><span className="eyebrow">05 / SCHEDULED WORK</span>
      <h1>Tasks</h1><p className="page-subtitle">View and control tasks in this gateway's scheduler.</p></div>
      <div className="heading-actions"><Button variant="outline" onClick={() => setReload(value => value + 1)} disabled={loading || !!busyId}>
        <RefreshCw size={14} aria-hidden="true" /> Refresh</Button></div></div>
    <p className="tasks-note">Disabling a task prevents future scheduling; it does not cancel a run already in progress.</p>
    {error && items && <div className="detail-error tasks-error" role="alert">{error}</div>}
    {error && !items ? <ErrorState message={error} onRetry={() => setReload(value => value + 1)} /> :
      loading && !items ? <LoadingState label="Loading tasks…" /> : items &&
      <section className="panel" aria-label="Scheduled tasks">
        <div className="panel-header"><h2 className="panel-title">Scheduled tasks</h2><span className="panel-note">{items.length} TOTAL</span></div>
        {items.length === 0 ? <EmptyState title="No tasks" description="Tasks created by the Agent will appear here." /> :
          <div className="tasks-table-wrap"><table className="tasks-table">
            <thead><tr><th>Task</th><th>Status</th><th>Schedule</th><th>Next run</th><th>Last run</th><th>Actions</th></tr></thead>
            <tbody>{items.map(task => <tr key={task.id}>
              <td><strong>{task.name}</strong><small className="mono">{task.id}</small></td>
              <td><span className={`task-status ${task.enabled ? 'task-enabled' : 'task-disabled'}`}>{task.enabled ? 'Enabled' : 'Disabled'}</span></td>
              <td>{scheduleLabel(task)}</td>
              <td className="mono">{localTime(task.state.next_run_at_ms)}</td>
              <td><span className="mono">{localTime(task.state.last_run_at_ms)}</span>
                {task.state.last_status && <small className={`task-last task-last-${task.state.last_status}`}>{task.state.last_status}</small>}</td>
              <td><div className="task-actions">
                <Button size="sm" variant="outline" disabled={!!busyId} onClick={() => change(task, 'toggle')}>
                  {busyId === task.id ? 'Saving…' : task.enabled ? 'Disable' : 'Enable'}</Button>
                <Button size="sm" variant="ghost" disabled={!!busyId} onClick={() => change(task, 'delete')}
                  aria-label={`Delete task ${task.name}`}><Trash2 size={13} aria-hidden="true" /> Delete</Button>
              </div></td>
            </tr>)}</tbody>
          </table>
          <div className="tasks-history">{items.map(task => <details key={task.id}>
            <summary>{task.name} · Run history ({task.state.run_history.length})</summary>
            {task.state.run_history.length ? <div className="tasks-history-wrap"><table>
              <thead><tr><th>Started</th><th>Status</th><th>Duration</th><th>Error</th></tr></thead>
              <tbody>{[...task.state.run_history].reverse().map((record, index) => <tr key={`${record.run_at_ms}-${index}`}>
                <td className="mono">{localTime(record.run_at_ms)}</td><td>{record.status}</td>
                <td className="mono">{formatDuration(record.duration_ms)}</td><td>{record.error || '—'}</td>
              </tr>)}</tbody>
            </table></div> : <p className="muted">No recorded runs.</p>}
          </details>)}</div></div>}
      </section>}
  </>
}
