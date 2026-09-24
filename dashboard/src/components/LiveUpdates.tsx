import { createContext, useContext, useEffect, useState } from 'react'
import type { ReactNode } from 'react'

type View = 'overview' | 'runs' | 'memory' | 'tasks'
type Versions = Record<View, number>
type LiveState = { versions: Versions; status: 'connecting' | 'connected' | 'disconnected' }

const LiveContext = createContext<LiveState>({
  versions: { overview: 0, runs: 0, memory: 0, tasks: 0 }, status: 'connecting',
})

export function useLiveUpdates(): LiveState {
  return useContext(LiveContext)
}

export function LiveUpdatesProvider({ children }: { children: ReactNode }) {
  const [versions, setVersions] = useState<Versions>({ overview: 0, runs: 0, memory: 0, tasks: 0 })
  const [status, setStatus] = useState<LiveState['status']>('connecting')

  useEffect(() => {
    let disposed = false
    let socket: WebSocket | null = null
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null
    let flushTimer: ReturnType<typeof setTimeout> | null = null
    const pending = new Set<View>()

    function invalidate(...views: View[]) {
      views.forEach(view => pending.add(view))
      if (flushTimer) return
      flushTimer = setTimeout(() => {
        flushTimer = null
        const changed = new Set(pending)
        pending.clear()
        if (!disposed) setVersions(previous => ({
          overview: previous.overview + Number(changed.has('overview')),
          runs: previous.runs + Number(changed.has('runs')),
          memory: previous.memory + Number(changed.has('memory')),
          tasks: previous.tasks + Number(changed.has('tasks')),
        }))
      }, 100)
    }

    function connect() {
      if (disposed) return
      const scheme = location.protocol === 'https:' ? 'wss:' : 'ws:'
      socket = new WebSocket(`${scheme}//${location.host}/api/dashboard/events`)
      socket.onopen = () => {
        if (disposed) return
        setStatus('connected')
        invalidate('overview', 'runs', 'memory', 'tasks')
      }
      socket.onmessage = event => {
        try {
          const message = JSON.parse(event.data) as { type?: string }
          if (message.type?.startsWith('run.')) invalidate('overview', 'runs')
          else if (message.type === 'task.changed') invalidate('overview', 'tasks')
        } catch { /* Ignore malformed invalidations. */ }
      }
      socket.onerror = () => socket?.close()
      socket.onclose = () => {
        if (disposed) return
        setStatus('disconnected')
        reconnectTimer = setTimeout(connect, 2000)
      }
    }

    connect()
    return () => {
      disposed = true
      if (reconnectTimer) clearTimeout(reconnectTimer)
      if (flushTimer) clearTimeout(flushTimer)
      socket?.close()
    }
  }, [])

  return <LiveContext.Provider value={{ versions, status }}>{children}</LiveContext.Provider>
}
