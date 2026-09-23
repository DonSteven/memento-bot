import type { MemorySearchResult, MemorySnapshot, OverviewData, RunDetail, RunPage } from './types'

export class ApiError extends Error {
  constructor(message: string, public status: number, public code?: string) {
    super(message)
    this.name = 'ApiError'
  }
}

async function requestJson<T>(path: string, signal: AbortSignal, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`/api/dashboard${path}`, {
      ...init, signal, headers: { Accept: 'application/json', ...init?.headers },
    })
  } catch (error) {
    if (signal.aborted) throw error
    throw new ApiError('Cannot reach the Dashboard service. Check that the gateway is running.', 0)
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null) as { error?: { message?: string; code?: string } } | null
    throw new ApiError(body?.error?.message || `Request failed (${response.status})`,
      response.status, body?.error?.code)
  }
  return response.json() as Promise<T>
}

const getJson = <T,>(path: string, signal: AbortSignal) => requestJson<T>(path, signal)

export const dashboardApi = {
  overview: (signal: AbortSignal) => getJson<OverviewData>('/overview', signal),
  runs: (signal: AbortSignal, cursor?: string | null) => {
    const query = new URLSearchParams({ limit: '50' })
    if (cursor) query.set('cursor', cursor)
    return getJson<RunPage>(`/runs?${query}`, signal)
  },
  run: (runId: string, signal: AbortSignal) =>
    getJson<RunDetail>(`/runs/${encodeURIComponent(runId)}`, signal),
  memory: (signal: AbortSignal) => getJson<MemorySnapshot>('/memory', signal),
  searchMemory: (query: string, limit: number, signal: AbortSignal) =>
    requestJson<MemorySearchResult>('/memory/search', signal, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, limit }),
    }),
}
