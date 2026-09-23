export function formatTime(value: string | null): string {
  if (!value) return '—'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? '—' : date.toLocaleString()
}

export function formatHour(value: string): string {
  const date = new Date(value)
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

export function formatDuration(value: number | null): string {
  if (value === null) return '—'
  if (value < 1000) return `${Math.round(value)} ms`
  return `${(value / 1000).toFixed(2)} s`
}

export function formatNumber(value: number | null): string {
  return value === null ? '—' : Intl.NumberFormat().format(value)
}

export function shortId(value: string): string {
  return value.length > 13 ? `${value.slice(0, 8)}…` : value
}
