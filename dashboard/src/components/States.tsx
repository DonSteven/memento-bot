import { AlertCircle, Inbox } from 'lucide-react'

import { Button } from './ui/button'

export function LoadingState({ label = 'Loading…' }: { label?: string }) {
  return <div className="state-panel" role="status"><span className="loading-dot" />{label}</div>
}

export function EmptyState({ title, description }: { title: string; description: string }) {
  return <div className="state-panel state-stack">
    <Inbox aria-hidden="true" size={24} />
    <strong>{title}</strong><span>{description}</span>
  </div>
}

export function ErrorState({ message, onRetry, notFound = false }:
  { message: string; onRetry?: () => void; notFound?: boolean }) {
  return <div className="state-panel state-stack" role="alert">
    <AlertCircle aria-hidden="true" size={24} />
    <strong>{notFound ? 'Run not found' : 'Unable to load data'}</strong>
    <span>{message}</span>
    {onRetry && <Button variant="outline" size="sm" onClick={onRetry}>Try again</Button>}
  </div>
}
