import type { RunStatus } from '../lib/types'
import { Badge } from './ui/badge'

const labels: Record<RunStatus, string> = {
  running: 'Running', completed: 'Completed', error: 'Error',
  stopped: 'Stopped', cancelled: 'Cancelled',
}

export function StatusBadge({ status }: { status: RunStatus }) {
  return <Badge className={`status status-${status}`}><span className="status-dot" />{labels[status]}</Badge>
}
