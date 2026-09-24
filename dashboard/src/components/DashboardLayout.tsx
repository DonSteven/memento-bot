import { Activity, ArrowUpRight, BookOpenText, Brain, CalendarClock, LayoutDashboard, ListTree } from 'lucide-react'
import { Link, NavLink, Outlet } from 'react-router-dom'
import { LiveUpdatesProvider, useLiveUpdates } from './LiveUpdates'

export function DashboardLayout() {
  return <LiveUpdatesProvider><DashboardFrame /></LiveUpdatesProvider>
}

function DashboardFrame() {
  const { status } = useLiveUpdates()
  return <div className="app-shell">
    <aside className="sidebar">
      <Link className="brand" to="/" aria-label="Memento Dashboard home">
        <span className="brand-mark"><Activity size={18} strokeWidth={2.5} /></span>
        <span><strong>MEMENTO</strong><small>RUN CONSOLE</small></span>
      </Link>
      <div className="sidebar-label">WORKSPACE</div>
      <nav className="nav" aria-label="Dashboard pages">
        <NavLink to="/" end className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}>
          <LayoutDashboard size={17} />Overview</NavLink>
        <NavLink to="/runs" className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}>
          <ListTree size={17} />Runs</NavLink>
        <NavLink to="/memory" className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}>
          <Brain size={17} />Memory</NavLink>
        <NavLink to="/knowledge" className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}>
          <BookOpenText size={17} />Knowledge</NavLink>
        <NavLink to="/tasks" className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}>
          <CalendarClock size={17} />Tasks</NavLink>
      </nav>
      <div className="sidebar-bottom">
        <span className="local-indicator" /> Local instance
        <ArrowUpRight size={13} aria-hidden="true" />
      </div>
    </aside>
    <div className="app-main">
      <header className="topbar">
        <span className="topbar-title">DASHBOARD <span>/</span> V1</span>
        <span className="topbar-note">{status === 'connected' ? 'Live updates connected' :
          status === 'disconnected' ? 'Live updates disconnected · Refresh manually' : 'Connecting live updates…'}</span>
      </header>
      {status === 'disconnected' && <div className="live-disconnected" role="status">Live updates disconnected. Use Refresh; reconnecting automatically.</div>}
      <main className="page-content" id="main-content"><Outlet /></main>
    </div>
  </div>
}
