import { lazy, StrictMode, Suspense } from 'react'
import { createRoot } from 'react-dom/client'
import { HashRouter, Navigate, Route, Routes } from 'react-router-dom'

import { DashboardLayout } from './components/DashboardLayout'
import { LoadingState } from './components/States'
import './styles.css'

const Overview = lazy(() => import('./pages/Overview').then(module => ({ default: module.Overview })))
const Runs = lazy(() => import('./pages/Runs').then(module => ({ default: module.Runs })))
const Memory = lazy(() => import('./pages/Memory').then(module => ({ default: module.Memory })))

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <HashRouter>
      <Suspense fallback={<LoadingState label="Loading dashboard…" />}>
        <Routes>
          <Route element={<DashboardLayout />}>
            <Route index element={<Overview />} />
            <Route path="runs" element={<Runs />} />
            <Route path="runs/:runId" element={<Runs />} />
            <Route path="memory" element={<Memory />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Route>
        </Routes>
      </Suspense>
    </HashRouter>
  </StrictMode>,
)
