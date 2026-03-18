import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import MyCv from './pages/MyCv'
import ScoreJD from './pages/ScoreJD'
import Jobs from './pages/Jobs'
import JobDetails from './pages/JobDetails'
import InterviewPrep from './pages/InterviewPrep'
import Settings from './pages/Settings'
import SolvePractice from './pages/SolvePractice'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Navigate to="/dashboard" replace />} />
        <Route path="dashboard" element={<Dashboard />} />
        <Route path="cv" element={<MyCv />} />
        <Route path="score" element={<ScoreJD />} />
        <Route path="jobs" element={<Jobs />} />
        <Route path="jobs/:id" element={<JobDetails />} />
        <Route path="jobs/:id/interview-prep" element={<InterviewPrep />} />
        <Route path="practice/coach" element={<SolvePractice />} />
        <Route path="settings" element={<Settings />} />
      </Route>
    </Routes>
  )
}
