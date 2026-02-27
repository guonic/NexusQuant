import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import DTWLabelingEntry from './pages/DTWLabelingEntry'

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-eidos-bg">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/dtw-labeling" element={<DTWLabelingEntry />} />
        </Routes>
      </div>
    </Router>
  )
}

export default App

