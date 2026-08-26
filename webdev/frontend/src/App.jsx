import { HashRouter, Routes, Route } from "react-router-dom";
import Sidebar from "./components/Sidebar.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import Models from "./pages/Models.jsx";
import Architecture from "./pages/Architecture.jsx";

// App shell: a collapsible sidebar + a routed main content area.
// HashRouter is used so client-side routes work on static hosting (Vercel) with no
// extra rewrite config — refreshing /models never 404s.
export default function App() {
  return (
    <HashRouter>
      <div className="flex min-h-screen">
        <Sidebar />
        <main className="flex-1 min-w-0 p-6 md:p-8 overflow-x-hidden">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/models" element={<Models />} />
            <Route path="/architecture" element={<Architecture />} />
          </Routes>
        </main>
      </div>
    </HashRouter>
  );
}

