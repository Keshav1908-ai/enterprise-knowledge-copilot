import { lazy, Suspense } from "react";
import { Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import { Loading } from "./components/ui";

// Route-level code splitting keeps the initial bundle small (charts load with their pages).
const ChatPage = lazy(() => import("./pages/ChatPage"));
const DashboardPage = lazy(() => import("./pages/DashboardPage"));
const EvaluationPage = lazy(() => import("./pages/EvaluationPage"));
const KnowledgeBasePage = lazy(() => import("./pages/KnowledgeBasePage"));
const MonitoringPage = lazy(() => import("./pages/MonitoringPage"));

function NotFound() {
  return (
    <div className="p-10">
      <h1 className="text-xl font-semibold">Page not found</h1>
      <p className="mt-1 text-ink-soft">Use the navigation to get back to the workspace.</p>
    </div>
  );
}

export default function App() {
  return (
    <Suspense fallback={<div className="px-6 py-8"><Loading /></div>}>
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<DashboardPage />} />
        <Route path="chat" element={<ChatPage />} />
        <Route path="chat/:conversationId" element={<ChatPage />} />
        <Route path="knowledge-base" element={<KnowledgeBasePage />} />
        <Route path="evaluation" element={<EvaluationPage />} />
        <Route path="monitoring" element={<MonitoringPage />} />
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
    </Suspense>
  );
}
