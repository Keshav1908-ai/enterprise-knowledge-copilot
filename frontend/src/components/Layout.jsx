import { Activity, FlaskConical, Gauge, Library, Menu, MessagesSquare, X } from "lucide-react";
import { useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { api } from "../services/api";
import { useApi } from "../hooks/useApi";

const NAV = [
  { to: "/", label: "Overview", icon: Gauge, end: true },
  { to: "/chat", label: "Ask the documents", icon: MessagesSquare },
  { to: "/knowledge-base", label: "Knowledge base", icon: Library },
  { to: "/evaluation", label: "Evaluation", icon: FlaskConical },
  { to: "/monitoring", label: "Monitoring", icon: Activity },
];

function Brand() {
  return (
    <div className="flex items-center gap-2.5 px-5 py-5">
      <svg width="28" height="28" viewBox="0 0 32 32" aria-hidden="true">
        <rect width="32" height="32" rx="7" fill="#F7F8FA" fillOpacity=".08" />
        <rect x="8" y="9" width="16" height="3" rx="1.5" fill="#F7F8FA" />
        <rect x="8" y="15" width="11" height="3" rx="1.5" fill="#F2D05E" />
        <rect x="8" y="21" width="14" height="3" rx="1.5" fill="#F7F8FA" />
      </svg>
      <div className="leading-tight">
        <p className="font-semibold text-white">Knowledge Copilot</p>
        <p className="text-xs text-white/55">Answers with sources</p>
      </div>
    </div>
  );
}

function SystemStatus() {
  const { data, error } = useApi(() => api.health(), [], { pollMs: 30000 });
  const ok = data?.status === "ok" && !error;
  const cfg = data?.config;
  return (
    <div className="mx-3 mb-4 rounded-md bg-white/5 p-3 text-xs text-white/70">
      <p className="flex items-center gap-2 font-medium text-white/90">
        <span className={`h-2 w-2 rounded-full ${ok ? "bg-[#5BC28A]" : error ? "bg-[#E8735F]" : "bg-white/40"}`} />
        {error ? "API unreachable" : ok ? "All systems running" : "Checking…"}
      </p>
      {cfg && (
        <dl className="mt-2 space-y-0.5">
          <div className="flex justify-between gap-2"><dt>Model</dt><dd className="truncate text-right text-white/90" title={cfg.llm_model}>{cfg.llm_provider === "openai" ? cfg.llm_model : "Extractive (local)"}</dd></div>
          <div className="flex justify-between gap-2"><dt>Embeddings</dt><dd className="truncate text-right text-white/90" title={cfg.embedding_model}>{cfg.embedding_provider === "hash" ? "Hash (offline)" : cfg.embedding_model.split("/").pop()}</dd></div>
          <div className="flex justify-between gap-2"><dt>Retrieval</dt><dd className="text-white/90">{cfg.retrieval_mode}, top {cfg.top_k}</dd></div>
        </dl>
      )}
    </div>
  );
}

export default function Layout() {
  const [open, setOpen] = useState(false);
  const nav = (
    <nav className="flex-1 space-y-0.5 px-3" aria-label="Main">
      {NAV.map(({ to, label, icon: Icon, end }) => (
        <NavLink
          key={to}
          to={to}
          end={end}
          onClick={() => setOpen(false)}
          className={({ isActive }) =>
            `flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors ${
              isActive ? "bg-white/12 font-medium text-white" : "text-white/70 hover:bg-white/6 hover:text-white"
            }`
          }
        >
          <Icon size={17} aria-hidden="true" />
          {label}
        </NavLink>
      ))}
    </nav>
  );
  return (
    <div className="flex h-full">
      <aside className="hidden w-60 shrink-0 flex-col bg-ink lg:flex">
        <Brand />
        {nav}
        <SystemStatus />
      </aside>

      {open && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <button className="absolute inset-0 bg-ink/40" aria-label="Close menu" onClick={() => setOpen(false)} />
          <aside className="relative flex h-full w-64 flex-col bg-ink">
            <div className="flex items-center justify-between pr-3">
              <Brand />
              <button onClick={() => setOpen(false)} className="text-white/80" aria-label="Close menu"><X size={20} /></button>
            </div>
            {nav}
            <SystemStatus />
          </aside>
        </div>
      )}

      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center gap-3 border-b border-rule bg-surface px-4 py-3 lg:hidden">
          <button onClick={() => setOpen(true)} aria-label="Open menu"><Menu size={20} /></button>
          <span className="font-semibold">Knowledge Copilot</span>
        </div>
        <main className="min-h-0 flex-1 overflow-y-auto scroll-thin">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
