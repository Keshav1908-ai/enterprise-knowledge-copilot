import { FlaskConical, Library, MessagesSquare } from "lucide-react";
import { Link } from "react-router-dom";
import { SERIES, SimpleBarChart } from "../components/charts";
import { fmtMs, fmtNum, fmtPct, fmtRelative, fmtScore } from "../components/format";
import { EmptyState, ErrorNotice, Loading, PageHeader, Panel, StatStrip, StatusBadge } from "../components/ui";
import { useApi } from "../hooks/useApi";
import { api } from "../services/api";

function Health() {
  const { data, error } = useApi(() => api.health(), [], { pollMs: 15000 });
  if (error) return <ErrorNotice error={error} />;
  if (!data) return <Loading label="Checking services" />;
  const rows = [
    ["API", "ok"],
    ["Metadata database", data.checks.database],
    ["Vector database", data.checks.vector_store],
  ];
  const c = data.config;
  return (
    <div className="space-y-4 text-sm">
      <ul className="space-y-2">
        {rows.map(([name, s]) => (
          <li key={name} className="flex items-center justify-between">
            <span>{name}</span>
            <StatusBadge status={s === "ok" ? "ok" : "degraded"} />
          </li>
        ))}
      </ul>
      <dl className="space-y-1.5 border-t border-rule pt-4 text-ink-soft">
        {[
          ["Indexed passages", fmtNum(data.checks.indexed_chunks)],
          ["Answer generation", c.llm_provider === "openai" ? `${c.llm_model}${c.llm_configured ? "" : " (API key missing)"}` : "Extractive, no LLM"],
          ["Embedding model", c.embedding_model],
          ["Search", `${c.retrieval_mode === "hybrid" ? "Hybrid" : "Semantic"}, top ${c.top_k}, threshold ${c.similarity_threshold}`],
          ["Chunking", `${c.chunk_size} characters, ${c.chunk_overlap} overlap`],
        ].map(([k, v]) => (
          <div key={k} className="flex justify-between gap-4">
            <dt>{k}</dt>
            <dd className="truncate text-right font-medium text-ink" title={v}>{v}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

export default function DashboardPage() {
  const { data, error, loading, refresh } = useApi(() => api.dashboard(), [], { pollMs: 10000 });

  if (loading && !data) return <div className="px-6 py-8"><Loading label="Loading overview" /></div>;
  const ing = data?.ingestion;
  const q = data?.queries;
  const ev = data?.latest_evaluation;

  return (
    <div className="mx-auto max-w-6xl space-y-6 px-6 py-8">
      <PageHeader
        title="Overview"
        description="Live figures from the knowledge base, chat traffic and the latest evaluation run."
      />
      <ErrorNotice error={error} onRetry={refresh} />
      {data && (
        <>
          <StatStrip
            items={[
              { label: "Documents", value: fmtNum(ing.total_documents), hint: `${fmtNum(ing.total_chunks)} passages` },
              { label: "Indexed", value: fmtNum(ing.indexed_documents), hint: ing.failed_documents ? `${ing.failed_documents} failed` : ing.processing_documents ? `${ing.processing_documents} in progress` : "All processed" },
              { label: "Questions asked", value: fmtNum(q.total_queries), hint: q.failed_queries ? `${q.failed_queries} failed` : "No failures" },
              { label: "Avg response time", value: fmtMs(q.avg_total_latency_ms), hint: q.p95_total_latency_ms ? `p95 ${fmtMs(q.p95_total_latency_ms)}` : "" },
              { label: "Tokens used", value: fmtNum(q.total_tokens), hint: q.queries_with_token_data ? `${fmtNum(q.total_input_tokens)} in / ${fmtNum(q.total_output_tokens)} out` : "No LLM calls yet" },
              { label: "Retrieval MRR", value: fmtScore(ev?.metrics?.retrieval?.mrr), hint: ev ? `Precision@${ev.metrics.k} ${fmtScore(ev.metrics.retrieval.precision_at_k)}` : "Run an evaluation" },
            ]}
          />

          <div className="grid gap-6 lg:grid-cols-3">
            <Panel title="Questions per day" description="Chat traffic, last 14 days with activity" className="lg:col-span-2">
              <SimpleBarChart
                data={data.timeseries}
                bars={[{ key: "queries", name: "Questions", color: SERIES.blue }, { key: "errors", name: "Errors", color: SERIES.red }]}
                stacked
                height={230}
              />
            </Panel>
            <Panel title="System">
              <Health />
            </Panel>
          </div>

          <div className="grid gap-6 lg:grid-cols-3">
            <Panel
              title="Recent questions"
              className="lg:col-span-2"
              bodyClass=""
              actions={<Link to="/monitoring" className="text-sm font-medium text-action hover:underline">Open monitoring</Link>}
            >
              {data.recent_queries.length === 0 ? (
                <EmptyState icon={MessagesSquare} title="No questions yet" action={<Link to="/chat" className="text-sm font-medium text-action hover:underline">Ask the documents</Link>}>
                  Response times and token usage appear here after the first question.
                </EmptyState>
              ) : (
                <table className="w-full text-sm">
                  <thead className="border-b border-rule text-left text-muted">
                    <tr>
                      <th className="px-5 py-2 font-medium">When</th>
                      <th className="px-3 py-2 font-medium">Result</th>
                      <th className="px-3 py-2 text-right font-medium">Response</th>
                      <th className="px-3 py-2 text-right font-medium">Passages</th>
                      <th className="px-5 py-2 text-right font-medium">Top score</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-rule">
                    {data.recent_queries.map((r) => (
                      <tr key={r.query_id}>
                        <td className="px-5 py-2.5 text-ink-soft">{fmtRelative(r.created_at)}</td>
                        <td className="px-3 py-2.5"><StatusBadge status={r.status} /></td>
                        <td className="px-3 py-2.5 text-right tabular-nums">{fmtMs(r.total_ms)}</td>
                        <td className="px-3 py-2.5 text-right tabular-nums">{r.num_retrieved}</td>
                        <td className="px-5 py-2.5 text-right tabular-nums">{fmtScore(r.max_score)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </Panel>

            <div className="space-y-6">
              <Panel title="Latest evaluation" actions={<Link to="/evaluation" className="text-sm font-medium text-action hover:underline">Details</Link>}>
                {!ev ? (
                  <EmptyState icon={FlaskConical} title="No evaluation yet">Run the benchmark to measure retrieval and answer quality.</EmptyState>
                ) : (
                  <div className="text-sm">
                    <p className="text-ink-soft">{ev.label || "Unlabelled run"}, {fmtRelative(ev.created_at)}</p>
                    {ev.is_synthetic && <p className="mt-1 text-xs text-warn">Synthetic demo benchmark</p>}
                    <dl className="mt-3 grid grid-cols-2 gap-3">
                      {[
                        ["Recall@K", fmtScore(ev.metrics.retrieval.recall_at_k)],
                        ["MRR", fmtScore(ev.metrics.retrieval.mrr)],
                        ["Faithfulness", fmtScore(ev.metrics.generation.faithfulness)],
                        ["Citation coverage", fmtPct(ev.metrics.generation.citation_coverage)],
                      ].map(([k, v]) => (
                        <div key={k}>
                          <dt className="text-muted">{k}</dt>
                          <dd className="text-lg font-semibold tabular-nums">{v}</dd>
                        </div>
                      ))}
                    </dl>
                  </div>
                )}
              </Panel>
              <Panel title="Recent conversations" bodyClass="py-2">
                {data.recent_conversations.length === 0 ? (
                  <p className="px-5 py-3 text-sm text-muted">None yet.</p>
                ) : (
                  <ul>
                    {data.recent_conversations.map((c) => (
                      <li key={c.id}>
                        <Link to={`/chat/${c.id}`} className="block truncate px-5 py-1.5 text-sm hover:text-action">{c.title}</Link>
                      </li>
                    ))}
                  </ul>
                )}
              </Panel>
            </div>
          </div>
          {ing.total_documents === 0 && (
            <Panel>
              <EmptyState icon={Library} title="Start by adding documents" action={<Link to="/knowledge-base" className="text-sm font-medium text-action hover:underline">Go to the knowledge base</Link>}>
                Upload PDFs, Word documents, text or Markdown files. They are indexed automatically.
              </EmptyState>
            </Panel>
          )}
        </>
      )}
    </div>
  );
}
