import { Activity } from "lucide-react";
import { useState } from "react";
import { NoData, SERIES, SimpleBarChart, TimeLineChart } from "../components/charts";
import { fmtMs, fmtNum, fmtPct, fmtRelative, fmtScore } from "../components/format";
import { EmptyState, ErrorNotice, inputCls, Loading, PageHeader, Panel, StatStrip, StatusBadge } from "../components/ui";
import { useApi } from "../hooks/useApi";
import { api } from "../services/api";

const WINDOWS = [
  { label: "Last 24 hours", hours: 24, bucket: "hour" },
  { label: "Last 7 days", hours: 168, bucket: "hour" },
  { label: "Last 30 days", hours: 720, bucket: "day" },
];

export default function MonitoringPage() {
  const [win, setWin] = useState(1);
  const [source, setSource] = useState("chat");
  const w = WINDOWS[win];
  const { data, error, loading, refresh } = useApi(() => api.metrics({ hours: w.hours, bucket: w.bucket, source }), [win, source], { pollMs: 10000 });
  const s = data?.summary;

  return (
    <div className="mx-auto max-w-6xl space-y-6 px-6 py-8">
      <PageHeader
        title="Monitoring"
        description="Measured on every request: retrieval and generation time, tokens, passages retrieved and their relevance scores."
        actions={
          <>
            <label className="sr-only" htmlFor="src">Traffic</label>
            <select id="src" className={`${inputCls} w-40`} value={source} onChange={(e) => setSource(e.target.value)}>
              <option value="chat">Chat traffic</option>
              <option value="evaluation">Evaluation runs</option>
              <option value="all">All traffic</option>
            </select>
            <label className="sr-only" htmlFor="win">Time window</label>
            <select id="win" className={`${inputCls} w-40`} value={win} onChange={(e) => setWin(Number(e.target.value))}>
              {WINDOWS.map((x, i) => <option key={x.label} value={i}>{x.label}</option>)}
            </select>
          </>
        }
      />
      <ErrorNotice error={error} onRetry={refresh} />
      {loading && !data && <Loading label="Loading metrics" />}
      {data && (
        <>
          <StatStrip
            items={[
              { label: "Requests", value: fmtNum(s.total_queries), hint: `${fmtNum(s.no_context_queries)} without evidence` },
              { label: "Error rate", value: fmtPct(s.error_rate, 1), hint: `${fmtNum(s.failed_queries)} failed` },
              { label: "Avg response", value: fmtMs(s.avg_total_latency_ms), hint: `p95 ${fmtMs(s.p95_total_latency_ms)}` },
              { label: "Avg retrieval", value: fmtMs(s.avg_retrieval_latency_ms), hint: `p95 ${fmtMs(s.p95_retrieval_latency_ms)}` },
              { label: "Avg generation", value: fmtMs(s.avg_generation_latency_ms), hint: `p95 ${fmtMs(s.p95_generation_latency_ms)}` },
              { label: "Tokens", value: fmtNum(s.total_tokens), hint: s.queries_with_token_data ? `${fmtNum(s.avg_tokens_per_query)} per request` : "No LLM token data" },
            ]}
          />

          {s.total_queries === 0 ? (
            <Panel>
              <EmptyState icon={Activity} title="No requests in this window">Ask a question in chat or run an evaluation; measurements appear here within seconds.</EmptyState>
            </Panel>
          ) : (
            <>
              <Panel title="Response latency" description="Average per time bucket. Generation is the LLM call; retrieval includes query embedding and vector search.">
                <TimeLineChart
                  data={data.timeseries}
                  unit=" ms"
                  lines={[
                    { key: "avg_total_ms", name: "Total", color: SERIES.ink },
                    { key: "avg_generation_ms", name: "Generation", color: SERIES.blue },
                    { key: "avg_retrieval_ms", name: "Retrieval", color: SERIES.gold },
                  ]}
                />
              </Panel>
              <div className="grid gap-6 lg:grid-cols-2">
                <Panel title="Requests and errors">
                  <SimpleBarChart data={data.timeseries} stacked bars={[{ key: "queries", name: "Requests", color: SERIES.blue }, { key: "errors", name: "Errors", color: SERIES.red }]} />
                </Panel>
                <Panel title="Token usage" description={`${fmtNum(s.total_input_tokens)} input and ${fmtNum(s.total_output_tokens)} output tokens`}>
                  {s.queries_with_token_data ? (
                    <SimpleBarChart data={data.timeseries} bars={[{ key: "tokens", name: "Tokens", color: SERIES.ink }]} />
                  ) : (
                    <NoData height={240}>No LLM tokens recorded. Extractive mode answers without calling an LLM.</NoData>
                  )}
                </Panel>
              </div>
              <div className="grid gap-6 lg:grid-cols-3">
                <Panel title="Relevance scores" description={`Cosine similarity of retrieved passages. Average ${fmtScore(s.avg_relevance_score)}, ${fmtScore(s.avg_retrieved_chunks, 1)} passages per request.`}>
                  <SimpleBarChart data={data.score_histogram} xKey="range" xFormatter={(v) => v} bars={[{ key: "count", name: "Passages", color: SERIES.gold }]} height={220} />
                </Panel>
                <Panel title="Recent requests" className="lg:col-span-2" bodyClass="">
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead className="border-b border-rule text-left text-muted">
                        <tr>
                          <th className="px-5 py-2 font-medium">When</th>
                          <th className="px-3 py-2 font-medium">Result</th>
                          <th className="px-3 py-2 text-right font-medium">Retrieval</th>
                          <th className="px-3 py-2 text-right font-medium">Generation</th>
                          <th className="px-3 py-2 text-right font-medium">Tokens</th>
                          <th className="px-5 py-2 text-right font-medium">Top score</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-rule">
                        {data.recent_queries.map((r) => (
                          <tr key={r.query_id} title={r.error || r.llm_model || ""}>
                            <td className="whitespace-nowrap px-5 py-2 text-ink-soft">{fmtRelative(r.created_at)}</td>
                            <td className="px-3 py-2"><StatusBadge status={r.status} /></td>
                            <td className="px-3 py-2 text-right tabular-nums">{fmtMs(r.retrieval_ms)}</td>
                            <td className="px-3 py-2 text-right tabular-nums">{fmtMs(r.generation_ms)}</td>
                            <td className="px-3 py-2 text-right tabular-nums">{fmtNum(r.total_tokens)}</td>
                            <td className="px-5 py-2 text-right tabular-nums">{fmtScore(r.max_score)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Panel>
              </div>
            </>
          )}

          <div className="grid gap-6 lg:grid-cols-2">
            <Panel title="API layer" description="Write requests and errors recorded by the request middleware">
              <dl className="grid grid-cols-3 gap-4 text-sm">
                <div><dt className="text-muted">Requests</dt><dd className="text-xl font-semibold tabular-nums">{fmtNum(data.api.total_requests)}</dd></div>
                <div><dt className="text-muted">4xx / 5xx</dt><dd className="text-xl font-semibold tabular-nums">{fmtNum(data.api.error_responses - data.api.server_errors)} / {fmtNum(data.api.server_errors)}</dd></div>
                <div><dt className="text-muted">Avg duration</dt><dd className="text-xl font-semibold tabular-nums">{fmtMs(data.api.avg_duration_ms)}</dd></div>
              </dl>
            </Panel>
            <Panel title="Document ingestion">
              <dl className="grid grid-cols-3 gap-4 text-sm">
                <div><dt className="text-muted">Indexed</dt><dd className="text-xl font-semibold tabular-nums">{fmtNum(data.ingestion.indexed_documents)} / {fmtNum(data.ingestion.total_documents)}</dd></div>
                <div><dt className="text-muted">Passages</dt><dd className="text-xl font-semibold tabular-nums">{fmtNum(data.ingestion.total_chunks)}</dd></div>
                <div><dt className="text-muted">Avg processing</dt><dd className="text-xl font-semibold tabular-nums">{fmtMs(data.ingestion.avg_processing_time_ms)}</dd></div>
              </dl>
            </Panel>
          </div>
        </>
      )}
    </div>
  );
}
