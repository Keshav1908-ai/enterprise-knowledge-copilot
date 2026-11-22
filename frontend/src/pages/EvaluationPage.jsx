import { FlaskConical, Info, Play } from "lucide-react";
import { Fragment, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { NoData, SERIES } from "../components/charts";
import { fmtDate, fmtMs, fmtPct, fmtScore } from "../components/format";
import { Button, EmptyState, ErrorNotice, Field, inputCls, Loading, PageHeader, Panel, StatusBadge } from "../components/ui";
import { useApi } from "../hooks/useApi";
import { api } from "../services/api";

const METRIC_HELP = {
  "Precision@K": "Share of the top-K retrieved passages that come from a document labelled relevant.",
  "Recall@K": "Share of the labelled relevant documents that appear in the top K.",
  MRR: "Mean reciprocal rank of the first relevant passage. 1.0 means it is always ranked first.",
  "Hit rate": "Share of questions with at least one relevant passage in the top K.",
  Faithfulness: "Share of answer sentences supported by the retrieved passages (automated estimate).",
  "Answer relevance": "Similarity between the question and the answer (automated estimate).",
  "Context relevance": "Similarity between the question and the retrieved passages (automated estimate).",
  "Citation coverage": "Share of answer sentences that carry a valid citation.",
  "Answer F1": "Token overlap between the answer and the reference answer.",
};

function MetricGroup({ title, note, metrics }) {
  return (
    <div>
      <div className="flex items-baseline justify-between gap-3">
        <h3 className="font-semibold">{title}</h3>
        <p className="text-xs text-muted">{note}</p>
      </div>
      <dl className="mt-3 grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-rule bg-rule sm:grid-cols-4">
        {metrics.map(([label, value, fmt = fmtScore]) => (
          <div key={label} className="bg-surface px-4 py-3" title={METRIC_HELP[label]}>
            <dt className="flex items-center gap-1 text-sm text-muted">
              {label}
              {METRIC_HELP[label] && <Info size={12} aria-hidden="true" />}
            </dt>
            <dd className="mt-0.5 text-2xl font-semibold tabular-nums">{fmt(value)}</dd>
            <dd className="mt-1.5 h-1 overflow-hidden rounded-full bg-rule">
              <span className="block h-full bg-action" style={{ width: `${Math.max(0, Math.min(1, value ?? 0)) * 100}%` }} />
            </dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function RunForm({ onRun, running }) {
  const [form, setForm] = useState({ label: "", top_k: "", retrieval_mode: "", judge: "" });
  const set = (k) => (e) => setForm({ ...form, [k]: e.target.value });
  return (
    <form
      className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5 lg:items-end"
      onSubmit={(e) => {
        e.preventDefault();
        const payload = { label: form.label || null };
        if (form.top_k) payload.top_k = Number(form.top_k);
        if (form.retrieval_mode) payload.retrieval_mode = form.retrieval_mode;
        if (form.judge) payload.judge = form.judge;
        onRun(payload);
      }}
    >
      <Field label="Run name">
        <input className={inputCls} value={form.label} onChange={set("label")} placeholder="e.g. hybrid top-5" maxLength={200} />
      </Field>
      <Field label="Top K">
        <input type="number" min="1" max="20" className={inputCls} value={form.top_k} onChange={set("top_k")} placeholder="Server default" />
      </Field>
      <Field label="Search method">
        <select className={inputCls} value={form.retrieval_mode} onChange={set("retrieval_mode")}>
          <option value="">Server default</option>
          <option value="vector">Semantic</option>
          <option value="hybrid">Hybrid</option>
        </select>
      </Field>
      <Field label="Answer judge">
        <select className={inputCls} value={form.judge} onChange={set("judge")}>
          <option value="">Server default</option>
          <option value="heuristic">Heuristic (deterministic)</option>
          <option value="llm">LLM judge</option>
        </select>
      </Field>
      <Button type="submit" icon={Play} disabled={running}>
        {running ? "Running benchmark…" : "Run benchmark"}
      </Button>
    </form>
  );
}

function ComparisonChart({ runs }) {
  const data = runs
    .filter((r) => r.status === "completed" && r.metrics)
    .slice(0, 8)
    .reverse()
    .map((r) => ({
      name: r.label || fmtDate(r.created_at),
      "Precision@K": r.metrics.retrieval.precision_at_k,
      "Recall@K": r.metrics.retrieval.recall_at_k,
      MRR: r.metrics.retrieval.mrr,
      Faithfulness: r.metrics.generation.faithfulness,
    }));
  if (data.length === 0) return <NoData>Completed runs appear here for comparison.</NoData>;
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid stroke="#EEF1F5" vertical={false} />
        <XAxis dataKey="name" stroke="#6B7690" fontSize={12} tickLine={false} axisLine={{ stroke: "#DDE2EA" }} />
        <YAxis domain={[0, 1]} stroke="#6B7690" fontSize={12} tickLine={false} axisLine={false} width={36} />
        <Tooltip formatter={(v) => fmtScore(v, 3)} contentStyle={{ borderRadius: 8, fontSize: 12 }} cursor={{ fill: "#F2F4F8" }} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Bar dataKey="Precision@K" fill={SERIES.blue} radius={[3, 3, 0, 0]} maxBarSize={22} />
        <Bar dataKey="Recall@K" fill={SERIES.ink} radius={[3, 3, 0, 0]} maxBarSize={22} />
        <Bar dataKey="MRR" fill={SERIES.gold} radius={[3, 3, 0, 0]} maxBarSize={22} />
        <Bar dataKey="Faithfulness" fill={SERIES.slate} radius={[3, 3, 0, 0]} maxBarSize={22} />
      </BarChart>
    </ResponsiveContainer>
  );
}

function QuestionTable({ results }) {
  const [open, setOpen] = useState(null);
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="border-b border-rule text-left text-muted">
          <tr>
            <th className="px-5 py-2 font-medium">Question</th>
            <th className="px-3 py-2 text-right font-medium">P@K</th>
            <th className="px-3 py-2 text-right font-medium">RR</th>
            <th className="px-3 py-2 text-right font-medium">Faithful</th>
            <th className="px-3 py-2 text-right font-medium">Cited</th>
            <th className="px-5 py-2 font-medium">Outcome</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-rule">
          {results.map((r) => {
            const outcome = r.error
              ? ["error", "Error"]
              : !r.answerable
                ? r.abstention_correct ? ["ok", "Correctly declined"] : ["bad", "Answered unanswerable"]
                : r.retrieval?.hit ? ["ok", "Relevant passage found"] : ["bad", "Missed relevant document"];
            const tone = { ok: "text-ok", bad: "text-bad", error: "text-bad" }[outcome[0]];
            return (
              <Fragment key={r.id}>
                <tr className="cursor-pointer hover:bg-paper/70" onClick={() => setOpen(open === r.id ? null : r.id)}>
                  <td className="max-w-md px-5 py-2.5">
                    <button className="text-left hover:text-action" aria-expanded={open === r.id}>{r.question}</button>
                  </td>
                  <td className="px-3 py-2.5 text-right tabular-nums">{fmtScore(r.retrieval?.precision_at_k)}</td>
                  <td className="px-3 py-2.5 text-right tabular-nums">{fmtScore(r.retrieval?.reciprocal_rank)}</td>
                  <td className="px-3 py-2.5 text-right tabular-nums">{fmtScore(r.generation?.faithfulness)}</td>
                  <td className="px-3 py-2.5 text-right tabular-nums">{fmtPct(r.generation?.citation_coverage)}</td>
                  <td className={`px-5 py-2.5 ${tone}`}>{outcome[1]}</td>
                </tr>
                {open === r.id && (
                    <tr className="bg-paper/60">
                    <td colSpan={6} className="px-5 py-4">
                      <div className="grid gap-5 md:grid-cols-2">
                        <div>
                          <p className="text-xs font-medium text-muted">System answer</p>
                          <p className="mt-1 whitespace-pre-line font-serif text-[15px]">{r.answer || r.error}</p>
                          {r.expected_answer && (
                            <>
                              <p className="mt-3 text-xs font-medium text-muted">Reference answer</p>
                              <p className="mt-1 font-serif text-[15px] text-ink-soft">{r.expected_answer}</p>
                            </>
                          )}
                        </div>
                        <div>
                          <p className="text-xs font-medium text-muted">Retrieved passages (labelled relevant: {r.relevant_documents.join(", ") || "none"})</p>
                          <ol className="mt-1 space-y-1">
                            {(r.retrieved || []).map((c, i) => (
                              <li key={c.chunk_id} className="flex items-center gap-2">
                                <span className="w-4 text-right text-xs text-muted">{i + 1}</span>
                                <span className={`h-2 w-2 rounded-full ${c.relevant ? "bg-ok" : "bg-rule-strong"}`} aria-label={c.relevant ? "relevant" : "not relevant"} />
                                <span className="truncate">{c.document_name}</span>
                                <span className="ml-auto tabular-nums text-xs text-muted">{fmtScore(c.score)}</span>
                              </li>
                            ))}
                          </ol>
                          {r.latency && <p className="mt-3 text-xs text-muted">Answered in {fmtMs(r.latency.total_ms)}</p>}
                        </div>
                      </div>
                    </td>
                  </tr>
                )}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default function EvaluationPage() {
  const results = useApi(() => api.evaluationResults(), []);
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState(null);
  const [selectedId, setSelectedId] = useState(null);
  const selected = useApi(() => api.evaluationRun(selectedId), [selectedId], { enabled: !!selectedId });

  async function run(payload) {
    setRunning(true);
    setRunError(null);
    try {
      const r = await api.runEvaluation(payload);
      setSelectedId(null);
      await results.refresh();
      if (r.status === "failed") setRunError({ message: r.error || "Evaluation failed" });
    } catch (e) {
      setRunError(e);
    } finally {
      setRunning(false);
    }
  }

  const data = results.data;
  const run_ = selectedId ? selected.data : data?.latest;
  const m = run_?.metrics;

  return (
    <div className="mx-auto max-w-6xl space-y-6 px-6 py-8">
      <PageHeader
        title="Evaluation"
        description="Runs a labelled question set through the live pipeline and scores retrieval and answers. Change the settings to compare configurations."
      />
      <ErrorNotice error={results.error} onRetry={results.refresh} />

      <Panel title="Run the benchmark" description={data?.dataset?.available ? `${data.dataset.name}: ${data.dataset.num_questions} questions over ${data.dataset.documents.length} documents` : undefined}>
        {data?.dataset?.synthetic && (
          <p className="mb-4 rounded-md bg-warn-tint px-3 py-2 text-sm text-warn">
            This benchmark uses fictional sample policies written for the project, so scores describe the demo corpus, not your own documents.
            Missing benchmark documents are indexed automatically before the run.
          </p>
        )}
        <RunForm onRun={run} running={running} />
        {running && <p className="mt-3 text-sm text-muted">Each question goes through retrieval and generation. With an LLM this can take a minute.</p>}
        {runError && <div className="mt-3"><ErrorNotice error={runError} /></div>}
      </Panel>

      {results.loading && !data ? (
        <Loading label="Loading evaluation results" />
      ) : !run_ ? (
        <Panel>
          <EmptyState icon={FlaskConical} title="No evaluation results yet">Run the benchmark above to get Precision@K, Recall@K, MRR and answer quality scores.</EmptyState>
        </Panel>
      ) : (
        <Panel
          title={run_.label || "Evaluation run"}
          description={`${fmtDate(run_.created_at)}, ${run_.num_questions} questions, ${fmtMs(run_.duration_ms)}, search ${run_.config.mode}, top ${run_.config.top_k}, ${run_.config.llm_model}`}
          actions={<StatusBadge status={run_.status} />}
        >
          {m ? (
            <div className="space-y-7">
              <MetricGroup
                title={`Retrieval at K = ${m.k}`}
                note="Deterministic, compared with labelled relevant documents"
                metrics={[["Precision@K", m.retrieval.precision_at_k], ["Recall@K", m.retrieval.recall_at_k], ["MRR", m.retrieval.mrr], ["Hit rate", m.retrieval.hit_rate]]}
              />
              <MetricGroup
                title="Answer quality"
                note={`Automated estimates (judge: ${m.generation.judge}), not human-verified`}
                metrics={[["Faithfulness", m.generation.faithfulness], ["Answer relevance", m.generation.answer_relevance], ["Citation coverage", m.generation.citation_coverage, (v) => fmtPct(v)], ["Answer F1", m.generation.answer_f1]]}
              />
              <div className="grid gap-4 text-sm sm:grid-cols-3">
                <p><span className="text-muted">Unanswerable questions declined: </span><span className="font-semibold">{m.abstention.correct_abstentions} of {m.abstention.unanswerable_questions}</span></p>
                <p><span className="text-muted">Avg retrieval latency: </span><span className="font-semibold">{fmtMs(m.retrieval.avg_retrieval_latency_ms)}</span></p>
                <p><span className="text-muted">Avg total latency: </span><span className="font-semibold">{fmtMs(m.performance.avg_total_latency_ms)}</span></p>
              </div>
              {run_.results && (
                <div className="-mx-5 border-t border-rule">
                  <p className="px-5 pt-4 font-semibold">Per-question results</p>
                  <p className="px-5 text-sm text-muted">Select a question to compare the answer with the reference.</p>
                  <div className="mt-2"><QuestionTable results={run_.results} /></div>
                </div>
              )}
            </div>
          ) : (
            <p className="text-sm text-bad">{run_.error}</p>
          )}
        </Panel>
      )}

      <div className="grid gap-6 lg:grid-cols-5">
        <Panel title="Configuration comparison" className="lg:col-span-3">
          <ComparisonChart runs={data?.runs || []} />
        </Panel>
        <Panel title="Run history" className="lg:col-span-2" bodyClass="py-1">
          {(data?.runs || []).length === 0 ? (
            <p className="px-5 py-4 text-sm text-muted">No runs yet.</p>
          ) : (
            <ul className="divide-y divide-rule">
              {data.runs.map((r) => (
                <li key={r.id}>
                  <button
                    onClick={() => setSelectedId(r.id)}
                    className={`flex w-full items-center gap-3 px-5 py-2.5 text-left text-sm hover:bg-paper ${run_?.id === r.id ? "bg-action-tint/60" : ""}`}
                  >
                    <span className="min-w-0 flex-1">
                      <span className="block truncate font-medium">{r.label || "Unlabelled run"}</span>
                      <span className="text-xs text-muted">{fmtDate(r.created_at)}, {r.config.mode}, top {r.config.top_k}</span>
                    </span>
                    <span className="text-right tabular-nums">
                      {r.metrics ? <>MRR {fmtScore(r.metrics.retrieval.mrr)}</> : <StatusBadge status={r.status} />}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </Panel>
      </div>
    </div>
  );
}
