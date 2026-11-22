import { FileText, Loader2, MessageSquarePlus, SendHorizontal, SlidersHorizontal, Trash2 } from "lucide-react";
import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { fmtMs, fmtNum, fmtRelative, fmtScore } from "../components/format";
import { ErrorNotice, Field, inputCls, ScoreBar } from "../components/ui";
import { useApi } from "../hooks/useApi";
import { api } from "../services/api";

const STARTERS = [
  "What is the company's annual leave policy?",
  "How many casual leaves are allowed per year?",
  "When must a lost laptop be reported?",
  "What is the daily meal allowance for international travel?",
];

// ---- answer rendering --------------------------------------------------------
function renderInline(text, onCite, activeKey, msgIndex) {
  const parts = text.split(/(\[\d+(?:\s*,\s*\d+)*\]|\*\*[^*]+\*\*)/g);
  return parts.map((part, i) => {
    const cite = part.match(/^\[(\d+(?:\s*,\s*\d+)*)\]$/);
    if (cite) {
      return (
        <span key={i} className="whitespace-nowrap">
          {cite[1].split(",").map((n) => {
            const id = Number(n.trim());
            const key = `${msgIndex}:${id}`;
            return (
              <button
                key={id}
                onClick={() => onCite(msgIndex, id)}
                aria-label={`Show source ${id}`}
                className={`ml-1 inline-flex h-[1.15rem] min-w-[1.15rem] -translate-y-px items-center justify-center rounded px-1 align-middle font-sans text-[11px] font-semibold transition-colors ${
                  activeKey === key ? "bg-marker text-ink" : "bg-action-tint text-action hover:bg-marker hover:text-ink"
                }`}
              >
                {id}
              </button>
            );
          })}
        </span>
      );
    }
    if (part.startsWith("**") && part.endsWith("**")) return <strong key={i}>{part.slice(2, -2)}</strong>;
    return <Fragment key={i}>{part}</Fragment>;
  });
}

function AnswerText({ text, onCite, activeKey, msgIndex }) {
  const blocks = [];
  let list = null;
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    const bullet = line.match(/^[-*•]\s+(.*)$/) || line.match(/^\d+\.\s+(.*)$/);
    if (bullet) {
      if (!list) {
        list = [];
        blocks.push({ type: "ul", items: list });
      }
      list.push(bullet[1]);
    } else {
      list = null;
      if (line) blocks.push({ type: "p", text: line.replace(/^#+\s*/, "") });
    }
  }
  return (
    <div className="answer-prose font-serif text-[16.5px] leading-[1.65] text-ink">
      {blocks.map((b, i) =>
        b.type === "ul" ? (
          <ul key={i}>
            {b.items.map((it, j) => (
              <li key={j}>{renderInline(it, onCite, activeKey, msgIndex)}</li>
            ))}
          </ul>
        ) : (
          <p key={i}>{renderInline(b.text, onCite, activeKey, msgIndex)}</p>
        ),
      )}
    </div>
  );
}

function AnswerMeta({ m }) {
  const lat = m.metrics?.latency;
  const tok = m.metrics?.token_usage;
  const mode = m.metrics?.generation_mode;
  return (
    <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted">
      {m.metrics?.status === "no_context" && (
        <span className="rounded bg-warn-tint px-1.5 py-0.5 font-medium text-warn">No supporting passage found</span>
      )}
      {lat && (
        <span title={`Retrieval ${fmtMs(lat.retrieval_ms)}, generation ${fmtMs(lat.generation_ms)}`}>
          {fmtMs(lat.total_ms)} total, {fmtMs(lat.retrieval_ms)} retrieval
        </span>
      )}
      <span>{tok?.total_tokens != null ? `${fmtNum(tok.total_tokens)} tokens` : "No LLM tokens used"}</span>
      {mode && <span>{mode === "extractive" ? "Extractive mode (no LLM)" : `Model: ${mode}`}</span>}
      {m.citations?.length > 0 && <span>{m.citations.length} cited source{m.citations.length > 1 ? "s" : ""}</span>}
    </div>
  );
}

// ---- evidence panel ----------------------------------------------------------
const stripMd = (s) => s.replace(/^#{1,6}\s*/gm, "");

function highlight(rawText, excerpt) {
  const text = stripMd(rawText);
  if (!excerpt) return text;
  const norm = (s) => s.replace(/\s+/g, " ");
  const t = norm(text);
  const ex = norm(excerpt).replace(/…$/, "");
  const at = t.indexOf(ex.slice(0, 80));
  if (at < 0) return t;
  const end = Math.min(t.length, at + ex.length);
  return (
    <>
      {t.slice(0, at)}
      <mark className="evidence-mark text-ink">{t.slice(at, end)}</mark>
      {t.slice(end)}
    </>
  );
}

function EvidencePanel({ message, activeSource, onSelect }) {
  if (!message) {
    return (
      <div className="p-6 text-sm text-muted">
        <p className="font-medium text-ink">Evidence</p>
        <p className="mt-1">Select a citation number in an answer to see the exact passage it came from.</p>
      </div>
    );
  }
  const chunks = message.retrieved_chunks || [];
  const cited = new Set((message.citations || []).map((c) => c.source_id));
  const citation = (message.citations || []).find((c) => c.source_id === activeSource);
  const chunk = chunks.find((c) => c.source_id === activeSource);
  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-rule px-5 py-4">
        <p className="font-semibold">Evidence</p>
        <p className="text-sm text-muted">
          {chunks.length ? `${chunks.length} passages retrieved, ${cited.size} cited` : "No passages passed the relevance threshold"}
        </p>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto scroll-thin">
        {chunk && (
          <article className="border-b border-rule bg-paper/60 px-5 py-5">
            <div className="flex items-start gap-2">
              <FileText size={18} className="mt-0.5 shrink-0 text-action" aria-hidden="true" />
              <div className="min-w-0">
                <p className="truncate font-medium" title={chunk.document_name}>{chunk.document_name}</p>
                <p className="text-xs text-muted">
                  Source {chunk.source_id}
                  {chunk.page_number ? `, page ${chunk.page_number}` : ""}, relevance {fmtScore(chunk.score)}
                </p>
              </div>
            </div>
            <blockquote className="mt-4 font-serif text-[15px] leading-relaxed text-ink-soft">
              {highlight(chunk.text, citation?.excerpt)}
            </blockquote>
            <p className="mt-3 break-all text-[11px] text-muted">Chunk {chunk.chunk_id}</p>
          </article>
        )}
        <ol className="divide-y divide-rule">
          {chunks.map((c) => (
            <li key={c.chunk_id}>
              <button
                onClick={() => onSelect(c.source_id)}
                className={`block w-full px-5 py-3 text-left transition-colors hover:bg-paper ${activeSource === c.source_id ? "bg-action-tint/60" : ""}`}
              >
                <div className="flex items-center gap-2 text-sm">
                  <span
                    className={`inline-flex h-5 min-w-5 items-center justify-center rounded px-1 text-[11px] font-semibold ${
                      cited.has(c.source_id) ? "bg-marker text-ink" : "bg-rule text-ink-soft"
                    }`}
                  >
                    {c.source_id}
                  </span>
                  <span className="min-w-0 flex-1 truncate font-medium">{c.document_name}</span>
                  {c.page_number && <span className="text-xs text-muted">p. {c.page_number}</span>}
                </div>
                <div className="mt-1.5 flex items-center gap-2 text-xs text-muted">
                  <ScoreBar value={c.score} /> <span className="tabular-nums">{fmtScore(c.score)}</span>
                  {c.keyword_score != null && <span>keyword {fmtScore(c.keyword_score)}</span>}
                  {!cited.has(c.source_id) && <span>not cited</span>}
                </div>
                <p className="mt-1.5 line-clamp-2 text-sm text-ink-soft">{stripMd(c.text)}</p>
              </button>
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}

// ---- page --------------------------------------------------------------------
export default function ChatPage() {
  const { conversationId } = useParams();
  const navigate = useNavigate();
  const [messages, setMessages] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [question, setQuestion] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState(null);
  const [selected, setSelected] = useState({ msg: null, source: null });
  const [showOptions, setShowOptions] = useState(false);
  const [options, setOptions] = useState({ top_k: "", retrieval_mode: "" });
  const bottomRef = useRef(null);
  const convs = useApi(() => api.listConversations(), []);
  const docs = useApi(() => api.listDocuments(), []);

  // Load a conversation when the URL changes (skip when we just created it ourselves).
  useEffect(() => {
    if (!conversationId) {
      setMessages([]);
      setActiveId(null);
      setSelected({ msg: null, source: null });
      return;
    }
    if (conversationId === activeId) return;
    let cancelled = false;
    api
      .getConversation(conversationId)
      .then((c) => {
        if (cancelled) return;
        setMessages(c.messages);
        setActiveId(c.id);
        const lastAssistant = c.messages.map((m, i) => [m, i]).filter(([m]) => m.role === "assistant").pop();
        setSelected({ msg: lastAssistant ? lastAssistant[1] : null, source: lastAssistant?.[0].citations?.[0]?.source_id ?? null });
      })
      .catch((e) => !cancelled && setError(e));
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conversationId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length, sending]);

  const indexedCount = docs.data?.counts?.indexed ?? null;

  async function send(text) {
    const q = (text ?? question).trim();
    if (!q || sending) return;
    setError(null);
    setSending(true);
    setQuestion("");
    const userMsg = { id: `local-${Date.now()}`, role: "user", content: q };
    setMessages((m) => [...m, userMsg]);
    try {
      const payload = { question: q, conversation_id: activeId };
      if (options.top_k) payload.top_k = Number(options.top_k);
      if (options.retrieval_mode) payload.retrieval_mode = options.retrieval_mode;
      const res = await api.chat(payload);
      const assistant = {
        id: res.message_id,
        role: "assistant",
        content: res.answer,
        citations: res.citations,
        retrieved_chunks: res.retrieved_chunks,
        standalone_question: res.standalone_question !== q ? res.standalone_question : null,
        metrics: {
          latency: res.latency,
          token_usage: res.token_usage,
          status: res.status,
          generation_mode: res.generation_mode === "openai" ? res.llm_model : res.generation_mode,
          grounding: res.grounding,
        },
      };
      setMessages((m) => {
        const next = [...m, assistant];
        setSelected({ msg: next.length - 1, source: res.citations[0]?.source_id ?? res.retrieved_chunks[0]?.source_id ?? null });
        return next;
      });
      if (!activeId) {
        setActiveId(res.conversation_id);
        navigate(`/chat/${res.conversation_id}`, { replace: true });
      }
      convs.refresh();
    } catch (e) {
      setError(e);
      setMessages((m) => m.filter((x) => x.id !== userMsg.id));
      setQuestion(q);
    } finally {
      setSending(false);
    }
  }

  async function removeConversation(id) {
    if (!window.confirm("Delete this conversation? This can't be undone.")) return;
    try {
      await api.deleteConversation(id);
      convs.refresh();
      if (id === activeId) navigate("/chat");
    } catch (e) {
      setError(e);
    }
  }

  const selectedMessage = selected.msg != null ? messages[selected.msg] : null;
  const activeKey = selected.msg != null ? `${selected.msg}:${selected.source}` : null;
  const conversationList = useMemo(() => convs.data || [], [convs.data]);

  return (
    <div className="flex h-full">
      {/* conversations */}
      <aside className="hidden w-64 shrink-0 flex-col border-r border-rule bg-surface md:flex">
        <div className="p-3">
          <button
            onClick={() => navigate("/chat")}
            className="flex h-9 w-full items-center justify-center gap-2 rounded-md border border-rule-strong text-sm font-medium hover:bg-paper"
          >
            <MessageSquarePlus size={16} aria-hidden="true" /> New conversation
          </button>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-3 scroll-thin">
          {conversationList.length === 0 && <p className="px-3 py-4 text-sm text-muted">Your conversations appear here.</p>}
          {conversationList.map((c) => (
            <div key={c.id} className={`group flex items-center rounded-md ${c.id === activeId ? "bg-action-tint" : "hover:bg-paper"}`}>
              <Link to={`/chat/${c.id}`} className="min-w-0 flex-1 px-3 py-2">
                <p className={`truncate text-sm ${c.id === activeId ? "font-medium text-action" : ""}`}>{c.title}</p>
                <p className="text-xs text-muted">
                  {Math.ceil(c.message_count / 2)} question{c.message_count > 2 ? "s" : ""}, {fmtRelative(c.updated_at)}
                </p>
              </Link>
              <button
                onClick={() => removeConversation(c.id)}
                className="mr-1 rounded p-1.5 text-muted opacity-0 hover:text-bad focus:opacity-100 group-hover:opacity-100"
                aria-label={`Delete conversation ${c.title}`}
              >
                <Trash2 size={14} />
              </button>
            </div>
          ))}
        </div>
      </aside>

      {/* thread */}
      <section className="flex min-w-0 flex-1 flex-col">
        <div className="min-h-0 flex-1 overflow-y-auto scroll-thin">
          <div className="mx-auto max-w-3xl px-5 py-8">
            {messages.length === 0 && !sending && (
              <div className="pt-6">
                <h1 className="text-[26px] font-semibold tracking-tight">Ask the documents</h1>
                <p className="mt-2 max-w-xl text-ink-soft">
                  Answers come only from your knowledge base. Every claim links to the passage it was taken from.
                </p>
                {indexedCount === 0 ? (
                  <div className="mt-6 rounded-lg border border-dashed border-rule-strong bg-surface p-5 text-sm">
                    <p className="font-medium">No documents are indexed yet.</p>
                    <p className="mt-1 text-ink-soft">
                      <Link to="/knowledge-base" className="font-medium text-action underline underline-offset-2">Upload documents</Link>{" "}
                      to the knowledge base first, then come back to ask questions.
                    </p>
                  </div>
                ) : (
                  <div className="mt-8">
                    <p className="text-sm text-muted">Try one of these</p>
                    <div className="mt-2 flex flex-col items-start gap-2">
                      {STARTERS.map((s) => (
                        <button key={s} onClick={() => send(s)} className="rounded-md border border-rule bg-surface px-3.5 py-2 text-left text-sm hover:border-action hover:text-action">
                          {s}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            <div className="space-y-8">
              {messages.map((m, i) =>
                m.role === "user" ? (
                  <div key={m.id || i} className="flex justify-end">
                    <p className="max-w-[85%] rounded-2xl rounded-br-sm bg-ink px-4 py-2.5 text-[15px] text-white">{m.content}</p>
                  </div>
                ) : (
                  <article key={m.id || i} className={`border-l-2 pl-5 ${selected.msg === i ? "border-marker" : "border-rule"}`}>
                    {m.standalone_question && (
                      <p className="mb-2 text-xs text-muted">Searched for: “{m.standalone_question}”</p>
                    )}
                    <AnswerText text={m.content} msgIndex={i} activeKey={activeKey} onCite={(msg, source) => setSelected({ msg, source })} />
                    <AnswerMeta m={m} />
                    {selected.msg === i && (m.retrieved_chunks?.length ?? 0) > 0 && (
                      <div className="mt-4 h-[28rem] overflow-hidden rounded-lg border border-rule bg-surface xl:hidden">
                        <EvidencePanel message={m} activeSource={selected.source} onSelect={(source) => setSelected({ msg: i, source })} />
                      </div>
                    )}
                    {(m.retrieved_chunks?.length ?? 0) > 0 && selected.msg !== i && (
                      <button onClick={() => setSelected({ msg: i, source: m.citations?.[0]?.source_id ?? m.retrieved_chunks[0].source_id })} className="mt-2 text-xs font-medium text-action hover:underline">
                        Show evidence
                      </button>
                    )}
                  </article>
                ),
              )}
              {sending && (
                <div className="flex items-center gap-2 border-l-2 border-rule pl-5 text-sm text-muted" role="status">
                  <Loader2 size={16} className="animate-spin" aria-hidden="true" /> Searching the knowledge base and drafting an answer…
                </div>
              )}
            </div>
            <div ref={bottomRef} />
          </div>
        </div>

        {/* composer */}
        <div className="border-t border-rule bg-surface">
          <div className="mx-auto max-w-3xl px-5 py-4">
            {error && <div className="mb-3"><ErrorNotice error={error} /></div>}
            {showOptions && (
              <div className="mb-3 grid grid-cols-2 gap-3 rounded-md border border-rule bg-paper p-3 sm:max-w-md">
                <Field label="Passages to retrieve" hint="Blank uses the server default">
                  <input type="number" min="1" max="20" className={inputCls} value={options.top_k} onChange={(e) => setOptions({ ...options, top_k: e.target.value })} placeholder="Default" />
                </Field>
                <Field label="Search method">
                  <select className={inputCls} value={options.retrieval_mode} onChange={(e) => setOptions({ ...options, retrieval_mode: e.target.value })}>
                    <option value="">Server default</option>
                    <option value="vector">Semantic</option>
                    <option value="hybrid">Hybrid (semantic + keyword)</option>
                  </select>
                </Field>
              </div>
            )}
            <form
              onSubmit={(e) => {
                e.preventDefault();
                send();
              }}
              className="flex items-end gap-2 rounded-xl border border-rule-strong bg-surface p-2 focus-within:border-action"
            >
              <button type="button" onClick={() => setShowOptions((v) => !v)} className={`rounded-md p-2 ${showOptions ? "bg-action-tint text-action" : "text-muted hover:text-ink"}`} aria-label="Retrieval options" aria-expanded={showOptions}>
                <SlidersHorizontal size={18} />
              </button>
              <label htmlFor="question" className="sr-only">Your question</label>
              <textarea
                id="question"
                rows={1}
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    send();
                  }
                }}
                placeholder={activeId ? "Ask a follow-up question" : "Ask a question about your documents"}
                className="max-h-40 min-h-[38px] flex-1 resize-none bg-transparent px-1 py-2 text-[15px] focus:outline-none"
                maxLength={4000}
              />
              <button type="submit" disabled={!question.trim() || sending} className="flex h-9 items-center gap-1.5 rounded-lg bg-action px-3.5 text-sm font-medium text-white hover:bg-action-dark disabled:opacity-40">
                <SendHorizontal size={16} aria-hidden="true" /> Ask
              </button>
            </form>
            <p className="mt-1.5 text-xs text-muted">Enter to send, Shift + Enter for a new line.</p>
          </div>
        </div>
      </section>

      {/* evidence */}
      <aside className="hidden w-[380px] shrink-0 border-l border-rule bg-surface xl:block">
        <EvidencePanel message={selectedMessage?.role === "assistant" ? selectedMessage : null} activeSource={selected.source} onSelect={(source) => setSelected((s) => ({ ...s, source }))} />
      </aside>
    </div>
  );
}
