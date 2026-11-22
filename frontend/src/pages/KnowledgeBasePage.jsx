import { FileText, Library, RefreshCw, Trash2, UploadCloud, X } from "lucide-react";
import { useRef, useState } from "react";
import { fmtBytes, fmtDate, fmtMs, fmtNum } from "../components/format";
import { Button, EmptyState, ErrorNotice, Loading, PageHeader, Panel, StatusBadge } from "../components/ui";
import { useApi } from "../hooks/useApi";
import { api } from "../services/api";

const TYPE_STYLE = {
  pdf: "bg-[#FBE9E7] text-[#B23A26]",
  docx: "bg-[#E6EEFB] text-[#2457A6]",
  txt: "bg-[#ECEFF3] text-[#4A5568]",
  md: "bg-[#E8F3EE] text-[#2C7A55]",
};

function TypeTag({ type }) {
  return (
    <span className={`inline-flex w-11 justify-center rounded px-1.5 py-0.5 text-[11px] font-semibold ${TYPE_STYLE[type] || "bg-paper text-muted"}`}>
      {type.toUpperCase()}
    </span>
  );
}

function DropZone({ onFiles, uploading, config }) {
  const [over, setOver] = useState(false);
  const input = useRef(null);
  const allowed = config?.allowed_extensions?.join(", ") || ".pdf, .docx, .txt, .md";
  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        if (e.dataTransfer.files.length) onFiles([...e.dataTransfer.files]);
      }}
      className={`flex flex-col items-center justify-center rounded-lg border-2 border-dashed px-6 py-10 text-center transition-colors ${
        over ? "border-action bg-action-tint" : "border-rule-strong bg-surface"
      }`}
    >
      <UploadCloud size={30} className={over ? "text-action" : "text-muted"} aria-hidden="true" />
      <p className="mt-3 font-medium">{uploading ? "Uploading…" : "Drop files here to add them to the knowledge base"}</p>
      <p className="mt-1 text-sm text-muted">
        {allowed} up to {config?.max_upload_size_mb ?? 25} MB each. Identical files are detected and skipped.
      </p>
      <Button className="mt-4" variant="secondary" disabled={uploading} onClick={() => input.current?.click()}>
        Choose files
      </Button>
      <input
        ref={input}
        type="file"
        multiple
        accept={allowed.replace(/ /g, "")}
        className="hidden"
        onChange={(e) => {
          if (e.target.files.length) onFiles([...e.target.files]);
          e.target.value = "";
        }}
      />
    </div>
  );
}

function UploadResults({ results, onClose }) {
  if (!results) return null;
  const tone = { accepted: "text-ok", duplicate: "text-warn", rejected: "text-bad" };
  const label = { accepted: "Uploaded", duplicate: "Already in knowledge base", rejected: "Rejected" };
  return (
    <div className="rounded-lg border border-rule bg-surface">
      <div className="flex items-center justify-between border-b border-rule px-4 py-2.5">
        <p className="text-sm font-medium">
          Upload results: {results.accepted} uploaded, {results.duplicates} duplicate, {results.rejected} rejected
        </p>
        <button onClick={onClose} aria-label="Dismiss upload results" className="text-muted hover:text-ink"><X size={16} /></button>
      </div>
      <ul className="divide-y divide-rule text-sm">
        {results.results.map((r, i) => (
          <li key={i} className="flex flex-wrap items-baseline gap-x-3 px-4 py-2">
            <span className="font-medium">{r.filename}</span>
            <span className={tone[r.status]}>{label[r.status]}</span>
            <span className="text-muted">{r.message}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function DocumentDrawer({ id, onClose }) {
  const { data, error, loading } = useApi(() => api.getDocument(id), [id], { pollMs: 3000 });
  return (
    <div className="fixed inset-0 z-40 flex justify-end">
      <button className="absolute inset-0 bg-ink/30" aria-label="Close details" onClick={onClose} />
      <aside className="relative flex h-full w-full max-w-lg flex-col bg-surface shadow-xl" role="dialog" aria-label="Document details">
        <div className="flex items-start justify-between border-b border-rule px-6 py-4">
          <div className="min-w-0">
            <p className="truncate text-lg font-semibold">{data?.filename || "Document"}</p>
            {data && <StatusBadge status={data.status} />}
          </div>
          <button onClick={onClose} className="text-muted hover:text-ink" aria-label="Close"><X size={20} /></button>
        </div>
        <div className="min-h-0 flex-1 overflow-y-auto p-6 scroll-thin">
          {loading && <Loading />}
          <ErrorNotice error={error} />
          {data && (
            <>
              {data.error_message && (
                <div className="mb-5 rounded-md bg-bad-tint p-3 text-sm text-bad">
                  <p className="font-medium">Indexing failed</p>
                  <p>{data.error_message}</p>
                </div>
              )}
              <dl className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm">
                {[
                  ["Size", fmtBytes(data.file_size)],
                  ["Pages", data.num_pages ?? "Not paginated"],
                  ["Characters", fmtNum(data.num_characters)],
                  ["Chunks in vector store", fmtNum(data.chunks_in_vector_store)],
                  ["Processing time", fmtMs(data.processing_time_ms)],
                  ["Indexed", fmtDate(data.indexed_at)],
                  ["Embedding model", data.embedding_model || "—"],
                  ["Uploaded", fmtDate(data.created_at)],
                ].map(([k, v]) => (
                  <div key={k}>
                    <dt className="text-muted">{k}</dt>
                    <dd className="break-words font-medium">{v}</dd>
                  </div>
                ))}
              </dl>
              <p className="mt-5 break-all text-xs text-muted">SHA-256 {data.content_hash}</p>
              {data.sample_chunks.length > 0 && (
                <div className="mt-6">
                  <h3 className="font-semibold">First chunks</h3>
                  <ol className="mt-2 space-y-3">
                    {data.sample_chunks.map((c) => (
                      <li key={c.chunk_id} className="rounded-md border border-rule p-3">
                        <p className="text-xs text-muted">
                          Chunk {Number(c.chunk_id.split("-").pop())}{c.page_number ? `, page ${c.page_number}` : ""}, {c.chars} characters
                        </p>
                        <p className="mt-1 font-serif text-sm text-ink-soft">{c.text}{c.chars > 300 ? "…" : ""}</p>
                      </li>
                    ))}
                  </ol>
                </div>
              )}
            </>
          )}
        </div>
      </aside>
    </div>
  );
}

export default function KnowledgeBasePage() {
  const [uploading, setUploading] = useState(false);
  const [results, setResults] = useState(null);
  const [actionError, setActionError] = useState(null);
  const [busy, setBusy] = useState({});
  const [openId, setOpenId] = useState(null);
  const cfg = useApi(() => api.config(), []);
  const docs = useApi(() => api.listDocuments(), [], { pollMs: 2500 });

  async function upload(files) {
    setUploading(true);
    setActionError(null);
    try {
      setResults(await api.uploadDocuments(files));
      docs.refresh();
    } catch (e) {
      setActionError(e);
    } finally {
      setUploading(false);
    }
  }

  async function act(id, fn) {
    setBusy((b) => ({ ...b, [id]: true }));
    setActionError(null);
    try {
      await fn();
      docs.refresh();
    } catch (e) {
      setActionError(e);
    } finally {
      setBusy((b) => ({ ...b, [id]: false }));
    }
  }

  const items = docs.data?.items || [];
  const counts = docs.data?.counts;

  return (
    <div className="mx-auto max-w-6xl space-y-6 px-6 py-8">
      <PageHeader
        title="Knowledge base"
        description="Documents here are split into passages, embedded and stored in the vector database. Only indexed documents are searched."
        actions={counts && (
          <p className="text-sm text-ink-soft">
            <span className="font-semibold text-ink">{counts.indexed}</span> of {counts.total} indexed
            {counts.failed > 0 && <span className="text-bad">, {counts.failed} failed</span>}
          </p>
        )}
      />
      <DropZone onFiles={upload} uploading={uploading} config={cfg.data} />
      <UploadResults results={results} onClose={() => setResults(null)} />
      <ErrorNotice error={actionError || docs.error} onRetry={docs.error ? docs.refresh : undefined} />

      <Panel title="Documents" bodyClass="">
        {docs.loading && !docs.data ? (
          <Loading label="Loading documents" />
        ) : items.length === 0 ? (
          <EmptyState icon={Library} title="The knowledge base is empty">
            Upload a policy, handbook or report above. Indexing starts automatically.
          </EmptyState>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead className="border-b border-rule text-left text-muted">
                <tr>
                  <th className="px-5 py-2.5 font-medium">Document</th>
                  <th className="px-3 py-2.5 font-medium">Status</th>
                  <th className="px-3 py-2.5 text-right font-medium">Chunks</th>
                  <th className="px-3 py-2.5 text-right font-medium">Size</th>
                  <th className="px-3 py-2.5 font-medium">Uploaded</th>
                  <th className="px-5 py-2.5 text-right font-medium"><span className="sr-only">Actions</span></th>
                </tr>
              </thead>
              <tbody className="divide-y divide-rule">
                {items.map((d) => (
                  <tr key={d.id} className="hover:bg-paper/70">
                    <td className="max-w-xs px-5 py-3">
                      <button onClick={() => setOpenId(d.id)} className="flex min-w-0 items-center gap-3 text-left">
                        <TypeTag type={d.file_type} />
                        <span className="truncate font-medium hover:text-action hover:underline" title={d.filename}>{d.filename}</span>
                      </button>
                      {d.status === "failed" && d.error_message && <p className="mt-1 truncate pl-14 text-xs text-bad" title={d.error_message}>{d.error_message}</p>}
                    </td>
                    <td className="px-3 py-3"><StatusBadge status={d.status} /></td>
                    <td className="px-3 py-3 text-right tabular-nums">{d.status === "indexed" ? fmtNum(d.num_chunks) : "—"}</td>
                    <td className="px-3 py-3 text-right tabular-nums text-ink-soft">{fmtBytes(d.file_size)}</td>
                    <td className="whitespace-nowrap px-3 py-3 text-ink-soft">{fmtDate(d.created_at)}</td>
                    <td className="whitespace-nowrap px-5 py-3 text-right">
                      <Button
                        size="sm"
                        variant="ghost"
                        icon={RefreshCw}
                        disabled={busy[d.id] || d.status === "processing" || d.status === "pending"}
                        onClick={() => act(d.id, () => api.reindexDocument(d.id))}
                        aria-label={`Re-index ${d.filename}`}
                      >
                        Re-index
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        icon={Trash2}
                        className="hover:!bg-bad-tint hover:!text-bad"
                        disabled={busy[d.id]}
                        onClick={() => {
                          if (window.confirm(`Delete "${d.filename}" and its indexed passages? This can't be undone.`)) {
                            act(d.id, () => api.deleteDocument(d.id));
                          }
                        }}
                        aria-label={`Delete ${d.filename}`}
                      >
                        Delete
                      </Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
      {openId && <DocumentDrawer id={openId} onClose={() => setOpenId(null)} />}
      <p className="flex items-center gap-2 text-xs text-muted">
        <FileText size={14} aria-hidden="true" /> Scanned PDFs without a text layer can't be indexed in this version (no OCR).
      </p>
    </div>
  );
}
