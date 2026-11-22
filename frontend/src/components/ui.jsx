import { AlertCircle, Loader2 } from "lucide-react";

export function Button({ variant = "primary", size = "md", className = "", icon: Icon, children, ...props }) {
  const base =
    "inline-flex items-center justify-center gap-2 rounded-md font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed";
  const sizes = { sm: "h-8 px-3 text-sm", md: "h-9 px-4 text-sm" };
  const variants = {
    primary: "bg-action text-white hover:bg-action-dark",
    secondary: "bg-surface text-ink border border-rule-strong hover:bg-paper",
    ghost: "text-ink-soft hover:bg-action-tint hover:text-action",
    danger: "bg-surface text-bad border border-rule-strong hover:bg-bad-tint hover:border-bad",
  };
  return (
    <button className={`${base} ${sizes[size]} ${variants[variant]} ${className}`} {...props}>
      {Icon && <Icon size={16} aria-hidden="true" />}
      {children}
    </button>
  );
}

const STATUS = {
  indexed: ["bg-ok-tint text-ok", "Indexed"],
  processing: ["bg-action-tint text-action", "Processing"],
  pending: ["bg-warn-tint text-warn", "Queued"],
  failed: ["bg-bad-tint text-bad", "Failed"],
  success: ["bg-ok-tint text-ok", "Answered"],
  no_context: ["bg-warn-tint text-warn", "No evidence"],
  error: ["bg-bad-tint text-bad", "Error"],
  completed: ["bg-ok-tint text-ok", "Completed"],
  running: ["bg-action-tint text-action", "Running"],
  ok: ["bg-ok-tint text-ok", "Healthy"],
  degraded: ["bg-bad-tint text-bad", "Degraded"],
};

export function StatusBadge({ status }) {
  const [cls, label] = STATUS[status] || ["bg-paper text-muted", status];
  const live = status === "processing" || status === "pending" || status === "running";
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium ${cls}`}>
      {live ? <Loader2 size={12} className="animate-spin" aria-hidden="true" /> : <span className="h-1.5 w-1.5 rounded-full bg-current" />}
      {label}
    </span>
  );
}

export function PageHeader({ title, description, actions }) {
  return (
    <header className="flex flex-wrap items-end justify-between gap-4 border-b border-rule pb-5">
      <div className="max-w-2xl">
        <h1 className="text-[26px] font-semibold leading-tight tracking-tight">{title}</h1>
        {description && <p className="mt-1.5 text-ink-soft">{description}</p>}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </header>
  );
}

export function Panel({ title, description, actions, children, className = "", bodyClass = "p-5" }) {
  return (
    <section className={`rounded-lg border border-rule bg-surface ${className}`}>
      {(title || actions) && (
        <div className="flex items-start justify-between gap-3 border-b border-rule px-5 py-3.5">
          <div>
            <h2 className="text-[15px] font-semibold">{title}</h2>
            {description && <p className="text-sm text-muted">{description}</p>}
          </div>
          {actions}
        </div>
      )}
      <div className={bodyClass}>{children}</div>
    </section>
  );
}

/** A single strip of figures separated by rules — reads as one instrument, not a card grid. */
export function StatStrip({ items }) {
  return (
    <dl className="grid grid-cols-2 overflow-hidden rounded-lg border border-rule bg-surface sm:grid-cols-3 lg:grid-cols-6">
      {items.map((it, i) => (
        <div
          key={it.label}
          className={`px-5 py-4 ${i > 0 ? "border-t border-rule sm:border-t-0" : ""} ${i % 6 !== 0 ? "lg:border-l lg:border-rule" : ""} ${i % 3 !== 0 ? "sm:border-l sm:border-rule" : ""}`}
        >
          <dt className="text-sm text-muted">{it.label}</dt>
          <dd className="mt-1 text-2xl font-semibold tabular-nums tracking-tight">{it.value}</dd>
          {it.hint && <dd className="mt-0.5 text-xs text-muted">{it.hint}</dd>}
        </div>
      ))}
    </dl>
  );
}

export function ErrorNotice({ error, onRetry }) {
  if (!error) return null;
  return (
    <div role="alert" className="flex items-start gap-3 rounded-md border border-bad/30 bg-bad-tint px-4 py-3 text-sm text-bad">
      <AlertCircle size={18} className="mt-0.5 shrink-0" aria-hidden="true" />
      <div className="flex-1">
        <p className="font-medium">{error.message}</p>
        {error.code && <p className="text-xs opacity-80">Error code: {error.code}</p>}
      </div>
      {onRetry && (
        <button onClick={onRetry} className="font-medium underline underline-offset-2">
          Try again
        </button>
      )}
    </div>
  );
}

export function Loading({ label = "Loading" }) {
  return (
    <div className="flex items-center gap-2 py-10 text-sm text-muted" role="status">
      <Loader2 size={16} className="animate-spin" aria-hidden="true" /> {label}…
    </div>
  );
}

export function EmptyState({ icon: Icon, title, children, action }) {
  return (
    <div className="flex flex-col items-center px-6 py-12 text-center">
      {Icon && <Icon size={28} className="text-rule-strong" aria-hidden="true" />}
      <p className="mt-3 font-medium">{title}</p>
      {children && <p className="mt-1 max-w-sm text-sm text-muted">{children}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

export function Field({ label, children, hint }) {
  return (
    <label className="block text-sm">
      <span className="font-medium text-ink-soft">{label}</span>
      <div className="mt-1">{children}</div>
      {hint && <span className="mt-1 block text-xs text-muted">{hint}</span>}
    </label>
  );
}

export const inputCls =
  "h-9 w-full rounded-md border border-rule-strong bg-surface px-3 text-sm text-ink focus:border-action focus:outline-none";

export function ScoreBar({ value, max = 1 }) {
  const pct = value == null ? 0 : Math.max(0, Math.min(1, value / max)) * 100;
  return (
    <span className="inline-block h-1.5 w-16 overflow-hidden rounded-full bg-rule align-middle" aria-hidden="true">
      <span className="block h-full rounded-full bg-action" style={{ width: `${pct}%` }} />
    </span>
  );
}
