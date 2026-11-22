export const fmtMs = (v) => {
  if (v === null || v === undefined) return "—";
  if (v >= 1000) return `${(v / 1000).toFixed(2)} s`;
  if (v >= 100) return `${Math.round(v)} ms`;
  return `${v.toFixed(1)} ms`;
};

export const fmtNum = (v) => (v === null || v === undefined ? "—" : Number(v).toLocaleString("en-IN"));

export const fmtPct = (v, digits = 0) =>
  v === null || v === undefined ? "—" : `${(v * 100).toFixed(digits)}%`;

export const fmtScore = (v, digits = 2) => (v === null || v === undefined ? "—" : Number(v).toFixed(digits));

export const fmtBytes = (b) => {
  if (b == null) return "—";
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / 1024 / 1024).toFixed(1)} MB`;
};

export const fmtDate = (iso) => {
  if (!iso) return "—";
  const d = new Date(iso.endsWith("Z") || iso.includes("+") ? iso : `${iso}Z`);
  return d.toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
};

export const fmtRelative = (iso) => {
  if (!iso) return "—";
  const d = new Date(iso.endsWith("Z") || iso.includes("+") ? iso : `${iso}Z`);
  const s = Math.round((Date.now() - d.getTime()) / 1000);
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return fmtDate(iso);
};

export const bucketLabel = (b) => {
  if (!b) return "";
  if (b.includes("T")) return b.slice(5).replace("T", " ");
  return b.slice(5);
};
