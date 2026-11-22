import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { bucketLabel } from "./format";

// Series colours: action blue, ink, marker gold, muted slate — validated for contrast on white.
export const SERIES = { blue: "#1D5FA8", ink: "#14213D", gold: "#C99A06", slate: "#8A94A8", red: "#B83A2A", green: "#2F8F5B" };

const axis = { stroke: "#6B7690", fontSize: 12, tickLine: false, axisLine: { stroke: "#DDE2EA" } };
const tooltipStyle = {
  contentStyle: { borderRadius: 8, border: "1px solid #DDE2EA", fontSize: 12, boxShadow: "0 4px 16px rgba(20,33,61,.08)" },
  labelStyle: { color: "#14213D", fontWeight: 600 },
};

export function NoData({ height = 220, children = "No data for this period yet." }) {
  return (
    <div className="flex items-center justify-center rounded-md bg-paper text-sm text-muted" style={{ height }}>
      {children}
    </div>
  );
}

export function TimeLineChart({ data, lines, height = 240, unit = "" }) {
  if (!data?.length) return <NoData height={height} />;
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
        <CartesianGrid stroke="#EEF1F5" vertical={false} />
        <XAxis dataKey="bucket" tickFormatter={bucketLabel} {...axis} minTickGap={24} />
        <YAxis {...axis} width={56} tickFormatter={(v) => `${Math.round(v)}${unit}`} />
        <Tooltip {...tooltipStyle} labelFormatter={bucketLabel} formatter={(v, n) => [v == null ? "—" : `${Number(v).toFixed(1)}${unit}`, n]} />
        <Legend iconType="plainline" wrapperStyle={{ fontSize: 12 }} />
        {lines.map((l) => (
          <Line key={l.key} type="monotone" dataKey={l.key} name={l.name} stroke={l.color} strokeWidth={2} dot={data.length < 3} connectNulls />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

export function SimpleBarChart({ data, bars, xKey = "bucket", height = 240, stacked = false, xFormatter = bucketLabel }) {
  if (!data?.length) return <NoData height={height} />;
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
        <CartesianGrid stroke="#EEF1F5" vertical={false} />
        <XAxis dataKey={xKey} tickFormatter={xFormatter} {...axis} minTickGap={16} />
        <YAxis {...axis} width={48} allowDecimals={false} />
        <Tooltip {...tooltipStyle} labelFormatter={xFormatter} cursor={{ fill: "#F2F4F8" }} />
        {bars.length > 1 && <Legend wrapperStyle={{ fontSize: 12 }} />}
        {bars.map((b) => (
          <Bar key={b.key} dataKey={b.key} name={b.name} fill={b.color} stackId={stacked ? "s" : undefined} radius={stacked ? 0 : [3, 3, 0, 0]} maxBarSize={36} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}
