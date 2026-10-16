import { useMemo } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from 'recharts';
import { Card, Typography } from 'antd';
import useToken from 'antd/es/theme/useToken';

interface BlogDateView {
  ddate: string;
  blog_id: string;
  title: string;
  count: number;
}

interface Props {
  title: string;
  data: BlogDateView[];
}

const COLORS = [
  '#6366f1', '#22c55e', '#f97316', '#06b6d4', '#a855f7', '#ef4444',
  '#eab308', '#14b8a6', '#ec4899', '#3b82f6', '#84cc16', '#f43f5e',
  '#8b5cf6', '#10b981', '#f59e0b', '#0ea5e9',
];

function ShortTitle({ title, color }: { title: string; color?: string }) {
  const [, token] = useToken();
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, maxWidth: 160 }}>
      <span
        style={{
          width: 10,
          height: 10,
          borderRadius: 2,
          background: color,
          flexShrink: 0,
        }}
      />
      <Typography.Text ellipsis style={{ color: token.colorText, maxWidth: 150 }}>
        {title}
      </Typography.Text>
    </span>
  );
}

export default function DailyBlogViewsChart({ title, data }: Props) {
  const [, token] = useToken();

  const { chartData, legendPayload } = useMemo(() => {
    // Assign one color per blog (in first-seen order), only for blogs viewed
    // in the window.
    const colors = new Map<string, string>();
    const titles = new Map<string, string>();
    for (const row of data) {
      if (!colors.has(row.blog_id)) {
        colors.set(row.blog_id, COLORS[colors.size % COLORS.length]);
      }
      if (!titles.has(row.blog_id)) {
        titles.set(row.blog_id, row.title);
      }
    }

    // Build the day list for the last month (calendar month back from today)
    const today = new Date();
    const monthStart = new Date(today);
    monthStart.setMonth(monthStart.getMonth() - 1);
    monthStart.setHours(12, 0, 0, 0);

    const windowDates: string[] = [];
    const dayCount = Math.round(
      (today.getTime() - monthStart.getTime()) / 86400000,
    );
    for (let i = 0; i <= dayCount; i++) {
      const d = new Date(monthStart);
      d.setDate(d.getDate() + i);
      windowDates.push(d.toISOString().slice(0, 10));
    }

    // One chart row per (date, blog) pair that actually has views — a date
    // with only 2 viewed blogs renders exactly 2 bars, no empty slots.
    // Empty dates still get one zero-height placeholder row so their slot
    // and axis label stay visible.
    // Coalesce rows by (date, blog_id): the API should already be grouped,
    // but if duplicates ever appear sum them so a bar never shows a wrong
    // (dropped or compounded) value. Unexpected duplicates are logged once.
    const seen = new Map<string, number>();
    const rows: {
      key: string;
      group: string;
      count: number;
      blogTitle: string;
      color: string;
      isEmpty?: boolean;
    }[] = [];

    for (const date of windowDates) {
      const dayRows = data
        .filter(r => r.ddate === date)
        .sort((a, b) => b.count - a.count);
      if (dayRows.length === 0) {
        rows.push({
          key: `${date}|__empty`,
          group: date,
          count: 0,
          blogTitle: '',
          color: 'transparent',
          isEmpty: true,
        });
        continue;
      }
      for (const row of dayRows) {
        const key = `${date}|${row.blog_id}`;
        if (seen.has(key)) {
          const zeroRow = rows.find(r => r.key === key);
          if (zeroRow) zeroRow.count += row.count;
          if (import.meta.env.DEV) {
            console.warn(
              `[DailyBlogViewsChart] duplicate view rows for blog ${row.blog_id} on ${date}; coalesced by summing.`,
            );
          }
          continue;
        }
        seen.set(key, row.count);
        rows.push({
          key,
          group: date,
          count: row.count,
          blogTitle: titles.get(row.blog_id) ?? row.title,
          color: colors.get(row.blog_id) ?? COLORS[0],
        });
      }
    }

    const legend = [...colors.entries()].map(([id, color]) => ({
      id,
      title: titles.get(id) ?? id,
      color,
    }));

    return {
      chartData: rows,
      legendPayload: legend,
    };
  }, [data]);

  return (
    <Card style={{ borderRadius: 12 }}>
      <Typography.Title level={4} style={{ color: token.colorText, marginBottom: 16 }}>
        {title}
      </Typography.Title>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart
          data={chartData}
          barCategoryGap="25%"
          barGap={2}
        >
          <CartesianGrid strokeDasharray="3 3" stroke={token.colorBorder} />
          <XAxis
            dataKey="key"
            tickFormatter={(value: string) => {
              // Show the date label only once for each date group
              const row = chartData.find(r => r.key === value);
              if (!row) return '';
              const isFirst = chartData.find(r => r.group === row.group) === row;
              return isFirst ? row.group : '';
            }}
            interval={0}
            angle={-35}
            textAnchor="end"
            height={55}
            tick={{ fill: token.colorTextSecondary, fontSize: 12 }}
            tickMargin={6}
            stroke={token.colorBorderSecondary}
            fontSize={12}
          />
          <YAxis
            allowDecimals={false}
            tick={{ fill: token.colorTextSecondary }}
            stroke={token.colorBorderSecondary}
            fontSize={12}
          />
          <Tooltip
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const row = payload[0].payload as (typeof chartData)[number];
              return (
                <div
                  style={{
                    background: token.colorBgContainer,
                    border: `1px solid ${token.colorBorder}`,
                    borderRadius: 8,
                    padding: '8px 12px',
                    fontSize: 13,
                  }}
                >
                  <div style={{ color: token.colorTextSecondary, marginBottom: 4 }}>
                    {row.group}
                  </div>
                  <div style={{ color: token.colorText }}>
                    <span
                      style={{
                        display: 'inline-block',
                        width: 10,
                        height: 10,
                        borderRadius: 2,
                        background: row.color,
                        marginRight: 6,
                      }}
                    />
                    {row.blogTitle}
                  </div>
                  <div style={{ color: token.colorTextSecondary, marginTop: 2 }}>
                    Views: <strong style={{ color: token.colorText }}>{row.count}</strong>
                  </div>
                </div>
              );
            }}
            cursor={{ fill: token.colorFillSecondary, opacity: 0.3 }}
          />
          <Bar dataKey="count" radius={[4, 4, 0, 0]} isAnimationActive={false}>
            {chartData.map(row => (
              <Cell key={row.key} fill={row.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      {legendPayload.length > 0 && (
        <div
          style={{
            display: 'flex',
            flexWrap: 'wrap',
            gap: '8px 18px',
            justifyContent: 'center',
            marginTop: 8,
          }}
        >
          {legendPayload.map(item => (
            <ShortTitle key={item.id} title={item.title} color={item.color} />
          ))}
        </div>
      )}
    </Card>
  );
}
