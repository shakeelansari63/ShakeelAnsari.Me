import { useMemo } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts';
import { Card, Typography } from 'antd';
import useToken from 'antd/es/theme/useToken';

interface BlogDateView {
  date: string;
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

function ShortTitle({ title }: { title: string }) {
  const [, token] = useToken();
  return (
    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6, maxWidth: 160 }}>
      <Typography.Text ellipsis style={{ color: token.colorText, maxWidth: 150 }}>
        {title}
      </Typography.Text>
    </span>
  );
}

const legendValueRenderer = (value: string) => <ShortTitle title={value} />;

export default function DailyBlogViewsChart({ title, data }: Props) {
  const [, token] = useToken();
  const { chartData, blogKeys } = useMemo(() => {
    const keys: string[] = [];
    const idToTitle = new Map<string, string>();
    for (const row of data) {
      if (!idToTitle.has(row.blog_id)) {
        idToTitle.set(row.blog_id, row.title);
        keys.push(row.title);
      }
    }

    const byDate = new Map<string, Record<string, number>>();
    const dates: string[] = [];

    const last15: string[] = [];
    for (let i = 14; i >= 0; i--) {
      const d = new Date();
      d.setDate(d.getDate() - i);
      last15.push(d.toISOString().slice(0, 10));
    }

    for (const date of last15) {
      const entry: Record<string, number> = {};
      for (const key of keys) entry[key] = 0;
      byDate.set(date, entry);
      dates.push(date);
    }

    for (const row of data) {
      const entry = byDate.get(row.date);
      const t = idToTitle.get(row.blog_id) ?? row.title;
      if (entry && t in entry) entry[t] = row.count;
    }

    return { chartData: dates.map(d => ({ date: d, ...byDate.get(d) })), blogKeys: keys };
  }, [data]);

  return (
    <Card style={{ borderRadius: 12 }}>
      <Typography.Title level={4} style={{ color: token.colorText, marginBottom: 16 }}>
        {title}
      </Typography.Title>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke={token.colorBorder} />
          <XAxis dataKey="date" tick={{ fill: token.colorTextSecondary }} stroke={token.colorBorderSecondary} fontSize={12} />
          <YAxis allowDecimals={false} tick={{ fill: token.colorTextSecondary }} stroke={token.colorBorderSecondary} fontSize={12} />
          <Tooltip
            contentStyle={{
              background: token.colorBgContainer,
              border: `1px solid ${token.colorBorder}`,
              borderRadius: 8,
            }}
            labelStyle={{ color: token.colorText }}
            itemStyle={{ color: token.colorText }}
          />
          <Legend formatter={legendValueRenderer} />
          {blogKeys.map((key, index) => (
            <Bar
              key={key}
              dataKey={key}
              fill={COLORS[index % COLORS.length]}
              radius={[4, 4, 0, 0]}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </Card>
  );
}
