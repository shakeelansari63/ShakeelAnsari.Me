import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from 'recharts';
import { Card, Typography } from 'antd';
import { settings } from '../../data/settings';

const COLORS = [settings.themeColor, '#743ad5', '#22c55e', '#f59e0b', '#3b82f6', '#ef4444', '#14b8a6', '#f97316', '#8b5cf6', '#ec4899'];

interface CountryEntry {
  code: string;
  country: string;
  count: number;
}

interface Props {
  title: string;
  data: CountryEntry[];
}

export default function CountryPieChart({ title, data }: Props) {
  return (
    <Card style={{ borderRadius: 12, height: '100%' }}>
      <Typography.Title level={4} style={{ color: 'var(--ant-color-text)', marginBottom: 16 }}>
        {title}
      </Typography.Title>
      <ResponsiveContainer width="100%" height={300}>
        <PieChart>
          <Pie
            data={data}
            dataKey="count"
            nameKey="country"
            cx="50%"
            cy="50%"
            outerRadius={100}
            label={(props: { payload?: { country: string; count: number } }) => props.payload ? `${props.payload.country}: ${props.payload.count}` : ''}
            labelLine={false}
          >
            {data.map((_, i) => (
              <Cell key={i} fill={COLORS[i % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip
            contentStyle={{
              background: 'var(--ant-color-bg-container)',
              border: '1px solid var(--ant-color-border)',
              borderRadius: 8,
            }}
            labelStyle={{ color: 'var(--ant-color-text)' }}
          />
        </PieChart>
      </ResponsiveContainer>
    </Card>
  );
}