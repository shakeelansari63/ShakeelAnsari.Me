import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';
import { Card, Typography } from 'antd';

interface Props {
  title: string;
  data: { date: string; count: number }[];
  color: string;
}

export default function DailyBarChart({ title, data, color }: Props) {
  return (
    <Card style={{ borderRadius: 12 }}>
      <Typography.Title level={4} style={{ color: 'var(--ant-color-text)', marginBottom: 16 }}>
        {title}
      </Typography.Title>
      <ResponsiveContainer width="100%" height={250}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--ant-color-border)" />
          <XAxis dataKey="date" stroke="var(--ant-color-text-tertiary)" fontSize={12} />
          <YAxis stroke="var(--ant-color-text-tertiary)" fontSize={12} />
          <Tooltip
            contentStyle={{
              background: 'var(--ant-color-bg-container)',
              border: '1px solid var(--ant-color-border)',
              borderRadius: 8,
            }}
            labelStyle={{ color: 'var(--ant-color-text)' }}
          />
          <Bar dataKey="count" fill={color} radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </Card>
  );
}