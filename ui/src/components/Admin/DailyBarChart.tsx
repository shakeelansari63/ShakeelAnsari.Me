import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';
import { Card, Typography } from 'antd';
import useToken from 'antd/es/theme/useToken';

interface Props {
  title: string;
  data: { date: string; count: number }[];
  color: string;
}

export default function DailyBarChart({ title, data, color }: Props) {
  const [, token] = useToken();
  return (
    <Card style={{ borderRadius: 12 }}>
      <Typography.Title level={4} style={{ color: token.colorText, marginBottom: 16 }}>
        {title}
      </Typography.Title>
      <ResponsiveContainer width="100%" height={250}>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke={token.colorBorder} />
          <XAxis dataKey="date" tick={{ fill: token.colorTextSecondary }} stroke={token.colorBorderSecondary} fontSize={12} />
          <YAxis tick={{ fill: token.colorTextSecondary }} stroke={token.colorBorderSecondary} fontSize={12} />
          <Tooltip
            contentStyle={{
              background: token.colorBgContainer,
              border: `1px solid ${token.colorBorder}`,
              borderRadius: 8,
            }}
            labelStyle={{ color: token.colorText }}
            itemStyle={{ color: token.colorText }}
          />
          <Bar dataKey="count" fill={color} radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </Card>
  );
}