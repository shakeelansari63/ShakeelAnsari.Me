import { Card, Row, Col, Typography } from 'antd';
import { settings } from '../../data/settings';

interface Props {
  totalViews: number;
  totalLikes: number;
  uniqueVisitors: number;
}

export default function SummaryCards({ totalViews, totalLikes, uniqueVisitors }: Props) {
  const items = [
    { value: totalViews, label: 'Total Views', color: settings.themeColor },
    { value: totalLikes, label: 'Total Likes', color: '#22c55e' },
    { value: uniqueVisitors, label: 'Unique Visitors', color: '#1890ff' },
  ];

  return (
    <Row gutter={[16, 16]}>
      {items.map(item => (
        <Col key={item.label} xs={24} md={8}>
          <Card style={{ textAlign: 'center', borderRadius: 12, height: '100%' }}>
            <Typography.Text strong style={{ fontSize: '2.5rem', color: item.color, display: 'block', marginBottom: 8 }}>
              {item.value}
            </Typography.Text>
            <Typography.Text type="secondary" style={{ margin: 0 }}>
              {item.label}
            </Typography.Text>
          </Card>
        </Col>
      ))}
    </Row>
  );
}