import { useNavigate } from 'react-router-dom';
import { Card, Row, Col, Space, Typography } from 'antd';
import {
  FileTextOutlined,
  StarOutlined,
  BookOutlined,
  BarChartOutlined,
} from '@ant-design/icons';
import { settings } from '../../data/settings';

export default function AlsoSeeSection() {
  const navigate = useNavigate();

  const items = [
    {
      icon: <FileTextOutlined />,
      title: 'Blogs',
      description: 'Thoughts on data engineering, AI, and software development.',
      route: '/blog',
    },
    {
      icon: <StarOutlined />,
      title: 'Expo',
      description: 'Showcase of projects and experiments I have built.',
      route: '/expo',
    },
    {
      icon: <BookOutlined />,
      title: 'Learn',
      description: 'Tutorials on programming languages and tech topics.',
      route: '/learn',
    },
    {
      icon: <BarChartOutlined />,
      title: 'Stats',
      description: 'GitHub statistics, languages, streaks and projects.',
      route: '/stats',
    },
  ];

  return (
    <Row gutter={[16, 16]} justify="center">
      {items.map(item => (
        <Col key={item.route} xs={24} sm={12} md={12} lg={6}>
          <Card
            className="glow-card"
            style={{
              cursor: 'pointer',
              borderRadius: 12,
              height: '100%',
              textAlign: 'center',
              transition: 'box-shadow 0.3s',
            }}
            styles={{ body: { padding: 16 } }}
            onClick={() => navigate(item.route)}
          >
            <Space direction="vertical" size="small" style={{ width: '100%' }}>
              <div style={{ color: settings.themeColor, fontSize: '1.75rem' }}>
                {item.icon}
              </div>
              <Typography.Title level={5} style={{ color: settings.themeColor, margin: 0 }}>
                {item.title}
              </Typography.Title>
              <Typography.Text type="secondary" style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: '0.8125rem' }}>
                {item.description}
              </Typography.Text>
            </Space>
          </Card>
        </Col>
      ))}
    </Row>
  );
}