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
    <Row gutter={[16, 16]}>
      {items.map(item => (
        <Col key={item.route} xs={24} md={12}>
          <Card
            hoverable
            style={{
              cursor: 'pointer',
              borderRadius: 12,
              height: '100%',
              textAlign: 'center',
              transition: 'all 0.3s',
            }}
            onClick={() => navigate(item.route)}
          >
            <Space direction="vertical" size="middle" style={{ width: '100%' }}>
              <div style={{ color: settings.themeColor, fontSize: '2.5rem' }}>
                {item.icon}
              </div>
              <Typography.Title level={4} style={{ color: settings.themeColor, margin: 0 }}>
                {item.title}
              </Typography.Title>
              <Typography.Text type="secondary" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
                {item.description}
              </Typography.Text>
            </Space>
          </Card>
        </Col>
      ))}
    </Row>
  );
}