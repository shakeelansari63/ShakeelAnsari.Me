import { Card, Space, Typography } from 'antd';
import { EyeOutlined, HeartOutlined } from '@ant-design/icons';

interface BlogEntry {
  id: string;
  title: string;
  views: number;
  likes: number;
}

interface Props {
  blogs: BlogEntry[];
}

export default function TopBlogsList({ blogs }: Props) {
  const { Text } = Typography;

  return (
    <Card style={{ borderRadius: 12 }}>
      <Typography.Title level={4} style={{ color: 'var(--ant-color-text)', marginBottom: 16 }}>
        Top Blogs by Views
      </Typography.Title>
      <div className="space-y-3">
        {blogs.map((blog, i) => (
          <div
            key={blog.id}
            className="flex items-center justify-between"
            style={{ padding: '8px 0', borderBottom: '1px solid var(--ant-color-border)' }}
          >
            <Space>
              <Text type="secondary" style={{ fontSize: '0.875rem' }}>
                {i + 1}.
              </Text>
              <Text strong style={{ color: 'var(--ant-color-primary)' }}>
                {blog.title || blog.id}
              </Text>
            </Space>
            <Space>
              <Space>
                <EyeOutlined />
                <Text type="secondary" style={{ fontSize: '0.875rem' }}>
                  {blog.views}
                </Text>
              </Space>
              <Space>
                <HeartOutlined />
                <Text type="secondary" style={{ fontSize: '0.875rem' }}>
                  {blog.likes}
                </Text>
              </Space>
            </Space>
          </div>
        ))}
      </div>
    </Card>
  );
}