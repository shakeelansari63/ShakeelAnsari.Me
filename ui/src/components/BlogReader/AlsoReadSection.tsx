import { useNavigate } from 'react-router-dom';
import { Card, Typography } from 'antd';
import type { BlogPost } from '../../models/types';

interface Props {
  posts: BlogPost[];
}

export default function AlsoReadSection({ posts }: Props) {
  const navigate = useNavigate();

  if (posts.length === 0) return null;

  const { Text } = Typography;

  return (
    <div className="mt-10">
      <h2 className="text-xl font-bold mb-6" style={{ color: 'var(--ant-color-primary)' }}>
        Also Read
      </h2>
      <div className="space-y-4">
        {posts.map(post => (
          <Card
            key={post.id}
            hoverable
            style={{
              borderRadius: 10,
              border: '1px solid var(--ant-color-border)',
              transition: 'all 0.2s',
            }}
            onClick={() => {
              window.scrollTo(0, 0);
              navigate(`/blog/${post.id}`);
            }}
          >
            <div className="flex flex-col gap-2">
              <Text strong style={{ color: 'var(--ant-color-primary)', fontSize: '1.0625rem' }}>
                {post.title}
              </Text>
              <Text type="secondary" style={{ fontFamily: "'Space Grotesk', sans-serif" }}>
                {post.excerpt}
              </Text>
              <Text type="secondary" style={{ fontSize: '0.75rem' }}>
                {post.date} · {post.readTime}
              </Text>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}