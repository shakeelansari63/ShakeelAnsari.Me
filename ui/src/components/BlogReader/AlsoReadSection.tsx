import { useNavigate } from 'react-router-dom';
import { Card, Typography } from 'antd';
import type { BlogPost } from '../../models/types';
import { settings } from '../../data/settings';

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
      <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
        {posts.map(post => (
          <Card
            key={post.id}
            className="glow-card"
            style={{
              borderRadius: 12,
              cursor: 'pointer',
              transition: 'box-shadow 0.3s',
            }}
            onClick={() => {
              window.scrollTo(0, 0);
              navigate(`/blog/${post.id}`);
            }}
          >
            <div className="flex flex-col gap-2">
              <Text strong style={{ display: 'block', width: '100%', color: settings.themeColor, fontSize: '1.0625rem' }}>
                {post.title}
              </Text>
              <Text type="secondary" style={{ display: 'block', width: '100%', fontFamily: "'Space Grotesk', sans-serif" }}>
                {post.excerpt}
              </Text>
              <Text type="secondary" style={{ display: 'block', width: '100%', fontSize: '0.75rem' }}>
                {post.date} · {post.readTime}
              </Text>
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}