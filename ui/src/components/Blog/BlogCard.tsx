import { useNavigate } from 'react-router-dom';
import { Card, Tag, Space } from 'antd';
import { EyeOutlined, HeartOutlined } from '@ant-design/icons';
import type { BlogPost } from '../../models/types';
import { settings } from '../../data/settings';

interface Props {
  post: BlogPost;
}

const tagStyle = {
  backgroundColor: `${settings.themeColor}1a`,
  borderColor: `${settings.themeColor}4d`,
  color: settings.themeColor,
  fontSize: '0.75rem',
};

export default function BlogCard({ post }: Props) {
  const navigate = useNavigate();

  return (
    <div className="col-span-12 md:col-span-6 lg:col-span-4">
      <Card
        hoverable
        cover={
          post.bannerImage ? (
            <img
              src={`/api/blogs/images/${post.bannerImage}`}
              alt={post.title}
              style={{
                width: '100%',
                aspectRatio: '16/9',
                objectFit: 'cover',
                borderRadius: '8px 8px 0 0',
              }}
              loading="lazy"
            />
          ) : null
        }
        actions={[
          <Space key="stats" size="small" style={{ color: 'var(--ant-color-text-tertiary)' }}>
            <span>
              <EyeOutlined style={{ marginRight: 4 }} />
              {post.views}
            </span>
            <span>
              <HeartOutlined style={{ marginRight: 4 }} />
              {post.likes}
            </span>
          </Space>,
        ]}
        onClick={() => navigate(`/blog/${post.id}`)}
        style={{
          cursor: 'pointer',
          borderRadius: 12,
          transition: 'all 0.3s',
          height: '100%',
        }}
      >
        <Card.Meta
          title={<h3 style={{ margin: 0, fontSize: '1.125rem', fontWeight: 600 }}>{post.title}</h3>}
          description={
            <div className="flex flex-col gap-2">
              <span className="text-sm" style={{ color: 'var(--ant-color-text-tertiary)' }}>
                {post.date} · {post.readTime}
              </span>
              <p
                className="m-0 line-clamp-2"
                style={{
                  color: 'var(--ant-color-text-secondary)',
                  fontFamily: "'Space Grotesk', sans-serif",
                  fontSize: '0.9375rem',
                }}
              >
                {post.excerpt}
              </p>
              <Space wrap size={[8, 4]}>
                {post.tags.map(tag => (
                  <Tag key={tag} style={tagStyle}>
                    {tag}
                  </Tag>
                ))}
              </Space>
            </div>
          }
        />
      </Card>
    </div>
  );
}