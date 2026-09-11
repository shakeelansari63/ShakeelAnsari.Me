import { Tag, Button, Space } from 'antd';
import { HeartOutlined, HeartFilled, ShareAltOutlined, EyeOutlined } from '@ant-design/icons';
import type { BlogPost } from '../../models/types';
import { settings } from '../../data/settings';

interface Props {
  post: BlogPost;
  stats: { views: number; likes: number; liked?: boolean };
  liking?: boolean;
  onLike?: () => void;
}

const tagStyle = {
  backgroundColor: `${settings.themeColor}1a`,
  borderColor: `${settings.themeColor}4d`,
  color: settings.themeColor,
  fontSize: '0.75rem',
};

export default function ArticleHeader({ post, stats, liking, onLike }: Props) {
  return (
    <>
      <h1 className="text-2xl md:text-3xl font-bold mb-2" style={{ color: 'var(--ant-color-text)' }}>
        {post.title}
      </h1>
      <div className="article-meta" style={{ color: 'var(--ant-color-text-secondary)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span>{post.date}</span>
          <span>·</span>
          <span>{post.readTime}</span>
        </div>
        <div className="article-meta-stats">
          <span style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <EyeOutlined />
            {stats.views}
          </span>
          <Button
            type="text"
            icon={stats.liked ? <HeartFilled style={{ color: '#ff4d4f' }} /> : <HeartOutlined />}
            loading={liking}
            onClick={onLike}
            danger={stats.liked}
            style={{ padding: '0 8px', fontSize: '0.875rem', color: settings.themeColor }}
          >
            {stats.likes}
          </Button>
          <Button
            type="text"
            icon={<ShareAltOutlined />}
            onClick={() => {
              const url = window.location.href;
              if (navigator.share) {
                navigator.share({ title: post.title, url }).catch(() => {
                  navigator.clipboard.writeText(url);
                });
              } else {
                navigator.clipboard.writeText(url);
              }
            }}
            style={{ padding: '0 8px', color: settings.themeColor }}
          >
            Share
          </Button>
        </div>
      </div>
      <Space wrap size={[8, 4]} className="mb-6" style={{ marginBottom: 32, display: 'flex' }}>
        {post.tags.map(tag => (
          <Tag key={tag} style={tagStyle}>
            {tag}
          </Tag>
        ))}
      </Space>
    </>
  );
}