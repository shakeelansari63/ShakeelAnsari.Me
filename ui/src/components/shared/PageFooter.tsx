import { Divider, Space, Typography } from 'antd';
import { GithubOutlined, LinkedinOutlined, TwitterOutlined, StarOutlined } from '@ant-design/icons';
import { seo } from '../../data/seo';
import { userData } from '../../data/profile';
import { settings } from '../../data/settings';

interface Props {
  className?: string;
}

export default function PageFooter({ className }: Props) {
  const repoUrl = 'https://github.com/shakeelansari63/ShakeelAnsari.Me';
  const { Text } = Typography;

  const linkStyle = {
    color: 'var(--ant-color-text-secondary)',
    transition: 'color 0.2s',
  };

  return (
    <footer className={`mt-8 pt-6 ${className || ''}`}>
      <Divider orientation="left" style={{ marginBottom: 16 }}>
        <Text type="secondary" strong>
          {'\u00A9'} {seo.domain}
        </Text>
      </Divider>
      <Space className="flex-wrap" wrap>
        <a
          href={repoUrl}
          target="_blank"
          rel="noopener noreferrer"
          title="Star on GitHub"
          style={linkStyle}
          onMouseEnter={e => (e.currentTarget.style.color = settings.themeColor)}
          onMouseLeave={e => (e.currentTarget.style.color = 'var(--ant-color-text-secondary)')}
        >
          <StarOutlined style={{ marginRight: 4 }} />
          Star
        </a>
        <Divider type="vertical" style={{ color: 'var(--ant-color-border)' }} />
        <a
          href={userData.github}
          target="_blank"
          rel="noopener noreferrer"
          title="GitHub"
          style={linkStyle}
          onMouseEnter={e => (e.currentTarget.style.color = settings.themeColor)}
          onMouseLeave={e => (e.currentTarget.style.color = 'var(--ant-color-text-secondary)')}
        >
          <GithubOutlined />
        </a>
        <a
          href={userData.linkedIn}
          target="_blank"
          rel="noopener noreferrer"
          title="LinkedIn"
          style={linkStyle}
          onMouseEnter={e => (e.currentTarget.style.color = settings.themeColor)}
          onMouseLeave={e => (e.currentTarget.style.color = 'var(--ant-color-text-secondary)')}
        >
          <LinkedinOutlined />
        </a>
        <a
          href={userData.twitter}
          target="_blank"
          rel="noopener noreferrer"
          title="X (Twitter)"
          style={linkStyle}
          onMouseEnter={e => (e.currentTarget.style.color = settings.themeColor)}
          onMouseLeave={e => (e.currentTarget.style.color = 'var(--ant-color-text-secondary)')}
        >
          <TwitterOutlined />
        </a>
      </Space>
    </footer>
  );
}