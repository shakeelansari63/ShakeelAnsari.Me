import { Divider, Space, Typography } from 'antd';
import { GithubOutlined, LinkedinOutlined, TwitterOutlined, StarOutlined } from '@ant-design/icons';
import { seo } from '../../data/seo';
import { userData } from '../../data/profile';

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
    <footer className={`mt-8 pt-6 mb-8 ${className || ''}`} style={{ paddingBottom: 32, marginBottom: 32 }}>
      <Divider orientation="left" style={{ marginBottom: 16 }}>
        <Text type="secondary" strong>
          {'\u00A9'} {seo.domain}
        </Text>
      </Divider>
      <div style={{ display: 'flex', justifyContent: 'flex-end', paddingBottom: 16 }}>
      <Space className="flex-wrap" wrap>
        <a
          href={repoUrl}
          target="_blank"
          rel="noopener noreferrer"
          title="Star on GitHub"
          className="footer-link"
          style={linkStyle}
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
          className="footer-link"
          style={linkStyle}
        >
          <GithubOutlined />
        </a>
        &nbsp;
        <a
          href={userData.linkedIn}
          target="_blank"
          rel="noopener noreferrer"
          title="LinkedIn"
          className="footer-link"
          style={linkStyle}
        >
          <LinkedinOutlined />
        </a>
        &nbsp;
        <a
          href={userData.twitter}
          target="_blank"
          rel="noopener noreferrer"
          title="X (Twitter)"
          className="footer-link"
          style={linkStyle}
        >
          <TwitterOutlined />
        </a>
      </Space>
      </div>
    </footer>
  );
}
