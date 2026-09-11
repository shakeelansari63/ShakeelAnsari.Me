import { useNavigate } from 'react-router-dom';
import { Card, Button, Space } from 'antd';
import { GithubOutlined, LinkOutlined, AppstoreOutlined } from '@ant-design/icons';
import type { ExpoProject } from '../../data/expo';
import { settings } from '../../data/settings';

interface Props {
  item: ExpoProject;
}

export default function ExpoCard({ item }: Props) {
  const navigate = useNavigate();

  return (
    <Card
      className="expo-card"
      cover={item.thumbnail ? (
        <img
          src={item.thumbnail}
          alt={item.name}
          style={{
            width: '100%',
            borderRadius: '12px 12px 0 0',
            maxHeight: '200px',
            objectFit: 'cover',
          }}
        />
      ) : undefined}
      style={{
        borderRadius: 12,
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
      }}
      styles={{ body: { flex: 1, display: 'flex', flexDirection: 'column' } }}
      actions={[
        <div key="actions" style={{ display: 'flex', justifyContent: 'center', width: '100%' }}>
          <Space wrap size={[8, 8]} style={{ justifyContent: 'center' }}>
            {item.appUrl && (
              <Button
                type="primary"
                icon={<AppstoreOutlined />}
                shape="round"
                className="gradient-btn"
                onClick={() => window.open(item.appUrl, '_blank')}
                style={{
                  background: `linear-gradient(135deg, ${settings.themeColor} 0%, #743ad5 100%)`,
                  border: 'none',
                  color: '#fff',
                  fontWeight: 500,
                }}
              >
                App
              </Button>
            )}
            {item.productPageUrl && (
              <Button
                type="default"
                icon={<LinkOutlined />}
                shape="round"
                className="outline-theme-btn"
                onClick={() => navigate(item.productPageUrl!)}
                style={{
                  border: `1px solid ${settings.themeColor}`,
                  color: settings.themeColor,
                }}
              >
                Details
              </Button>
            )}
            {item.codeUrl && (
              <Button
                type="default"
                icon={<GithubOutlined />}
                shape="round"
                onClick={() => window.open(item.codeUrl, '_blank')}
              >
                Code
              </Button>
            )}
          </Space>
        </div>,
      ]}
    >
      <div className="flex flex-col flex-1 p-4">
        <Card.Meta
          title={<h3 style={{ margin: 0, fontSize: '1.125rem', fontWeight: 600, color: settings.themeColor }}>{item.name}</h3>}
          description={
            <p
              className="m-0 flex-1"
              style={{
                color: 'var(--ant-color-text-secondary)',
                fontFamily: "'Space Grotesk', sans-serif",
                lineHeight: 1.6,
              }}
            >
              {item.description}
            </p>
          }
        />
      </div>
    </Card>
  );
}