import { useNavigate } from 'react-router-dom';
import { Card, Button, Space } from 'antd';
import { GithubOutlined, LinkOutlined, AppstoreOutlined } from '@ant-design/icons';
import type { ExpoProject } from '../../data/expo';

interface Props {
  item: ExpoProject;
}

export default function ExpoCard({ item }: Props) {
  const navigate = useNavigate();

  return (
    <Card
      hoverable
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
    >
      <div className="flex flex-col flex-1 p-4">
        <Card.Meta
          title={<h3 style={{ margin: 0, fontSize: '1.125rem', fontWeight: 600, color: 'var(--ant-color-primary)' }}>{item.name}</h3>}
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
        <div className="mt-4 pt-4 border-t" style={{ borderColor: 'var(--ant-color-border)' }}>
          <Space wrap size={[8, 8]}>
            {item.appUrl && (
              <Button
                type="default"
                icon={<AppstoreOutlined />}
                onClick={() => window.open(item.appUrl, '_blank')}
                style={{
                  background: 'linear-gradient(135deg, var(--ant-color-primary) 0%, #743ad5 100%)',
                  border: 'none',
                  borderRadius: 8,
                  color: 'white',
                }}
              >
                App
              </Button>
            )}
            {item.productPageUrl && (
              <Button
                type="default"
                icon={<LinkOutlined />}
                onClick={() => navigate(item.productPageUrl!)}
                style={{ borderRadius: 8 }}
              >
                Details
              </Button>
            )}
            {item.codeUrl && (
              <Button
                type="default"
                icon={<GithubOutlined />}
                onClick={() => window.open(item.codeUrl, '_blank')}
                style={{ borderRadius: 8 }}
              >
                Code
              </Button>
            )}
          </Space>
        </div>
      </div>
    </Card>
  );
}