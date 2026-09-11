import { Card, Row, Col, Button, Typography } from 'antd';
import { GithubOutlined, LinkOutlined } from '@ant-design/icons';
import type { GitProject } from '../../models/types';
import { userData } from '../../data/profile';
import { settings } from '../../data/settings';

interface Props {
  projects: GitProject[];
}

export default function ProjectsSection({ projects }: Props) {
  const openProject = (name: string) => {
    window.open(`https://github.com/${userData.githubUser}/${name}`, '_blank');
  };

  return (
    <>
      <Row gutter={[16, 16]}>
        {projects.map(project => (
          <Col key={project.id} xs={24} md={12}>
            <Card
              className="glow-card"
              style={{ cursor: 'pointer', borderRadius: 12, height: '100%', transition: 'box-shadow 0.3s' }}
              onClick={() => project.name && openProject(project.name)}
            >
              <div className="flex items-center gap-2 mb-2" style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <GithubOutlined style={{ color: settings.themeColor }} />
                <Typography.Text strong style={{ color: settings.themeColor, fontSize: '1.0625rem' }}>
                  {project.name ?? ''}
                </Typography.Text>
              </div>
              {project.description && (
                <Typography.Text type="secondary" style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: '0.875rem' }}>
                  {project.description}
                </Typography.Text>
              )}
            </Card>
          </Col>
        ))}
      </Row>
      <div style={{ display: 'flex', justifyContent: 'center', width: '100%', marginTop: 32 }}>
        <Button
          type="primary"
          icon={<LinkOutlined />}
          shape="round"
          className="gradient-btn"
          onClick={() => window.open(`https://github.com/${userData.githubUser}?tab=repositories`, '_blank')}
          style={{
            background: `linear-gradient(135deg, ${settings.themeColor} 0%, #743ad5 100%)`,
            border: 'none',
            color: '#fff',
            fontWeight: 500,
          }}
        >
          See all Projects
        </Button>
      </div>
    </>
  );
}