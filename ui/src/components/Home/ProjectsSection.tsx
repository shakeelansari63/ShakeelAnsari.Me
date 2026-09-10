import { Card, Row, Col, Button, Typography } from 'antd';
import { GithubOutlined, LinkOutlined } from '@ant-design/icons';
import type { GitProject } from '../../models/types';
import { userData } from '../../data/profile';

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
              hoverable
              style={{ cursor: 'pointer', borderRadius: 12, height: '100%' }}
              onClick={() => project.name && openProject(project.name)}
            >
              <div className="flex items-center gap-2 mb-2">
                <GithubOutlined style={{ color: 'var(--ant-color-primary)' }} />
                <Typography.Text strong style={{ color: 'var(--ant-color-primary)', fontSize: '1.0625rem' }}>
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
      <div className="mt-6 flex justify-center">
        <Button
          type="default"
          icon={<LinkOutlined />}
          onClick={() => window.open(`https://github.com/${userData.githubUser}?tab=repositories`, '_blank')}
          style={{
            background: 'linear-gradient(135deg, var(--ant-color-primary) 0%, #743ad5 100%)',
            border: 'none',
            borderRadius: 8,
            color: 'white',
          }}
        >
          See all Projects
        </Button>
      </div>
    </>
  );
}