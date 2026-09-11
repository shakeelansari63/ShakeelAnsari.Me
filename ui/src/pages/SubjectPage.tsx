import { useState, useEffect } from 'react';
import { Helmet } from 'react-helmet-async';
import { useParams, useNavigate } from 'react-router-dom';
import { Card, Button, Skeleton, Space, Typography } from 'antd';
import { LeftOutlined } from '@ant-design/icons';
import ToolBar from '../components/shared/ToolBar';
import PageFooter from '../components/shared/PageFooter';
import { fetchSubjectChapters } from '../services/api';
import type { LearnChapter } from '../models/types';
import { seo } from '../data/seo';
import { settings } from '../data/settings';
import { buildSubjectTitle } from '../services/helper';

export default function SubjectPage() {
  const { subjectId } = useParams<{ subjectId: string }>();
  const navigate = useNavigate();
  const [chapters, setChapters] = useState<LearnChapter[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (subjectId) {
      fetchSubjectChapters(subjectId)
        .then(setChapters)
        .finally(() => setLoading(false));
    }
  }, [subjectId]);

  const subjectTitle = subjectId ? buildSubjectTitle(subjectId) : '';

  return (
    <>
      <Helmet>
        <title>{subjectTitle ? `${subjectTitle} — Learn — ${seo.name}` : `Learn — ${seo.name}`}</title>
        <meta name="description" content={`${subjectTitle} — learning resources by ${seo.name}.`} />
        <meta property="og:title" content={subjectTitle ? `${subjectTitle} — Learn — ${seo.name}` : `Learn — ${seo.name}`} />
        <meta property="og:url" content={`https://${seo.domain}/learn/${subjectId}`} />
      </Helmet>
      <ToolBar />
      <div className="app-container">
        <div className="mb-6">
          <Button
            type="text"
            icon={<LeftOutlined />}
            onClick={() => navigate('/learn')}
            style={{ color: 'var(--ant-color-primary)' }}
          >
            Back to Learn
          </Button>
        </div>

        <h1 className="text-2xl font-bold mb-6" style={{ color: 'var(--ant-color-text)' }}>{subjectTitle}</h1>

        {loading ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {[0, 1, 2, 3, 4].map(i => (
              <Card key={i} style={{ borderRadius: 12 }}>
                <Skeleton active title={{ width: '45%' }} paragraph={false} />
              </Card>
            ))}
          </div>
        ) : chapters.length === 0 ? (
          <div className="text-center py-12">
            <div style={{ fontSize: '3rem', color: 'var(--ant-color-text-tertiary)', marginBottom: 16 }}>
              📖
            </div>
            <Typography.Title level={4} style={{ color: 'var(--ant-color-text-secondary)' }}>
              No chapters yet.
            </Typography.Title>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            {chapters.map(chapter => (
              <Card
                key={chapter.id}
                className="glow-card"
                style={{
                  cursor: 'pointer',
                  borderRadius: 12,
                  transition: 'box-shadow 0.3s',
                }}
                onClick={() => navigate(`/learn/${subjectId}/${chapter.chapter_id}`)}
              >
                <Space>
                  <Typography.Text type="secondary" style={{ fontSize: '0.875rem' }}>
                    {`Topic: ${parseInt(chapter.chapter_id.replace(/^ch(\d+).*$/, '$1'), 10)}`}
                  </Typography.Text>
                  <Typography.Text strong style={{ color: settings.themeColor }}>
                    {chapter.title}
                  </Typography.Text>
                </Space>
              </Card>
            ))}
          </div>
        )}
        <PageFooter />
      </div>
    </>
  );
}
