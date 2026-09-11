import { useState, useEffect } from 'react';
import { Helmet } from 'react-helmet-async';
import { useNavigate } from 'react-router-dom';
import { Card, Skeleton, Row, Col, Typography } from 'antd';
import ToolBar from '../components/shared/ToolBar';
import PageFooter from '../components/shared/PageFooter';
import { fetchLearnSubjects } from '../services/api';
import type { LearnSubject } from '../models/types';
import { seo } from '../data/seo';

export default function LearnPage() {
  const navigate = useNavigate();
  const [subjects, setSubjects] = useState<LearnSubject[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchLearnSubjects()
      .then(setSubjects)
      .finally(() => setLoading(false));
  }, []);

  const { Title } = Typography;

  return (
    <>
      <Helmet>
        <title>{`Learn — ${seo.name}`}</title>
        <meta name="description" content={`Tutorials and learning resources by ${seo.name}.`} />
        <meta property="og:title" content={`Learn — ${seo.name}`} />
        <meta property="og:description" content={`Tutorials and learning resources by ${seo.name}.`} />
        <meta property="og:url" content={`https://${seo.domain}/learn`} />
      </Helmet>
      <ToolBar />
      <div className="app-container pb-4">
        <Title level={2} className="mb-6" style={{ color: 'var(--ant-color-text)' }}>
          Learn
        </Title>

        {loading ? (
          <Row gutter={[16, 16]}>
            <Col xs={24} md={12}>
              <Skeleton active avatar paragraph={{ rows: 3 }} />
            </Col>
            <Col xs={24} md={12}>
              <Skeleton active avatar paragraph={{ rows: 3 }} />
            </Col>
            <Col xs={24} md={12}>
              <Skeleton active avatar paragraph={{ rows: 3 }} />
            </Col>
          </Row>
        ) : subjects.length === 0 ? (
          <div className="text-center py-12">
            <div style={{ fontSize: '3rem', color: 'var(--ant-color-text-tertiary)', marginBottom: 16 }}>
              📚
            </div>
            <Title level={4} style={{ color: 'var(--ant-color-text-secondary)' }}>
              No subjects available yet.
            </Title>
          </div>
        ) : (
          <Row gutter={[16, 16]}>
            {subjects.map(subject => (
              <Col key={subject.id} xs={24} md={12} lg={8}>
                <Card
                  className="glow-card"
                  cover={
                    <img
                      src={`/api/learn/images/${subject.folder}/${subject.thumbnail}`}
                      alt={subject.title}
                      style={{
                        width: '100%',
                        aspectRatio: '16/9',
                        objectFit: 'contain',
                        background: 'var(--ant-color-bg-container)',
                      }}
                      loading="lazy"
                    />
                  }
                  style={{
                    borderRadius: 12,
                    height: '100%',
                    cursor: 'pointer',
                    transition: 'box-shadow 0.3s',
                  }}
                  onClick={() => navigate(`/learn/${subject.id}`)}
                >
                  <Card.Meta
                    title={<Title level={4} style={{ color: 'var(--ant-color-primary)' }}>{subject.title}</Title>}
                  />
                </Card>
              </Col>
            ))}
          </Row>
        )}
        <PageFooter />
      </div>
    </>
  );
}