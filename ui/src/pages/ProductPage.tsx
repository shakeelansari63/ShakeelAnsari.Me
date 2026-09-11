import { useState, useEffect } from 'react';
import { Helmet } from 'react-helmet-async';
import { useParams, useNavigate } from 'react-router-dom';
import { Button, Skeleton } from 'antd';
import { LeftOutlined } from '@ant-design/icons';
import ToolBar from '../components/shared/ToolBar';
import MarkdownRenderer from '../components/shared/MarkdownRenderer';
import PageFooter from '../components/shared/PageFooter';
import { fetchProductContent } from '../services/api';
import { seo } from '../data/seo';

export default function ProductPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [loading, setLoading] = useState(true);
  const [notFound, setNotFound] = useState(false);

  useEffect(() => {
    if (id) {
      fetchProductContent(id).then(data => {
        if (data) {
          setTitle(data.title);
          setContent(data.content);
        } else {
          setNotFound(true);
        }
        setLoading(false);
      });
    }
  }, [id]);

  const metaTitle = title ? `${title} — ${seo.name}` : `Product — ${seo.name}`;

  return (
    <>
      <Helmet>
        <title>{metaTitle}</title>
        <meta name="description" content={title ? `${title} — ${seo.name}` : seo.description} />
        <meta property="og:title" content={metaTitle} />
        <meta property="og:url" content={`https://${seo.domain}/product/${id}`} />
      </Helmet>
      <ToolBar />
      <div className="app-container">
        <div className="mb-6">
          <Button
            type="text"
            icon={<LeftOutlined />}
            onClick={() => navigate('/expo')}
            style={{ color: 'var(--ant-color-primary)' }}
          >
            Back to Expo
          </Button>
        </div>

        {loading ? (
          <div className="space-y-4">
            <Skeleton active avatar paragraph={{ rows: 3 }} />
          </div>
        ) : notFound ? (
          <div className="text-center py-12">
            <h2 style={{ color: 'var(--ant-color-text)' }}>Product not found</h2>
            <Button
              type="text"
              onClick={() => navigate('/expo')}
              style={{ marginTop: 16, color: 'var(--ant-color-primary)' }}
            >
              Back to Expo
            </Button>
          </div>
        ) : (
          <article>
            <h1 className="text-2xl md:text-3xl font-bold mb-6" style={{ color: 'var(--ant-color-primary)' }}>
              {title}
            </h1>
            <MarkdownRenderer content={content} />
            <PageFooter />
          </article>
        )}
      </div>
    </>
  );
}