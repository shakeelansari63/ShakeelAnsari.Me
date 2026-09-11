import { Helmet } from 'react-helmet-async';
import { useNavigate } from 'react-router-dom';
import { Button, Result } from 'antd';
import { HomeOutlined } from '@ant-design/icons';
import ToolBar from '../components/shared/ToolBar';
import { seo } from '../data/seo';

export default function NotFoundPage() {
  const navigate = useNavigate();

  return (
    <>
      <Helmet>
        <title>{`404 — ${seo.name}`}</title>
        <meta name="description" content="Page not found." />
        <meta name="robots" content="noindex, follow" />
      </Helmet>
      <ToolBar />
      <div className="app-container" style={{ minHeight: '60vh' }}>
        <Result
          status="404"
          title="404"
          subTitle="Page not found"
          extra={
            <Button type="primary" icon={<HomeOutlined />} onClick={() => navigate('/')}>
              Go Home
            </Button>
          }
          style={{ textAlign: 'center' }}
        />
      </div>
    </>
  );
}