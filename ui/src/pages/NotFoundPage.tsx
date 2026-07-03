import { Helmet } from 'react-helmet-async';
import { useNavigate } from 'react-router-dom';
import { Button } from 'primereact/button';
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
      <div className="app-container flex flex-column align-items-center justify-content-center" style={{ minHeight: '60vh' }}>
        <h1 className="text-pink-500 text-6xl font-bold m-0">404</h1>
        <p className="text-gray-400 text-xl mt-3 mb-4">Page not found</p>
        <Button
          label="Go Home"
          icon="pi pi-home"
          text
          severity="secondary"
          className="text-pink-500"
          onClick={() => navigate('/')}
          style={{ outline: 'none', boxShadow: 'none' }}
        />
      </div>
    </>
  );
}
