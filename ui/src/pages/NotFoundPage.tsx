import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button } from 'primereact/button';
import ToolBar from '../components/shared/ToolBar';

export default function NotFoundPage() {
  useEffect(() => { document.title = "404 — [{#SEO-NAME#}]"; }, []);
  const navigate = useNavigate();

  return (
    <>
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
