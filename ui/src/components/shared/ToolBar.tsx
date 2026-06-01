import { useNavigate, useLocation } from 'react-router-dom';
import { Toolbar } from 'primereact/toolbar';
import { Button } from 'primereact/button';
import { userData } from '../../services/data';

export default function ToolBar() {
  const navigate = useNavigate();
  const location = useLocation();

  const scrollTo = (id: string) => {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const startContent = (
    <span className="text-pink-500 font-bold text-2xl no-underline cursor-pointer" onClick={() => navigate('/')}>
      @{userData.devUsername}
    </span>
  );

  const endContent = (
    <div className="flex gap-2">
      {location.pathname === '/' ? (
        <>
          <Button text severity="secondary" className="text-pink-500" icon="pi pi-folder" label="Projects" onClick={() => scrollTo('projects')} style={{ outline: 'none', boxShadow: 'none' }} />
          <Button text severity="secondary" className="text-pink-500" icon="pi pi-chart-bar" label="Stats" onClick={() => scrollTo('stats')} style={{ outline: 'none', boxShadow: 'none' }} />
          <Button text severity="secondary" className="text-pink-500" icon="pi pi-calendar" label="Contributions" onClick={() => scrollTo('contributions')} style={{ outline: 'none', boxShadow: 'none' }} />
        </>
      ) : (
        <Button text severity="secondary" className="text-pink-500" icon="pi pi-home" label="Home" onClick={() => navigate('/')} style={{ outline: 'none', boxShadow: 'none' }} />
      )}
      {!location.pathname.startsWith('/blog') && (
        <Button text severity="secondary" className="text-pink-500" icon="pi pi-book" label="Blog" onClick={() => navigate('/blog')} style={{ outline: 'none', boxShadow: 'none' }} />
      )}
      {location.pathname !== '/expo' && (
        <Button text severity="secondary" className="text-pink-500" icon="pi pi-star" label="Expo" onClick={() => navigate('/expo')} style={{ outline: 'none', boxShadow: 'none' }} />
      )}
    </div>
  );

  return <Toolbar start={startContent} end={endContent} className="border-none" style={{ position: 'sticky', top: 0, zIndex: 1000, background: '#18181b', borderRadius: '0 0 8px 8px', boxShadow: '0 2px 4px -1px rgba(128, 128, 128, 0.3)' }} />;
}
