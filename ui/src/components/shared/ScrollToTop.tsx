import { useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';
import { Button } from 'antd';
import { UpOutlined } from '@ant-design/icons';
import { settings } from '../../data/settings';

export default function ScrollToTop() {
  const { pathname } = useLocation();
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);

  useEffect(() => {
    const check = () => setVisible(document.documentElement.classList.contains('scrolled'));
    const observer = new MutationObserver(check);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    check();
    return () => observer.disconnect();
  }, []);

  if (!visible) return null;

  return (
    <Button
      type="primary"
      shape="circle"
      icon={<UpOutlined style={{ color: '#fff' }} />}
      onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
      className="gradient-btn"
      style={{
        position: 'fixed',
        bottom: 24,
        right: 24,
        zIndex: 1100,
        background: `linear-gradient(135deg, ${settings.themeColor} 0%, #743ad5 100%)`,
        border: 'none',
        color: '#fff',
      }}
      aria-label="Scroll to top"
    />
  );
}