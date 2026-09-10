import { useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';
import { Button } from 'antd';
import { UpOutlined } from '@ant-design/icons';

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
      icon={<UpOutlined />}
      onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
      style={{
        position: 'fixed',
        bottom: 24,
        right: 24,
        zIndex: 1100,
        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
      }}
      aria-label="Scroll to top"
    />
  );
}