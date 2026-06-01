import { useEffect, useState } from 'react';
import { Button } from 'primereact/button';

export default function ScrollToTop() {
  const [visible, setVisible] = useState(false);

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
      icon="pi pi-chevron-up"
      rounded
      severity="secondary"
      onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
      style={{
        position: 'fixed',
        bottom: '1.5rem',
        right: '1.5rem',
        zIndex: 1100,
        outline: 'none',
        boxShadow: 'none',
      }}
    />
  );
}
