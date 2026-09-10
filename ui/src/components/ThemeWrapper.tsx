import useToken from 'antd/es/theme/useToken';
import { useEffect } from 'react';

export default function ThemeWrapper({ children }: { children: React.ReactNode }) {
  const [, token] = useToken();

  useEffect(() => {
    const body = document.body;
    body.style.background = token.colorBgLayout;
    body.style.color = token.colorText;
    body.style.transition = 'background-color 0.3s ease, color 0.3s ease';
  }, [token.colorBgLayout, token.colorText]);

  return (
    <div style={{ minHeight: '100vh', background: 'var(--ant-color-bg-layout)' }}>
      {children}
    </div>
  );
}