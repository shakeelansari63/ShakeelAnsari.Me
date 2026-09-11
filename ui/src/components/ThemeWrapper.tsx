import useToken from 'antd/es/theme/useToken';
import { useEffect } from 'react';
import { useTheme } from '../context/ThemeContext';

const LIGHT_APP_BG = '#f2f2f2';

export default function ThemeWrapper({ children }: { children: React.ReactNode }) {
  const [, token] = useToken();
  const { theme } = useTheme();
  const appBg = theme === 'dark' ? token.colorBgLayout : LIGHT_APP_BG;

  useEffect(() => {
    const body = document.body;
    body.style.background = appBg;
    body.style.color = token.colorText;
    body.style.transition = 'background-color 0.3s ease, color 0.3s ease';
  }, [appBg, token.colorText]);

  return (
    <div style={{ minHeight: '100vh', background: appBg }}>
      {children}
    </div>
  );
}
