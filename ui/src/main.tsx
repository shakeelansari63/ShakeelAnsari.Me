import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { HelmetProvider } from 'react-helmet-async';
import { ConfigProvider, theme } from 'antd';
import 'antd/dist/reset.css';
import { ThemeProvider, useTheme } from './context/ThemeContext';
import ThemeWrapper from './components/ThemeWrapper';
import { settings } from './data/settings';
import './App.scss';
import App from './App';

function AppWithTheme() {
  const { theme: themeMode } = useTheme();

  return (
    <ConfigProvider
      theme={{
        algorithm: themeMode === 'dark' ? theme.darkAlgorithm : theme.defaultAlgorithm,
        token: {
          colorPrimary: settings.themeColor,
          colorSuccess: '#52c41a',
          colorWarning: '#faad14',
          colorError: '#ff4d4f',
          colorInfo: '#1890ff',
          borderRadius: 8,
          fontFamily: "'Space Grotesk', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif",
          fontSize: 14,
        },
        components: {
          Button: {
            controlHeight: 36,
            fontWeight: 500,
          },
          Card: {
            borderRadiusLG: 12,
            boxShadow: '0 2px 8px rgba(0, 0, 0, 0.08)',
            boxShadowSecondary: '0 4px 16px rgba(0, 0, 0, 0.12)',
          },
          Layout: {
            headerBg: 'transparent',
            siderBg: 'transparent',
          },
          Input: {
            borderRadius: 8,
            controlHeight: 36,
          },
          Select: {
            borderRadius: 8,
            controlHeight: 36,
          },
          Table: {
            borderRadius: 8,
            headerBg: 'transparent',
            rowHoverBg: 'transparent',
          },
          Modal: {
            borderRadiusLG: 12,
            contentBg: 'transparent',
          },
          Drawer: {
            borderRadiusLG: 12,
          },
          Tag: {
            borderRadius: 6,
            fontSize: 12,
          },
          Breadcrumb: {
            itemColor: 'rgba(0, 0, 0, 0.65)',
          },
          Pagination: {
            itemBg: 'transparent',
            itemActiveBg: settings.themeColor,
          },
          Skeleton: {
            colorFill: 'rgba(0, 0, 0, 0.06)',
            colorFillContent: 'rgba(0, 0, 0, 0.1)',
          },
        },
      }}
    >
      <ThemeWrapper>
        <App />
      </ThemeWrapper>
    </ConfigProvider>
  );
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <HelmetProvider>
      <BrowserRouter>
        <ThemeProvider>
          <AppWithTheme />
        </ThemeProvider>
      </BrowserRouter>
    </HelmetProvider>
  </StrictMode>
);