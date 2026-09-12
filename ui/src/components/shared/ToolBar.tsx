import { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Layout, Menu, Button, Drawer, Space } from 'antd';
import {
  HomeOutlined,
  StarOutlined,
  BookOutlined,
  ReadOutlined,
  BarChartOutlined,
  CalendarOutlined,
  FolderOutlined,
  SunOutlined,
  MoonOutlined,
  MenuOutlined,
  CloseOutlined,
} from '@ant-design/icons';
import { userData } from '../../data/profile';
import { settings } from '../../data/settings';
import { useTheme } from '../../context/ThemeContext';
import styles from './ToolBar.module.scss';

const { Header } = Layout;

export default function ToolBar() {
  const navigate = useNavigate();
  const location = useLocation();
  const [menuOpen, setMenuOpen] = useState(false);
  const { theme, toggleTheme } = useTheme();

  const scrollTo = (id: string) => {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const statsNav = (id: string) => {
    setMenuOpen(false);
    if (location.pathname !== '/stats') {
      navigate('/stats');
      setTimeout(() => scrollTo(id), 300);
    } else {
      scrollTo(id);
    }
  };

  const menuItems = [
    ...(settings.showExpo
      ? [
          {
            key: 'expo',
            label: 'Expo',
            icon: <StarOutlined />,
            visible: !location.pathname.startsWith('/expo'),
            action: () => {
              setMenuOpen(false);
              navigate('/expo');
            },
          },
        ]
      : []),
    ...(settings.showBlogs
      ? [
          {
            key: 'blogs',
            label: 'Blogs',
            icon: <BookOutlined />,
            visible: !location.pathname.startsWith('/blog'),
            action: () => {
              setMenuOpen(false);
              navigate('/blog');
            },
          },
        ]
      : []),
    ...(settings.showTutorial
      ? [
          {
            key: 'learn',
            label: 'Learn',
            icon: <ReadOutlined />,
            visible: !location.pathname.startsWith('/learn'),
            action: () => {
              setMenuOpen(false);
              navigate('/learn');
            },
          },
        ]
      : []),
    {
      key: 'stats',
      label: 'Stats',
      icon: <BarChartOutlined />,
      visible: !location.pathname.startsWith('/stats'),
      action: () => {
        setMenuOpen(false);
        navigate('/stats');
      },
    },
    {
      key: 'contributions',
      label: 'Contributions',
      icon: <CalendarOutlined />,
      visible: location.pathname === '/stats',
      action: () => statsNav('contributions'),
    },
    {
      key: 'projects',
      label: 'Projects',
      icon: <FolderOutlined />,
      visible: location.pathname === '/stats',
      action: () => statsNav('projects'),
    },
  ];

  const visibleMenuItems = menuItems.filter(item => item.visible);

  const handleMenuClick = (key: string) => {
    const item = visibleMenuItems.find(m => m.key === key);
    if (item) item.action();
  };

  const startContent = location.pathname === '/' || location.pathname === '/stats' ? (
    <span
      className={styles.brand}
      onClick={() => navigate('/')}
    >
      @{userData.devUsername}
    </span>
  ) : (
    <Button type="text" icon={<HomeOutlined />} onClick={() => navigate('/')}>
      Home
    </Button>
  );

  const navItems = (
    <Space className={styles.navGroup} size="small">
      {settings.showExpo && !location.pathname.startsWith('/expo') && (
        <Button type="text" icon={<StarOutlined />} onClick={() => navigate('/expo')}>
          Expo
        </Button>
      )}
      {settings.showBlogs && !location.pathname.startsWith('/blog') && (
        <Button type="text" icon={<BookOutlined />} onClick={() => navigate('/blog')}>
          Blogs
        </Button>
      )}
      {settings.showTutorial && !location.pathname.startsWith('/learn') && (
        <Button type="text" icon={<ReadOutlined />} onClick={() => navigate('/learn')}>
          Learn
        </Button>
      )}
      {!location.pathname.startsWith('/stats') && (
        <Button type="text" icon={<BarChartOutlined />} onClick={() => navigate('/stats')}>
          Stats
        </Button>
      )}
      {location.pathname === '/stats' && (
        <>
          <Button type="text" icon={<CalendarOutlined />} onClick={() => statsNav('contributions')}>
            Contributions
          </Button>
          <Button type="text" icon={<FolderOutlined />} onClick={() => statsNav('projects')}>
            Projects
          </Button>
        </>
      )}
    </Space>
  );

  const themeToggle = (
    <Button
      type="text"
      icon={theme === 'light' ? <MoonOutlined /> : <SunOutlined />}
      onClick={toggleTheme}
      aria-label={theme === 'light' ? 'Switch to Dark Mode' : 'Switch to Light Mode'}
      className={styles.themeToggle}
    />
  );

  const mobileMenuButton = (
    <Button
      type="text"
      icon={menuOpen ? <CloseOutlined /> : <MenuOutlined />}
      onClick={() => setMenuOpen(open => !open)}
      aria-label={menuOpen ? 'Close menu' : 'Open menu'}
      className={styles.mobileMenuBtn}
    />
  );

  return (
    <>
      <Drawer
        title="Navigation"
        placement="right"
        open={menuOpen}
        onClose={() => setMenuOpen(false)}
        width={240}
        styles={{
          header: { borderBottom: 'none', padding: '16px 20px' },
          body: { padding: '8px 16px' },
        }}
        className={styles.drawer}
      >
        <Menu
          mode="inline"
          selectable
          onClick={({ key }) => handleMenuClick(key)}
          style={{ borderRight: 'none' }}
          items={visibleMenuItems.map(item => ({
            key: item.key,
            label: item.label,
            icon: item.icon,
          }))}
        />
      </Drawer>
      <Header className={styles.header}>
        <div className={styles.headerInner}>
          <div className={styles.start}>{startContent}</div>
          <div className={styles.end}>
            {navItems}
            {themeToggle}
            {mobileMenuButton}
          </div>
        </div>
      </Header>
    </>
  );
}
