import { useState } from 'react';
import { Helmet } from 'react-helmet-async';
import { Input, Button, Card, message } from 'antd';
import { LockOutlined, UserOutlined, LogoutOutlined, SyncOutlined } from '@ant-design/icons';
import ToolBar from '../components/shared/ToolBar';
import AnalyticsDashboard from '../components/Admin/AnalyticsDashboard';
import { seo } from '../data/seo';
import { settings } from '../data/settings';

export default function AdminPage() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [syncingLearn, setSyncingLearn] = useState(false);
  const [token, setToken] = useState(() => localStorage.getItem('admin_token'));

  const handleLogin = async () => {
    setError('');
    setLoading(true);
    try {
      const res = await fetch('/api/admin/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      });
      const data = await res.json();
      if (!res.ok) {
        setError(data.error || 'Login failed');
        return;
      }
      localStorage.setItem('admin_token', data.token);
      setToken(data.token);
      message.success('Logged in successfully');
    } catch {
      setError('Network error');
      message.error('Network error');
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('admin_token');
    setToken(null);
    message.success('Logged out');
  };

  const handleSync = async () => {
    setSyncing(true);
    try {
      const res = await fetch('/api/admin/sync-blogs', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
      });
      if (res.status === 401) {
        localStorage.removeItem('admin_token');
        setToken(null);
        message.warning('Session expired. Please log in again.');
        return;
      }
      const data = await res.json();
      message[res.ok ? 'success' : 'error'](data.message || data.error || (res.ok ? 'Synced' : 'Error'));
    } catch {
      message.error('Network error');
    } finally {
      setSyncing(false);
    }
  };

  const handleSyncLearn = async () => {
    setSyncingLearn(true);
    try {
      const res = await fetch('/api/admin/sync-learn', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
      });
      if (res.status === 401) {
        localStorage.removeItem('admin_token');
        setToken(null);
        message.warning('Session expired. Please log in again.');
        return;
      }
      const data = await res.json();
      message[res.ok ? 'success' : 'error'](data.message || data.error || (res.ok ? 'Synced' : 'Error'));
    } catch {
      message.error('Network error');
    } finally {
      setSyncingLearn(false);
    }
  };

  if (token) {
    return (
      <>
        <Helmet>
          <title>{`Admin — ${seo.name}`}</title>
          <meta name="robots" content="noindex, nofollow" />
        </Helmet>
        <ToolBar />
        <div className="app-container">
          <div className="flex items-center justify-between mt-4 mb-6">
            <h1 className="text-3xl font-bold m-0" style={{ color: 'var(--ant-color-text)' }}>Admin</h1>
            <Button type="text" icon={<LogoutOutlined />} onClick={handleLogout} danger>
              Logout
            </Button>
          </div>
          <Card style={{ marginBottom: 16, borderRadius: 12 }}>
            <p className="m-0" style={{ color: 'var(--ant-color-text-secondary)' }}>Welcome to the admin panel.</p>
          </Card>
          <Card style={{ marginBottom: 16, borderRadius: 12 }}>
            <div className="flex items-center justify-between">
              <span style={{ color: 'var(--ant-color-text-secondary)' }}>Sync blog metadata from markdown files</span>
              <Button icon={<SyncOutlined />} loading={syncing} onClick={handleSync}>
                Sync Blogs
              </Button>
            </div>
          </Card>
          <Card style={{ marginBottom: 16, borderRadius: 12 }}>
            <div className="flex items-center justify-between">
              <span style={{ color: 'var(--ant-color-text-secondary)' }}>Sync learning subjects and chapters from markdown files</span>
              <Button icon={<SyncOutlined />} loading={syncingLearn} onClick={handleSyncLearn}>
                Sync Learning
              </Button>
            </div>
          </Card>

          <Card className="no-glow" style={{ marginTop: 16, borderRadius: 12 }}>
            <span className="text-lg font-bold block mb-4" style={{ color: 'var(--ant-color-text)' }}>
              Blog Analytics & Insights
            </span>
            <AnalyticsDashboard token={token} />
          </Card>
        </div>
        <span className='mt-2 mb-2'>&nbsp;</span>
      </>
    );
  }

  return (
    <>
      <Helmet>
        <title>{`Admin — ${seo.name}`}</title>
        <meta name="robots" content="noindex, nofollow" />
      </Helmet>
      <ToolBar />
      <div className="app-container">
        <div className="flex justify-center mt-12">
          <Card title="Admin Login" style={{ maxWidth: 400, width: '100%', borderRadius: 12 }}>
            <div className="space-y-4">
              <div>
                <label htmlFor="username" className="block mb-1" style={{ color: 'var(--ant-color-text)' }}>
                  Username
                </label>
                <Input
                  id="username"
                  prefix={<UserOutlined />}
                  value={username}
                  onChange={e => setUsername(e.target.value)}
                  style={{ width: '100%' }}
                  onPressEnter={handleLogin}
                />
              </div>
              <div>
                <label htmlFor="password" className="block mb-1" style={{ color: 'var(--ant-color-text)' }}>
                  Password
                </label>
                <Input.Password
                  id="password"
                  prefix={<LockOutlined />}
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  style={{ width: '100%' }}
                  onPressEnter={handleLogin}
                />
              </div>
              {error && <p className="m-0" style={{ color: '#ff4d4f' }}>{error}</p>}
              <Button
                type="primary"
                icon={<LockOutlined />}
                loading={loading}
                onClick={handleLogin}
                block
                shape="round"
                className="gradient-btn"
                style={{
                  marginTop: 8,
                  background: `linear-gradient(135deg, ${settings.themeColor} 0%, #743ad5 100%)`,
                  border: 'none',
                  color: '#fff',
                  fontWeight: 500,
                }}
              >
                Login
              </Button>
            </div>
          </Card>
        </div>
      </div>
    </>
  );
}
