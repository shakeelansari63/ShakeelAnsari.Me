import { useState, useRef } from 'react';
import { Helmet } from 'react-helmet-async';
import { InputText } from 'primereact/inputtext';
import { Password } from 'primereact/password';
import { Button } from 'primereact/button';
import { Card } from 'primereact/card';
import { Toast } from 'primereact/toast';
import ToolBar from '../components/shared/ToolBar';
import AnalyticsDashboard from '../components/Admin/AnalyticsDashboard';
import { seo } from '../data/seo';

export default function AdminPage() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [syncingLearn, setSyncingLearn] = useState(false);
  const [token, setToken] = useState(() => localStorage.getItem('admin_token'));
  const toast = useRef<Toast>(null);

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
    } catch {
      setError('Network error');
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('admin_token');
    setToken(null);
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
        toast.current?.show({ severity: 'warn', summary: 'Session Expired', detail: 'Please log in again', life: 5000 });
        return;
      }
      const data = await res.json();
      toast.current?.show({
        severity: res.ok ? 'success' : 'error',
        summary: res.ok ? 'Synced' : 'Error',
        detail: data.message || data.error || 'Unknown error',
        life: 3000,
      });
    } catch {
      toast.current?.show({ severity: 'error', summary: 'Error', detail: 'Network error', life: 3000 });
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
        toast.current?.show({ severity: 'warn', summary: 'Session Expired', detail: 'Please log in again', life: 5000 });
        return;
      }
      const data = await res.json();
      toast.current?.show({
        severity: res.ok ? 'success' : 'error',
        summary: res.ok ? 'Synced' : 'Error',
        detail: data.message || data.error || 'Unknown error',
        life: 3000,
      });
    } catch {
      toast.current?.show({ severity: 'error', summary: 'Error', detail: 'Network error', life: 3000 });
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
        <Toast ref={toast} />
        <div className="app-container">
          <div className="flex align-items-center justify-content-between mt-4 mb-4">
            <h1 className="text-white text-3xl font-bold m-0">Admin</h1>
            <Button label="Logout" icon="pi pi-sign-out" text severity="secondary" className="text-pink-500" onClick={handleLogout} style={{ outline: 'none', boxShadow: 'none' }} />
          </div>
          <Card className="mb-3">
            <p className="text-gray-400 m-0">Welcome to the admin panel.</p>
          </Card>
          <Card className="mb-3">
            <div className="flex align-items-center justify-content-between">
              <span className="text-gray-400">Sync blog metadata from markdown files</span>
              <Button label="Sync Blogs" icon="pi pi-refresh" loading={syncing} onClick={handleSync} style={{ outline: 'none', boxShadow: 'none' }} />
            </div>
          </Card>
          <Card>
            <div className="flex align-items-center justify-content-between">
              <span className="text-gray-400">Sync learning subjects and chapters from markdown files</span>
              <Button label="Sync Learning" icon="pi pi-refresh" loading={syncingLearn} onClick={handleSyncLearn} style={{ outline: 'none', boxShadow: 'none' }} />
            </div>
          </Card>

          <Card className="mt-3 no-hover">
            <span className="text-gray-400 text-lg font-bold block mb-3">Blog Analytics &amp; Insights</span>
            <AnalyticsDashboard token={token} />
          </Card>
        </div>
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
        <div className="flex justify-content-center mt-6">
          <Card title="Admin Login" className="w-full" style={{ maxWidth: '400px' }} pt={{ body: { className: 'p-3' } }}>
            <div className="flex flex-column gap-3">
              <div>
                <label htmlFor="username" className="text-white block mb-1">Username</label>
                <InputText
                  id="username"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="w-full"
                  style={{ outline: 'none', boxShadow: 'none' }}
                />
              </div>
              <div>
                <label htmlFor="password" className="text-white block mb-1">Password</label>
                <Password
                  id="password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  feedback={false}
                  className="w-full"
                  inputClassName="w-full"
                  inputStyle={{ outline: 'none', boxShadow: 'none' }}
                  onKeyDown={(e) => e.key === 'Enter' && handleLogin()}
                />
              </div>
              {error && <p className="text-pink-500 m-0">{error}</p>}
              <Button label="Login" icon="pi pi-lock" loading={loading} onClick={handleLogin} style={{ outline: 'none', boxShadow: 'none' }} />
            </div>
          </Card>
        </div>
      </div>
    </>
  );
}
