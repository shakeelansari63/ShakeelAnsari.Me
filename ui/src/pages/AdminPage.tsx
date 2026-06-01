import { useState } from 'react';
import { InputText } from 'primereact/inputtext';
import { Password } from 'primereact/password';
import { Button } from 'primereact/button';
import { Card } from 'primereact/card';
import ToolBar from '../components/shared/ToolBar';

export default function AdminPage() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
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

  if (token) {
    return (
      <>
        <ToolBar />
        <div className="app-container">
          <div className="flex align-items-center justify-content-between mt-4 mb-4">
            <h1 className="text-white text-3xl font-bold m-0">Admin</h1>
            <Button label="Logout" icon="pi pi-sign-out" text severity="secondary" className="text-pink-500" onClick={handleLogout} style={{ outline: 'none', boxShadow: 'none' }} />
          </div>
          <Card className="mb-3">
            <p className="text-gray-400 m-0">Welcome to the admin panel.</p>
          </Card>
        </div>
      </>
    );
  }

  return (
    <>
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
