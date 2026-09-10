import { Card } from 'antd';
import type { GitProfile } from '../../models/types';
import { userData } from '../../data/profile';

interface Props {
  profile: GitProfile | null;
}

const codeHeader = (
  <div
    style={{
      display: 'flex',
      alignItems: 'center',
      gap: 8,
      padding: '10px 12px',
      backgroundColor: 'var(--ant-color-bg-container)',
      borderRadius: '8px 8px 0 0',
      borderBottom: '1px solid var(--ant-color-border)',
    }}
  >
    <span style={{ width: 12, height: 12, borderRadius: '50%', backgroundColor: '#ff5f56', display: 'inline-block' }} />
    <span style={{ width: 12, height: 12, borderRadius: '50%', backgroundColor: '#ffbd2e', display: 'inline-block' }} />
    <span style={{ width: 12, height: 12, borderRadius: '50%', backgroundColor: '#27c93f', display: 'inline-block' }} />
  </div>
);

export default function HeroUserSkills({ profile }: Props) {
  return (
    <Card
      style={{
        borderRadius: 12,
        boxShadow: '0 2px 8px rgba(0, 0, 0, 0.08)',
        transition: 'box-shadow 0.3s, transform 0.3s',
        height: '100%',
      }}
      onMouseEnter={e => {
        e.currentTarget.style.boxShadow = '0 8px 24px rgba(0, 0, 0, 0.12)';
        e.currentTarget.style.transform = 'translateY(-2px)';
      }}
      onMouseLeave={e => {
        e.currentTarget.style.boxShadow = '0 2px 8px rgba(0, 0, 0, 0.08)';
        e.currentTarget.style.transform = 'translateY(0)';
      }}
    >
      {codeHeader}
      <div
        style={{
          padding: 16,
          fontFamily: "'Cascadia Code', 'Fira Code', 'JetBrains Mono', monospace",
          lineHeight: 1.6,
          fontSize: '14px',
          color: 'var(--ant-color-text)',
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{ color: 'var(--ant-color-primary)' }}>const</span>
          <span style={{ color: '#a855f7', fontWeight: 600 }}>
            {userData?.alias ?? 'coder'}
          </span>
          <span style={{ color: 'var(--ant-color-primary)' }}> =</span>
          <span> {'{'}</span>
        </div>
        <div style={{ marginTop: 4, display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{ marginLeft: 24 }} />
          <span>name</span>
          <span style={{ color: 'var(--ant-color-primary)' }}>:</span>
          <span style={{ color: '#52c41a' }}>
            "{profile?.name ?? userData.devUsername}"
          </span>
          <span style={{ color: 'var(--ant-color-primary)' }}>:</span>
        </div>
        <div style={{ marginTop: 4, display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{ marginLeft: 24 }} />
          <span>company</span>
          <span style={{ color: 'var(--ant-color-primary)' }}>:</span>
          <span style={{ color: '#52c41a' }}>
            "{profile?.company ?? ''}"
          </span>
          <span style={{ color: 'var(--ant-color-primary)' }}>:</span>
        </div>
        <div style={{ marginTop: 4, display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{ marginLeft: 24 }} />
          <span>location</span>
          <span style={{ color: 'var(--ant-color-primary)' }}>:</span>
          <span style={{ color: '#52c41a' }}>
            "{profile?.location ?? ''}"
          </span>
          <span style={{ color: 'var(--ant-color-primary)' }}>:</span>
        </div>
        <div style={{ marginTop: 4, display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{ marginLeft: 24 }} />
          <span>followers</span>
          <span style={{ color: 'var(--ant-color-primary)' }}>:</span>
          <span style={{ color: '#faad14' }}>
            {profile?.followers ?? 0}
          </span>
          <span style={{ color: 'var(--ant-color-primary)' }}>:</span>
        </div>
        <div style={{ marginTop: 4, display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{ marginLeft: 24 }} />
          <span>following</span>
          <span style={{ color: 'var(--ant-color-primary)' }}>:</span>
          <span style={{ color: '#faad14' }}>
            {profile?.following ?? 0}
          </span>
          <span style={{ color: 'var(--ant-color-primary)' }}>:</span>
        </div>
        <div style={{ marginTop: 4, display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{ marginLeft: 24 }} />
          <span>repositories</span>
          <span style={{ color: 'var(--ant-color-primary)' }}>:</span>
          <span style={{ color: '#faad14' }}>
            {profile?.public_repos ?? 0}
          </span>
          <span style={{ color: 'var(--ant-color-primary)' }}>:</span>
        </div>
        <div style={{ marginTop: 4, display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{ marginLeft: 24 }} />
          <span>hirable</span>
          <span style={{ color: 'var(--ant-color-primary)' }}>:</span>
          <span style={{ color: '#faad14' }}>
            {String(profile?.hireable ?? false)}
          </span>
          <span style={{ color: 'var(--ant-color-primary)' }}>:</span>
        </div>
        <div style={{ marginTop: 4, display: 'flex', alignItems: 'center', gap: 4 }}>
          <span style={{ marginLeft: 20 }}>{'}'}</span>
        </div>
      </div>
    </Card>
  );
}