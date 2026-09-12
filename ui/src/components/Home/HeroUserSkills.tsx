import { Card } from 'antd';
import type { GitProfile } from '../../models/types';
import { userData } from '../../data/profile';
import { settings } from '../../data/settings';
import { seo } from '../../data/seo';

function getCurrentRole(): string {
  for (const job of userData.work) {
    const current = job.roles.find(r => r.endDate === null);
    if (current) return current.title;
  }
  return '';
}

function getYearsOfExperience(): number {
  const starts = userData.work.flatMap(job => job.roles.map(r => r.startDate)).filter(Boolean);
  if (starts.length === 0) return 0;
  const earliest = starts.sort()[0];
  const [y, m] = earliest.split('-').map(Number);
  const now = new Date();
  const months = (now.getFullYear() - y) * 12 + (now.getMonth() + 1 - m);
  return Math.max(0, Math.floor(months / 12));
}

interface Props {
  profile: GitProfile | null;
}

const codeHeader = (
  <div className="code-header">
    <span className="dot-red" />
    <span className="dot-orange" />
    <span className="dot-green" />
  </div>
);

const rowStyle = {
  display: 'flex',
  flexDirection: 'row' as const,
};

const padStyle = { padding: '0 4px' };
const indentStyle = { paddingLeft: 24 };

export default function HeroUserSkills({ profile }: Props) {
  return (
    <Card
      title={codeHeader}
      className="h-full glow-card"
      styles={{
        header: { padding: 0, minHeight: 'auto', borderBottom: 'none' },
        title: { padding: 0 },
        body: { padding: 0 },
      }}
      style={{
        borderRadius: 12,
        boxShadow: '0 2px 8px rgba(0, 0, 0, 0.08)',
        transition: 'box-shadow 0.3s',
        height: '100%',
        overflow: 'hidden',
      }}
    >
      <div
        style={{
          padding: 16,
          overflow: 'auto',
          wordBreak: 'break-word',
          fontFamily: "'Cascadia Code', 'Fira Code', 'JetBrains Mono', monospace",
          lineHeight: 1.5,
          fontSize: '1.125rem',
        }}
      >
        <div style={rowStyle}>
          <span style={{ ...padStyle, color: '#22c55e' }}>$</span>
          <span style={{ ...padStyle, color: settings.themeColor }}>cat</span>
          <span style={{ ...padStyle }}>whoami.js</span>
        </div>
        <div style={rowStyle} className='mt-4'>
          <span style={{ ...padStyle, color: '#6b7280' }}>{`// ${userData.comment}`}</span>
        </div>
        <div style={rowStyle}  className='mt-6'>
          <span style={{ ...padStyle, color: settings.themeColor }}>const</span>
          <span style={{ ...padStyle, color: '#a855f7', fontWeight: 'bold' }}>
            {userData?.alias ?? 'coder'}
          </span>
          <span style={{ ...padStyle, color: settings.themeColor }}>=</span>
          <span style={padStyle}>{'{'}</span>
        </div>
        <div style={rowStyle}>
          <span style={indentStyle} />
          <span style={{ ...padStyle, whiteSpace: 'nowrap' }}>name :</span>
          <span style={{ ...padStyle, color: '#22c55e' }}>
            &quot;{profile?.name ?? userData.devUsername}&quot;
          </span>
          <span style={{ ...padStyle, whiteSpace: 'nowrap' }}>,</span>
        </div>
        <div style={rowStyle}>
          <span style={indentStyle} />
          <span style={{ ...padStyle, whiteSpace: 'nowrap' }}>company :</span>
          <span style={{ ...padStyle, color: '#22c55e' }}>
            &quot;{profile?.company ?? ''}&quot;
          </span>
          <span style={{ ...padStyle, whiteSpace: 'nowrap' }}>,</span>
        </div>
        <div style={rowStyle}>
          <span style={indentStyle} />
          <span style={{ ...padStyle, whiteSpace: 'nowrap' }}>role :</span>
          <span style={{ ...padStyle, color: '#22c55e' }}>
            &quot;{getCurrentRole()}&quot;
          </span>
          <span style={{ ...padStyle, whiteSpace: 'nowrap' }}>,</span>
        </div>
        <div style={rowStyle}>
          <span style={indentStyle} />
          <span style={{ ...padStyle, whiteSpace: 'nowrap' }}>location :</span>
          <span style={{ ...padStyle, color: '#22c55e' }}>
            &quot;{profile?.location ?? ''}&quot;
          </span>
          <span style={{ ...padStyle, whiteSpace: 'nowrap' }}>,</span>
        </div>
        <div style={rowStyle}>
          <span style={indentStyle} />
          <span style={{ ...padStyle, whiteSpace: 'nowrap' }}>experience :</span>
          <span style={{ ...padStyle, color: '#22c55e' }}>
            &quot;{Math.max(0, getYearsOfExperience() - 1)}+ years&quot;
          </span>
          <span style={{ ...padStyle, whiteSpace: 'nowrap' }}>,</span>
        </div>
        <div style={rowStyle}>
          <span style={indentStyle} />
          <span style={{ ...padStyle, whiteSpace: 'nowrap' }}>website :</span>
          <span style={{ ...padStyle, color: '#22c55e' }}>
            &quot;{seo.domain}&quot;
          </span>
          <span style={{ ...padStyle, whiteSpace: 'nowrap' }}>,</span>
        </div>
        <div style={rowStyle}>
          <span style={indentStyle} />
          <span style={{ ...padStyle, whiteSpace: 'nowrap' }}>followers :</span>
          <span style={{ ...padStyle, color: '#f97316' }}>
            {profile?.followers ?? 0}
          </span>
          <span style={{ ...padStyle, whiteSpace: 'nowrap' }}>,</span>
        </div>
        <div style={rowStyle}>
          <span style={indentStyle} />
          <span style={{ ...padStyle, whiteSpace: 'nowrap' }}>following :</span>
          <span style={{ ...padStyle, color: '#f97316' }}>
            {profile?.following ?? 0}
          </span>
          <span style={{ ...padStyle, whiteSpace: 'nowrap' }}>,</span>
        </div>
        <div style={rowStyle}>
          <span style={indentStyle} />
          <span style={{ ...padStyle, whiteSpace: 'nowrap' }}>repositories :</span>
          <span style={{ ...padStyle, color: '#f97316' }}>
            {profile?.public_repos ?? 0}
          </span>
          <span style={{ ...padStyle, whiteSpace: 'nowrap' }}>,</span>
        </div>
        <div style={rowStyle}>
          <span style={indentStyle} />
          <span style={{ ...padStyle, whiteSpace: 'nowrap' }}>hirable :</span>
          <span style={{ ...padStyle, color: '#f97316' }}>
            {String(profile?.hireable ?? false)}
          </span>
          <span style={{ ...padStyle, whiteSpace: 'nowrap' }}>,</span>
        </div>
        <div style={rowStyle}>
          <span style={padStyle}>{'};'}</span>
        </div>
      </div>
    </Card>
  );
}
