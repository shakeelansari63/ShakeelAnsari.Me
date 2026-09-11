import { Card, Button } from 'antd';
import { GithubOutlined, MailOutlined, LinkedinOutlined, TwitterOutlined, CheckCircleOutlined } from '@ant-design/icons';
import LazyImage from '../shared/LazyImage';
import type { GitProfile } from '../../models/types';
import { userData } from '../../data/profile';
import { settings } from '../../data/settings';

interface Props {
  profile: GitProfile | null;
}

export default function HeroUserDetail({ profile }: Props) {
  const socialLinks = [
    { href: userData.github, icon: <GithubOutlined />, label: 'GitHub' },
    { href: userData.email, icon: <MailOutlined />, label: 'Email' },
    { href: userData.linkedIn, icon: <LinkedinOutlined />, label: 'LinkedIn' },
    { href: userData.twitter, icon: <TwitterOutlined />, label: 'Twitter' },
  ];

  return (
    <Card
      className="h-full glow-card"
      style={{
        borderRadius: 12,
        boxShadow: '0 2px 8px rgba(0, 0, 0, 0.08)',
        transition: 'box-shadow 0.3s',
      }}
    >
      <div className="flex flex-col items-center p-2 text-center">
        <LazyImage
          src={profile?.avatar_url ?? ''}
          alt="Avatar"
          maxWidth={180}
          maxHeight={180}
          aspectRatio="1 / 1"
          rounded
        />
        {profile?.bio && (
          <div className="mt-2" style={{ color: 'var(--ant-color-text-secondary)', fontSize: '1.2rem', lineHeight: 1.5, textAlign: 'center' }}>
            ⚡{profile.bio}⚡
          </div>
        )}
        <div className="mt-1 flex flex-wrap w-full px-12" style={{ display: 'flex', justifyContent: 'space-around', alignItems: 'center', gap: 20, marginTop: 28, marginBottom: 12 }}>
          {socialLinks.map((link, index) => (
            <a
              key={index}
              href={link.href}
              target="_blank"
              rel="noopener noreferrer"
              title={link.label}
              className="social-link"
              style={{ color: settings.themeColor, padding: '4px 8px' }}
            >
              <span style={{ fontSize: '1.15rem' }}>{link.icon}</span>
            </a>
          ))}
        </div>
        <div className="mt-1" style={{ display: 'flex', justifyContent: 'center', width: '100%' }}>
          <Button
            type="primary"
            icon={<CheckCircleOutlined />}
            shape="round"
            className="gradient-btn"
            onClick={() => window.open(userData.badges, '_blank')}
            style={{
              background: `linear-gradient(135deg, ${settings.themeColor} 0%, #743ad5 100%)`,
              border: 'none',
              color: '#fff',
              fontWeight: 500,
            }}
          >
            Check my badges
          </Button>
        </div>
      </div>
    </Card>
  );
}
