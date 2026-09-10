import { Card, Button } from 'antd';
import { GithubOutlined, MailOutlined, LinkedinOutlined, TwitterOutlined, CheckCircleOutlined } from '@ant-design/icons';
import LazyImage from '../shared/LazyImage';
import type { GitProfile } from '../../models/types';
import { userData } from '../../data/profile';

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
      className="h-full"
      style={{
        borderRadius: 12,
        boxShadow: '0 2px 8px rgba(0, 0, 0, 0.08)',
        transition: 'box-shadow 0.3s, transform 0.3s',
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
      <div className="flex flex-col items-center p-6 text-center">
        <LazyImage
          src={profile?.avatar_url ?? ''}
          alt="Avatar"
          maxWidth={180}
          maxHeight={180}
          aspectRatio="1 / 1"
          rounded
        />
        {profile?.bio && (
          <div className="mt-4 text-lg" style={{ color: 'var(--ant-color-text-secondary)' }}>
            ⚡{profile.bio}⚡
          </div>
        )}
        <div className="mt-6 flex flex-wrap justify-center gap-3">
          {socialLinks.map((link, index) => (
            <a
              key={index}
              href={link.href}
              target="_blank"
              rel="noopener noreferrer"
              title={link.label}
              style={{ color: 'var(--ant-color-primary)', transition: 'color 0.2s, transform 0.2s' }}
              onMouseEnter={e => {
                e.currentTarget.style.color = 'var(--ant-color-primary-hover)';
                e.currentTarget.style.transform = 'scale(1.1)';
              }}
              onMouseLeave={e => {
                e.currentTarget.style.color = 'var(--ant-color-primary)';
                e.currentTarget.style.transform = 'scale(1)';
              }}
            >
              <span style={{ fontSize: '1.5rem' }}>{link.icon}</span>
            </a>
          ))}
        </div>
        <div className="mt-6">
          <Button
            type="default"
            icon={<CheckCircleOutlined />}
            onClick={() => window.open(userData.badges, '_blank')}
            style={{
              background: 'linear-gradient(135deg, var(--ant-color-primary) 0%, #743ad5 100%)',
              border: 'none',
              borderRadius: 8,
              padding: '8px 20px',
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