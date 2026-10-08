import { useEffect, useState } from 'react';
import type { ReactNode } from 'react';
import { Card, Skeleton } from 'antd';
import {
  BankOutlined,
  ClockCircleOutlined,
  EnvironmentOutlined,
  GithubOutlined,
  MailOutlined,
  ProjectOutlined,
} from '@ant-design/icons';
import { useTheme } from '../../context/ThemeContext';
import type { ProfileStats } from '../../models/types';
import { fetchProfileStats } from '../../services/stats';
import ContributionAreaChart from './ContributionAreaChart';

function abbreviateNumber(value: number): string {
  if (value < 1000) return `${value}`;
  const units = ['k', 'M', 'B', 'T'];
  let number = value;
  let unitIndex = -1;
  do {
    number /= 1000;
    unitIndex += 1;
  } while (number >= 1000 && unitIndex < units.length - 1);
  return `${parseFloat(number.toFixed(2))}${units[unitIndex]}`;
}

function formatJoined(createdAt: string | null): string | null {
  if (!createdAt) return null;
  const created = new Date(createdAt);
  if (Number.isNaN(created.getTime())) return null;
  const diff = new Date(Date.now() - created.getTime());
  const years = diff.getUTCFullYear() - 1970;
  const months = diff.getUTCMonth();
  const days = diff.getUTCDate() - 1;
  const plural = (count: number) => (count === 1 ? '' : 's');
  if (years > 0) return `Joined GitHub ${years} year${plural(years)} ago`;
  if (months > 0) return `Joined GitHub ${months} month${plural(months)} ago`;
  return `Joined GitHub ${days} day${plural(days)} ago`;
}

function buildRows(stats: ProfileStats): { icon: ReactNode; text: string }[] {
  const rows: { icon: ReactNode; text: string }[] = [];
  if (stats.contributions !== null) {
    rows.push({
      icon: <GithubOutlined />,
      text: `${abbreviateNumber(stats.contributions)} Contributions last year`,
    });
  }
  rows.push({
    icon: <ProjectOutlined />,
    text: `${abbreviateNumber(stats.publicRepos)} Public Repos`,
  });
  const joined = formatJoined(stats.createdAt);
  if (joined) rows.push({ icon: <ClockCircleOutlined />, text: joined });
  if (stats.email) {
    rows.push({ icon: <MailOutlined />, text: stats.email });
  } else if (stats.company) {
    rows.push({ icon: <BankOutlined />, text: stats.company });
  } else if (stats.location) {
    rows.push({ icon: <EnvironmentOutlined />, text: stats.location });
  }
  return rows;
}

export default function ProfileStatsCard() {
  const [stats, setStats] = useState<ProfileStats | null>(null);
  const [loading, setLoading] = useState(true);
  const { theme } = useTheme();

  useEffect(() => {
    fetchProfileStats()
      .then(setStats)
      .finally(() => setLoading(false));
  }, []);

  const isLight = theme === 'light';
  const secondaryColor = isLight ? 'rgba(0, 0, 0, 0.65)' : 'rgba(255, 255, 255, 0.65)';
  const tertiaryColor = isLight ? 'rgba(0, 0, 0, 0.45)' : 'rgba(255, 255, 255, 0.45)';

  return (
    <Card className="glow-card" style={{ borderRadius: 12, transition: 'box-shadow 0.3s' }}>
      {loading ? (
        <Skeleton active title={false} paragraph={{ rows: 5 }} />
      ) : !stats ? (
        <p className="text-center py-6 text-sm" style={{ color: tertiaryColor }}>
          Unable to load profile stats right now.
        </p>
      ) : (
        <div className="image-entrance text-left">
          <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
            <h2 className="text-xl font-semibold" style={{ margin: 0 }}>
              {stats.title}
            </h2>
            {stats.contributions !== null && (
              <span className="text-xs" style={{ color: tertiaryColor }}>
                contributions in the last year
              </span>
            )}
          </div>
          <div className="flex flex-col md:flex-row gap-6 mt-4">
            <ul className="space-y-3 shrink-0 md:w-80">
              {buildRows(stats).map((row, index) => (
                <li key={index} className="flex items-center gap-2.5">
                  <span style={{ color: secondaryColor, fontSize: 16, display: 'flex' }}>
                    {row.icon}
                  </span>
                  <span className="text-sm" style={{ overflowWrap: 'anywhere' }}>
                    {row.text}
                  </span>
                </li>
              ))}
            </ul>
            <div className="flex-1 min-w-0">
              <ContributionAreaChart monthly={stats.monthly} />
            </div>
          </div>
        </div>
      )}
    </Card>
  );
}
