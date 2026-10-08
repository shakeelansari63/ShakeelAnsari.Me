import { useEffect, useState } from 'react';
import { Card, Skeleton } from 'antd';
import { useTheme } from '../../context/ThemeContext';
import { settings } from '../../data/settings';
import { fetchStreakStats } from '../../services/stats';
import type { StreakRange, StreakStats } from '../../models/types';

const MONTHS = [
  'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
];

const FIRE_PATH =
  'M 1.5 0.67 C 1.5 0.67 2.24 3.32 2.24 5.47 C 2.24 7.53 0.89 9.2 -1.17 9.2 C -3.23 9.2 -4.79 7.53 -4.79 5.47 L -4.76 5.11 C -6.78 7.51 -8 10.62 -8 13.99 C -8 18.41 -4.42 22 0 22 C 4.42 22 8 18.41 8 13.99 C 8 8.6 5.41 3.79 1.5 0.67 Z M -0.29 19 C -2.07 19 -3.51 17.6 -3.51 15.86 C -3.51 14.24 -2.46 13.1 -0.7 12.74 C 1.07 12.38 2.9 11.53 3.92 10.16 C 4.31 11.45 4.51 12.81 4.51 14.2 C 4.51 16.85 2.36 19 -0.29 19 Z';

function formatDay(date: string, currentYear: string): string {
  const [year, month, day] = date.split('-');
  const label = `${MONTHS[Number(month) - 1] ?? month} ${Number(day)}`;
  return year === currentYear ? label : `${label}, ${year}`;
}

function formatRange(range: StreakRange, currentYear: string): string {
  if (range.start === range.end) {
    return formatDay(range.start, currentYear);
  }
  return `${formatDay(range.start, currentYear)} - ${formatDay(range.end, currentYear)}`;
}

export default function StreakStatsCard() {
  const [stats, setStats] = useState<StreakStats | null>(null);
  const [loading, setLoading] = useState(true);
  const { theme } = useTheme();

  useEffect(() => {
    fetchStreakStats()
      .then(setStats)
      .finally(() => setLoading(false));
  }, []);

  const isLight = theme === 'light';
  const accent = settings.themeColor;
  const tertiaryColor = isLight ? 'rgba(0, 0, 0, 0.45)' : 'rgba(255, 255, 255, 0.45)';
  const borderColor = isLight ? 'rgba(0, 0, 0, 0.09)' : 'rgba(255, 255, 255, 0.09)';

  return (
    <Card
      className="glow-card"
      style={{ borderRadius: 12, transition: 'box-shadow 0.3s', height: '100%', textAlign: 'left' }}
      styles={{
        body: {
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
        },
      }}
    >
      {loading ? (
        <Skeleton active title={false} paragraph={{ rows: 5 }} />
      ) : !stats ? (
        <p
          className="text-center py-6 text-sm"
          style={{
            color: tertiaryColor,
            minHeight: 200,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          Unable to load streak stats right now.
        </p>
      ) : (
        <div className="image-entrance grid grid-cols-3">
          <div className="flex flex-col items-center text-center px-1 pt-[38px] pb-2">
            <div className="text-[28px] font-bold leading-9" style={{ color: accent }}>
              {stats.total.toLocaleString('en-US')}
            </div>
            <div className="text-sm mt-1" style={{ color: accent }}>
              Total Contributions
            </div>
            <div className="text-[11px] mt-1" style={{ color: tertiaryColor }}>
              {formatDay(stats.firstContribution, stats.today.slice(0, 4))} - Present
            </div>
          </div>

          <div
            className="flex flex-col items-center text-center px-1 pb-2"
            style={{ borderLeft: `1px solid ${borderColor}` }}
          >
            <div className="relative w-full max-w-24 h-[104px] mx-auto">
              <svg viewBox="0 0 96 104" className="w-full h-full" aria-hidden="true">
                <circle
                  cx="48"
                  cy="56"
                  r="40"
                  fill="none"
                  stroke={accent}
                  strokeWidth="5"
                  strokeDasharray="224.86 26.47"
                  transform="rotate(-71 48 56)"
                />
                <path d={FIRE_PATH} fill={accent} transform="translate(48 4.5)" />
              </svg>
              <span
                className="absolute inset-0 flex items-center justify-center text-[28px] font-bold leading-none"
                style={{ color: accent }}
              >
                {stats.current.length}
              </span>
            </div>
            <div className="text-sm font-bold mt-1" style={{ color: accent }}>
              Current Streak
            </div>
            <div className="text-[11px] mt-1" style={{ color: tertiaryColor }}>
              {formatRange(stats.current, stats.today.slice(0, 4))}
            </div>
          </div>

          <div
            className="flex flex-col items-center text-center px-1 pt-[38px] pb-2"
            style={{ borderLeft: `1px solid ${borderColor}` }}
          >
            <div className="text-[28px] font-bold leading-9" style={{ color: accent }}>
              {stats.longest.length}
            </div>
            <div className="text-sm mt-1" style={{ color: accent }}>
              Longest Streak
            </div>
            <div className="text-[11px] mt-1" style={{ color: tertiaryColor }}>
              {formatRange(stats.longest, stats.today.slice(0, 4))}
            </div>
          </div>
        </div>
      )}
    </Card>
  );
}
