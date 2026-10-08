import { useEffect, useState } from 'react';
import { Card, Skeleton } from 'antd';
import { useTheme } from '../../context/ThemeContext';
import { settings } from '../../data/settings';
import { userData } from '../../data/profile';
import { fetchProductiveTime } from '../../services/stats';
import { niceTicks } from './chartUtils';

const VIEW_WIDTH = 340;
const VIEW_HEIGHT = 200;
const PAD_LEFT = 38;
const PAD_RIGHT = 10;
const PAD_TOP = 14;
const PLOT_HEIGHT = 112;
const X_TICK_HOURS = [0, 6, 12, 18, 23];

function formatUtcTitle(): string {
  const parsed = Number(userData.timezone);
  const offset = Number.isFinite(parsed) ? parsed : 0;
  const sign = offset >= 0 ? '+' : '';
  return `Commits (UTC ${sign}${offset.toFixed(2)})`;
}

export default function ProductiveTimeCard() {
  const [buckets, setBuckets] = useState<number[] | null>(null);
  const [loading, setLoading] = useState(true);
  const { theme } = useTheme();

  useEffect(() => {
    fetchProductiveTime()
      .then(setBuckets)
      .finally(() => setLoading(false));
  }, []);

  const isLight = theme === 'light';
  const axisColor = isLight ? 'rgba(0, 0, 0, 0.15)' : 'rgba(255, 255, 255, 0.15)';
  const textColor = isLight ? 'rgba(0, 0, 0, 0.45)' : 'rgba(255, 255, 255, 0.45)';
  const tertiaryColor = textColor;
  const accent = settings.themeColor;

  const plotWidth = VIEW_WIDTH - PAD_LEFT - PAD_RIGHT;
  const baseline = PAD_TOP + PLOT_HEIGHT;
  const bandWidth = plotWidth / 24;
  const barWidth = bandWidth * 0.9;
  const { ticks, top } = niceTicks(buckets ? Math.max(...buckets) : 0);

  return (
    <Card
      className="glow-card"
      style={{ borderRadius: 12, transition: 'box-shadow 0.3s', height: '100%', textAlign: 'left' }}
    >
      {loading ? (
        <Skeleton active title={false} paragraph={{ rows: 5 }} />
      ) : !buckets ? (
        <p
          className="text-center py-6 text-sm"
          style={{ color: tertiaryColor, minHeight: 200, display: 'flex', alignItems: 'center', justifyContent: 'center' }}
        >
          No recent commits to show.
        </p>
      ) : (
        <div className="image-entrance">
          <h2 className="text-xl font-semibold" style={{ margin: 0, marginBottom: 8 }}>
            {formatUtcTitle()}
          </h2>
          <svg
            viewBox={`0 0 ${VIEW_WIDTH} ${VIEW_HEIGHT}`}
            className="w-full h-auto"
            role="img"
            aria-label="Commits per hour of day"
          >
            <line x1={PAD_LEFT} y1={PAD_TOP} x2={PAD_LEFT} y2={baseline} stroke={axisColor} strokeWidth="1" />
            <line x1={PAD_LEFT} y1={baseline} x2={PAD_LEFT + plotWidth} y2={baseline} stroke={axisColor} strokeWidth="1" />

            {ticks.map((tick) => {
              const y = baseline - (tick / top) * PLOT_HEIGHT;
              return (
                <g key={tick}>
                  <line x1={PAD_LEFT - 4} y1={y} x2={PAD_LEFT} y2={y} stroke={axisColor} strokeWidth="1" />
                  <text x={PAD_LEFT - 7} y={y + 3.5} fontSize="11" fill={textColor} textAnchor="end">
                    {tick}
                  </text>
                </g>
              );
            })}

            {buckets.map((count, hour) => {
              const height = (count / top) * PLOT_HEIGHT;
              const x = PAD_LEFT + hour * bandWidth + (bandWidth - barWidth) / 2;
              return (
                <rect
                  key={hour}
                  x={x}
                  y={baseline - height}
                  width={barWidth}
                  height={height}
                  fill={accent}
                >
                  <title>{`${count} commits around ${hour}:00`}</title>
                </rect>
              );
            })}

            {X_TICK_HOURS.map((hour) => (
              <text
                key={hour}
                x={PAD_LEFT + hour * bandWidth + bandWidth / 2}
                y={baseline + 16}
                fontSize="11"
                fill={textColor}
                textAnchor="middle"
              >
                {hour}
              </text>
            ))}

            <text x={VIEW_WIDTH - PAD_RIGHT} y={VIEW_HEIGHT - 8} fontSize="10" fill={textColor} textAnchor="end">
              per day hour
            </text>
          </svg>
        </div>
      )}
    </Card>
  );
}
