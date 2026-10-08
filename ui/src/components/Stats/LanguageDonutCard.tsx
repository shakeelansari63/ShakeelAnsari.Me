import { Card, Skeleton } from 'antd';
import { useTheme } from '../../context/ThemeContext';
import type { LanguageStat } from '../../models/types';

interface Props {
  title: string;
  data: LanguageStat[] | null;
  loading: boolean;
}

const SIZE = 160;
const CENTER = SIZE / 2;
const RADIUS = 58;
const STROKE_WIDTH = 30;
const GAP = 2;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

export default function LanguageDonutCard({ title, data, loading }: Props) {
  const { theme } = useTheme();
  const isLight = theme === 'light';
  const tertiaryColor = isLight ? 'rgba(0, 0, 0, 0.45)' : 'rgba(255, 255, 255, 0.45)';
  const borderColor = isLight ? 'rgba(0, 0, 0, 0.09)' : 'rgba(255, 255, 255, 0.09)';

  const total = data ? data.reduce((sum, stat) => sum + stat.value, 0) : 0;
  let accumulated = 0;
  const segments = (data ?? []).map((stat) => {
    const fraction = total > 0 ? stat.value / total : 0;
    const start = accumulated;
    accumulated += fraction;
    return {
      ...stat,
      dash: Math.max(fraction * CIRCUMFERENCE - GAP, 0.5),
      offset: -start * CIRCUMFERENCE,
    };
  });

  return (
    <Card
      className="glow-card"
      style={{ borderRadius: 12, transition: 'box-shadow 0.3s', height: '100%', textAlign: 'left' }}
    >
      {loading ? (
        <Skeleton active title={false} paragraph={{ rows: 5 }} />
      ) : !data || data.length === 0 ? (
        <p
          className="text-center py-6 text-sm"
          style={{ color: tertiaryColor, minHeight: 200, display: 'flex', alignItems: 'center', justifyContent: 'center' }}
        >
          {data ? 'No language data to display' : 'Unable to load language stats right now.'}
        </p>
      ) : (
        <div className="image-entrance">
          <h2 className="text-xl font-semibold" style={{ margin: 0, marginBottom: 16 }}>
            {title}
          </h2>
          <div className="flex flex-col sm:flex-row items-center gap-4">
            <ul className="flex-1 space-y-2.5 min-w-0 w-full">
              {data.map((stat) => (
                <li key={stat.name} className="flex items-center gap-2.5">
                  <span
                    className="shrink-0"
                    style={{
                      width: 14,
                      height: 14,
                      borderRadius: 3,
                      backgroundColor: stat.color,
                      border: `1px solid ${borderColor}`,
                    }}
                  />
                  <span className="text-sm" style={{ overflowWrap: 'anywhere' }}>
                    {stat.name}
                  </span>
                </li>
              ))}
            </ul>
            <svg
              width={SIZE}
              height={SIZE}
              viewBox={`0 0 ${SIZE} ${SIZE}`}
              className="shrink-0"
              role="img"
              aria-label={`${title} chart`}
            >
              <g transform={`rotate(-90 ${CENTER} ${CENTER})`}>
                {segments.map((segment) => (
                  <circle
                    key={segment.name}
                    cx={CENTER}
                    cy={CENTER}
                    r={RADIUS}
                    fill="none"
                    stroke={segment.color}
                    strokeWidth={STROKE_WIDTH}
                    strokeDasharray={`${segment.dash} ${CIRCUMFERENCE - segment.dash}`}
                    strokeDashoffset={segment.offset}
                  >
                    <title>{`${segment.name}: ${segment.value}`}</title>
                  </circle>
                ))}
              </g>
            </svg>
          </div>
        </div>
      )}
    </Card>
  );
}
