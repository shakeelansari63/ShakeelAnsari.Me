import { useId } from 'react';
import { useTheme } from '../../context/ThemeContext';
import { settings } from '../../data/settings';
import type { MonthlyContribution } from '../../models/types';
import { niceTicks } from './chartUtils';

interface Props {
  monthly: MonthlyContribution[];
}

const VIEW_WIDTH = 460;
const VIEW_HEIGHT = 180;
const PAD_TOP = 14;
const PAD_BOTTOM = 26;
const PAD_LEFT = 8;
const PAD_RIGHT = 46;

function monotoneLine(points: { x: number; y: number }[]): string {
  const count = points.length;
  if (count < 2) return '';
  const dx: number[] = [];
  const slopes: number[] = [];
  for (let i = 0; i < count - 1; i++) {
    dx[i] = points[i + 1].x - points[i].x;
    slopes[i] = (points[i + 1].y - points[i].y) / dx[i];
  }
  const tangents: number[] = new Array(count);
  tangents[0] = slopes[0];
  tangents[count - 1] = slopes[count - 2];
  for (let i = 1; i < count - 1; i++) {
    if (slopes[i - 1] * slopes[i] <= 0) {
      tangents[i] = 0;
    } else {
      const weight1 = 2 * dx[i] + dx[i - 1];
      const weight2 = dx[i] + 2 * dx[i - 1];
      tangents[i] = (weight1 + weight2) / (weight1 / slopes[i - 1] + weight2 / slopes[i]);
    }
  }
  let path = `M${points[0].x},${points[0].y}`;
  for (let i = 0; i < count - 1; i++) {
    const step = dx[i] / 3;
    path += `C${points[i].x + step},${points[i].y + tangents[i] * step} ${
      points[i + 1].x - step
    },${points[i + 1].y + tangents[i + 1] * step} ${points[i + 1].x},${points[i + 1].y}`;
  }
  return path;
}

export default function ContributionAreaChart({ monthly }: Props) {
  const { theme } = useTheme();
  const gradientId = useId();
  const isLight = theme === 'light';

  const axisColor = isLight ? 'rgba(0, 0, 0, 0.15)' : 'rgba(255, 255, 255, 0.15)';
  const textColor = isLight ? 'rgba(0, 0, 0, 0.45)' : 'rgba(255, 255, 255, 0.45)';
  const accent = settings.themeColor;

  if (monthly.length === 0) {
    return (
      <div
        className="flex items-center justify-center w-full"
        style={{ minHeight: 160, color: 'var(--ant-color-text-tertiary)' }}
      >
        <span className="text-sm">Contribution chart unavailable</span>
      </div>
    );
  }

  const plotWidth = VIEW_WIDTH - PAD_LEFT - PAD_RIGHT;
  const plotHeight = VIEW_HEIGHT - PAD_TOP - PAD_BOTTOM;
  const baseline = VIEW_HEIGHT - PAD_BOTTOM;
  const { ticks, top } = niceTicks(Math.max(...monthly.map((point) => point.value)));

  const points = monthly.map((point, index) => ({
    x: PAD_LEFT + (monthly.length === 1 ? plotWidth / 2 : (index * plotWidth) / (monthly.length - 1)),
    y: PAD_TOP + (1 - point.value / top) * plotHeight,
  }));

  const linePath = monotoneLine(points);
  const areaPath =
    linePath.length > 0
      ? `${linePath}L${points[points.length - 1].x},${baseline}L${points[0].x},${baseline}Z`
      : '';

  return (
    <svg
      viewBox={`0 0 ${VIEW_WIDTH} ${VIEW_HEIGHT}`}
      className="w-full h-auto"
      role="img"
      aria-label="Monthly contributions in the last year"
    >
      <defs>
        <linearGradient id={`areaFill-${gradientId}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={accent} stopOpacity="0.35" />
          <stop offset="100%" stopColor={accent} stopOpacity="0.02" />
        </linearGradient>
      </defs>

      {areaPath.length > 0 && <path d={areaPath} fill={`url(#areaFill-${gradientId})`} />}
      {linePath.length > 0 && (
        <path
          d={linePath}
          fill="none"
          stroke={accent}
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      )}

      <line x1={PAD_LEFT} y1={baseline} x2={PAD_LEFT + plotWidth} y2={baseline} stroke={axisColor} strokeWidth="1" />
      <line
        x1={PAD_LEFT + plotWidth}
        y1={PAD_TOP}
        x2={PAD_LEFT + plotWidth}
        y2={baseline}
        stroke={axisColor}
        strokeWidth="1"
      />

      {ticks.map((tick) => {
        const y = PAD_TOP + (1 - tick / top) * plotHeight;
        return (
          <g key={tick}>
            <line
              x1={PAD_LEFT + plotWidth}
              y1={y}
              x2={PAD_LEFT + plotWidth + 4}
              y2={y}
              stroke={axisColor}
              strokeWidth="1"
            />
            <text x={PAD_LEFT + plotWidth + 8} y={y + 3.5} fontSize="11" fill={textColor}>
              {tick}
            </text>
          </g>
        );
      })}

      {monthly.map((point, index) =>
        index % 2 === 0 ? (
          <text
            key={point.label}
            x={points[index].x}
            y={VIEW_HEIGHT - 8}
            fontSize="11"
            fill={textColor}
            textAnchor={index === 0 ? 'start' : index === monthly.length - 1 ? 'end' : 'middle'}
          >
            {point.label}
          </text>
        ) : null,
      )}
    </svg>
  );
}
