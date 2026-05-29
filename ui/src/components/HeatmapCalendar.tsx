import { useMemo } from 'react';
import type { HeatMapDate, CellData } from '../models/types';

const DAYS_IN_WEEK = 7;
const MILISECONDS_IN_DAY = 1000 * 60 * 60 * 24;
const GAP = 3;
const RECT_SIZE = 12;
const WIDTH_LABEL = 30;
const HEIGHT_LABEL = 20;

const mapedDays: Record<string, number> = { '0': 6, '1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5 };
const daysLabels = ['Mon', '', 'Wed', '', 'Fri', '', 'Sun'];

function convertToDate(date: Date) {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate());
}

function getNumberDays(startDate: Date, endDate: Date) {
  const timeDiff =
    new Date(endDate.setHours(0, 0, 0, 0)).getTime() -
    new Date(startDate.setHours(0, 0, 0, 0)).getTime();
  return Math.abs(Math.round(timeDiff / MILISECONDS_IN_DAY));
}

function getEmptyDaysAtStart(startDate: Date) {
  return mapedDays[convertToDate(startDate).getDay().toString()] ?? 0;
}

function getEmptyDaysAtEnd(endDate: Date) {
  return DAYS_IN_WEEK - 1 - (mapedDays[convertToDate(endDate).getDay().toString()] ?? 0);
}

function getWeeksCount(startDate: Date, endDate: Date) {
  const days = getNumberDays(startDate, endDate) + getEmptyDaysAtStart(startDate) + getEmptyDaysAtEnd(endDate);
  return Math.round(days / DAYS_IN_WEEK);
}

function getEndDayOfWeek(weekIndex: number, startDate: Date) {
  const startDayWithEmpty = new Date(convertToDate(startDate).getTime() + getEmptyDaysAtStart(startDate) * MILISECONDS_IN_DAY);
  return new Date(startDayWithEmpty.getTime() + weekIndex * DAYS_IN_WEEK * MILISECONDS_IN_DAY);
}

interface Props {
  dates: HeatMapDate[];
  startDate: Date;
  endDate: Date;
  classForValue?: (value: HeatMapDate) => string;
}

export default function HeatmapCalendar({ dates, startDate, endDate, classForValue }: Props) {
  const weeks = useMemo(() => getWeeksCount(startDate, endDate), [startDate, endDate]);
  const emptyDays = useMemo(() => getEmptyDaysAtStart(startDate), [startDate]);
  const totalDays = useMemo(() => getNumberDays(startDate, endDate), [startDate, endDate]);

  const width = (RECT_SIZE + GAP) * weeks - GAP + WIDTH_LABEL;
  const height = (RECT_SIZE + GAP) * DAYS_IN_WEEK - GAP + HEIGHT_LABEL;

  function generateCell(index: number): CellData {
    const adjustedIndex = index - emptyDays;
    const currentDate = new Date(startDate.getTime() + adjustedIndex * MILISECONDS_IN_DAY);
    const matched = dates.find((d) => convertToDate(d.date).getTime() === currentDate.getTime());
    const value = matched ? matched.value : null;
    return {
      date: currentDate,
      value,
      cssClass: classForValue ? classForValue({ date: currentDate, value }) : 'fill-value-0',
    };
  }

  function isOutOfRange(weekIdx: number, dayIdx: number) {
    const index = weekIdx * DAYS_IN_WEEK + dayIdx;
    return index < emptyDays || index > emptyDays + totalDays;
  }

  function getMonthLabel(weekIdx: number) {
    const d = getEndDayOfWeek(weekIdx, startDate);
    const day = d.getDate();
    if (day < 1 || day > 7) return null;
    return d.toLocaleDateString('en-US', { month: 'short' });
  }

  return (
    <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height}>
      {Array.from({ length: weeks }, (_, weekIdx) => {
        const label = getMonthLabel(weekIdx);
        return label ? (
          <text
            key={`month-${weekIdx}`}
            x={weekIdx * (RECT_SIZE + GAP) + WIDTH_LABEL}
            y={0}
            dominantBaseline="central"
            textAnchor="start"
            className="fill-current text-month"
            style={{ alignmentBaseline: 'text-before-edge' }}
          >
            {label}
          </text>
        ) : null;
      })}

      {Array.from({ length: weeks }, (_, weekIdx) =>
        Array.from({ length: DAYS_IN_WEEK }, (_, dayIdx) => {
          if (isOutOfRange(weekIdx, dayIdx)) return null;
          const cellIndex = weekIdx * DAYS_IN_WEEK + dayIdx;
          const data = generateCell(cellIndex);
          return (
            <rect
              key={`${weekIdx}-${dayIdx}`}
              x={weekIdx * (RECT_SIZE + GAP) + WIDTH_LABEL}
              y={dayIdx * (RECT_SIZE + GAP) + HEIGHT_LABEL}
              rx={2}
              ry={2}
              width={RECT_SIZE}
              height={RECT_SIZE}
              className={data.cssClass}
            />
          );
        })
      )}

      {daysLabels.map((label, i) =>
        label ? (
          <text
            key={`day-${i}`}
            x={0}
            y={i * (RECT_SIZE + GAP) + HEIGHT_LABEL}
            dominantBaseline="central"
            textAnchor="start"
            className="fill-current text-small"
            style={{ alignmentBaseline: 'text-before-edge' }}
          >
            {label}
          </text>
        ) : null
      )}
    </svg>
  );
}
