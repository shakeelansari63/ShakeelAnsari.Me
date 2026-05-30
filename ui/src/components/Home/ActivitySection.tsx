import { Card } from 'primereact/card';
import { useEffect, useState } from 'react';
import type { ContribSubject, HeatMapDate } from '../../models/types';
import { fetchUserContributions } from '../../services/api';
import HeatmapCalendar from './HeatmapCalendar';

function classForValue({ value }: HeatMapDate) {
  if (value === null || value === undefined) return 'fill-value-0';
  switch (value) {
    case 1: return 'fill-value-1';
    case 2: return 'fill-value-2';
    case 3: return 'fill-value-3';
    case 4: return 'fill-value-4';
    default: return 'fill-value-0';
  }
}

export default function ActivitySection() {
  const [contrib, setContrib] = useState<ContribSubject | null>(null);

  useEffect(() => {
    fetchUserContributions().then(setContrib);
  }, []);

  return (
    <Card className="text-center">
      {contrib ? (
        <HeatmapCalendar
          dates={contrib.data}
          startDate={contrib.startDate}
          endDate={contrib.endDate}
          classForValue={classForValue}
        />
      ) : (
        <p className="text-gray-400">Loading contribution data...</p>
      )}
    </Card>
  );
}
