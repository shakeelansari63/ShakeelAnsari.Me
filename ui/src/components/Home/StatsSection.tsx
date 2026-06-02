import { Card } from 'primereact/card';
import LazyImage from '../shared/LazyImage';
import { getUserMainStats } from '../../services/stats';

export default function StatsSection() {
  return (
    <div className="grid">
      <div className="col-12">
        <Card className="text-center">
          <LazyImage
            src={getUserMainStats()}
            alt="Profile stats"
          />
        </Card>
      </div>
    </div>
  );
}
