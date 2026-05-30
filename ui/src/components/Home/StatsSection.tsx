import { Card } from 'primereact/card';
import { getUserMainStats } from '../../services/stats';

export default function StatsSection() {
  return (
    <div className="grid">
      <div className="col-12">
        <Card className="text-center">
          <img
            src={getUserMainStats()}
            className="max-w-full"
            alt="Profile stats"
          />
        </Card>
      </div>
    </div>
  );
}
