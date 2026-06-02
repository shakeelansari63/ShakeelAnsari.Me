import { Card } from 'primereact/card';
import { Skeleton } from 'primereact/skeleton';

export default function SkeletonCard() {
  return (
    <div className="col-12">
      <Card>
        <div className="flex flex-column">
          <Skeleton width="75%" height="1.5rem" className="mb-2" />
          <Skeleton width="33%" height="0.875rem" className="mb-2" />
          <Skeleton width="100%" height="0.875rem" className="mb-1" />
          <Skeleton width="85%" height="0.875rem" className="mb-3" />
          <div className="flex gap-2">
            <Skeleton width="4rem" height="1.5rem" borderRadius="16px" />
            <Skeleton width="5rem" height="1.5rem" borderRadius="16px" />
            <Skeleton width="3.5rem" height="1.5rem" borderRadius="16px" />
          </div>
        </div>
      </Card>
    </div>
  );
}
