import { Card } from 'primereact/card';

export default function SkeletonCard() {
  return (
    <div className="col-12">
      <Card>
        <div className="flex flex-column gap-3">
          <div className="h-6 w-3/4 skeleton-pulse" />
          <div className="h-4 w-1/3 skeleton-pulse" />
          <div className="h-4 w-full skeleton-pulse" />
          <div className="h-4 w-5/6 skeleton-pulse" />
          <div className="flex gap-2">
            <div className="h-6 w-16 skeleton-pulse" style={{ borderRadius: '1rem' }} />
            <div className="h-6 w-20 skeleton-pulse" style={{ borderRadius: '1rem' }} />
            <div className="h-6 w-14 skeleton-pulse" style={{ borderRadius: '1rem' }} />
          </div>
        </div>
      </Card>
    </div>
  );
}
