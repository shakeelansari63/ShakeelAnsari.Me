export default function SkeletonArticle() {
  return (
    <div className="app-container">
      <div className="mb-3 flex align-items-center justify-content-between">
        <div className="h-6 w-16 skeleton-pulse" />
        <div className="h-6 w-6 skeleton-pulse" style={{ borderRadius: '50%' }} />
      </div>
      <div className="flex flex-column gap-4">
        <div className="h-10 w-5/6 skeleton-pulse" />
        <div className="flex align-items-center gap-3">
          <div className="h-4 w-24 skeleton-pulse" />
          <div className="h-4 w-16 skeleton-pulse" />
          <div className="h-4 w-12 skeleton-pulse" />
        </div>
        <div className="flex gap-2">
          <div className="h-6 w-16 skeleton-pulse" style={{ borderRadius: '1rem' }} />
          <div className="h-6 w-20 skeleton-pulse" style={{ borderRadius: '1rem' }} />
          <div className="h-6 w-14 skeleton-pulse" style={{ borderRadius: '1rem' }} />
        </div>
        <div className="flex flex-column gap-2 mt-4">
          <div className="h-4 w-full skeleton-pulse" />
          <div className="h-4 w-full skeleton-pulse" />
          <div className="h-4 w-5/6 skeleton-pulse" />
          <div className="h-4 w-full skeleton-pulse" />
          <div className="h-4 w-4/6 skeleton-pulse" />
          <div className="h-4 w-full skeleton-pulse" />
          <div className="h-4 w-3/4 skeleton-pulse" />
          <div className="h-4 w-full skeleton-pulse" />
          <div className="h-4 w-5/6 skeleton-pulse" />
        </div>
      </div>
    </div>
  );
}
