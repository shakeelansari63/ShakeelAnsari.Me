import { Card, Skeleton } from 'antd';

export default function SkeletonCard() {
  return (
    <div style={{ width: '100%' }}>
      <Card style={{ height: '100%', borderRadius: 12 }}>
        <Skeleton active avatar={{ size: 'large', shape: 'square' }} title={{ width: '70%' }} paragraph={{ rows: 3, width: ['80%', '60%', '100%'] }} />
        <div className="mt-3 flex items-center justify-end gap-2" style={{ color: 'var(--ant-color-text-tertiary)' }}>
          <Skeleton active style={{ width: 60, height: 16, borderRadius: 4 }} />
          <Skeleton active style={{ width: 60, height: 16, borderRadius: 4 }} />
        </div>
      </Card>
    </div>
  );
}