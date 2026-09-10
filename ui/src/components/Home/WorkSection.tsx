import { Card, Timeline, Typography } from 'antd';
import { userData } from '../../data/profile';
import type { WorkRole } from '../../data/work';
import { settings } from '../../data/settings';

const monthNames = [
  'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
];

function formatDate(ym: string) {
  const [y, m] = ym.split('-');
  return `${monthNames[+m - 1]} ${y}`;
}

function companyDateRange(roles: WorkRole[]) {
  const starts = roles.map(r => r.startDate).filter(Boolean) as string[];
  const hasCurrent = roles.some(r => r.endDate === null);
  const ends = roles.map(r => r.endDate).filter(Boolean) as string[];
  const start = starts.length ? starts.sort()[0] : null;
  const end = hasCurrent ? null : (ends.length ? ends.sort().reverse()[0] : null);
  return { start, end };
}

export default function WorkSection() {
  const { Text } = Typography;

  const items = userData.work.map((job) => {
    const range = companyDateRange(job.roles);
    return {
      label: range.start ? formatDate(range.start) + (range.end ? ` — ${formatDate(range.end)}` : ' — Present') : '',
      dot: (
        <div
          style={{
            width: 14,
            height: 14,
            borderRadius: '50%',
            background: `linear-gradient(135deg, ${settings.themeColor}, #743ad5)`,
            boxShadow: '0 0 0 3px var(--ant-color-bg-container)',
          }}
        />
      ),
      children: (
        <Card style={{ borderRadius: 10, width: '100%' }}>
          <Typography.Title level={5} style={{ color: settings.themeColor, marginBottom: 8 }}>
            {job.company}
          </Typography.Title>
          <div style={{ marginLeft: 8 }}>
            {job.roles.map((role, j) => (
              <div key={j} style={{ marginBottom: j < job.roles.length - 1 ? 16 : 0 }}>
                <div className="flex items-start gap-2">
                  <div
                    style={{
                      width: 8,
                      height: 8,
                      borderRadius: '50%',
                      background: settings.themeColor,
                      marginTop: 6,
                      flexShrink: 0,
                    }}
                  />
                  <div className="flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <Text type="secondary" style={{ fontSize: '0.875rem', fontWeight: 500, color: settings.themeColor }}>
                        {role.title}
                      </Text>
                      {role.startDate && (
                        <Text type="secondary" style={{ fontSize: '0.75rem', color: 'var(--ant-color-text-tertiary)' }}>
                          {formatDate(role.startDate)}
                          {role.endDate ? ` — ${formatDate(role.endDate)}` : ' — Present'}
                        </Text>
                      )}
                    </div>
                    {role.description && (
                      <Text
                        type="secondary"
                        style={{
                          marginTop: 8,
                          fontFamily: "'Space Grotesk', sans-serif",
                          fontSize: '0.875rem',
                        }}
                      >
                        {role.description}
                      </Text>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </Card>
      ),
    };
  });

  return (
    <Timeline mode="left" pending={false} items={items} style={{ maxWidth: '100%' }} />
  );
}