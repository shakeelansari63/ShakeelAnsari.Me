import { Card } from 'primereact/card';
import { userData } from '../../data/profile';
import type { WorkRole } from '../../data/work';

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
  return (
    <div className="grid">
      {userData.work.map((job, i) => {
        const range = companyDateRange(job.roles);
        return (
          <div key={i} className="col-12">
            <Card className="h-full">
              <div className="flex align-items-start gap-3">
                <div className="flex flex-column align-items-center" style={{ minWidth: '2px' }}>
                  <div
                    style={{
                      width: '14px',
                      height: '14px',
                      borderRadius: '50%',
                      background: 'linear-gradient(135deg, #d53a9d, #743ad5)',
                      marginTop: '6px',
                      flexShrink: 0,
                    }}
                  />
                  {i < userData.work.length - 1 && (
                    <div style={{ width: '2px', flex: 1, background: 'linear-gradient(to bottom, #d53a9d, #271250)', minHeight: '24px' }} />
                  )}
                </div>
                <div className="flex-1" style={{ marginTop: '-2px' }}>
                  <div className="flex flex-column md:flex-row md:align-items-center md:justify-content-between gap-1 mb-2">
                    <span className="font-bold text-lg text-pink-400">{job.company}</span>
                    <span className="text-sm" style={{ color: '#888' }}>
                      {range.start ? formatDate(range.start) : ''}
                      {range.start ? (range.end ? ` — ${formatDate(range.end)}` : ' — Present') : ''}
                    </span>
                  </div>
                  <div className="flex flex-column gap-3 ml-1">
                    {job.roles.map((role, j) => (
                      <div key={j}>
                        <div className="flex align-items-start gap-2">
                          <div
                            style={{
                              width: '8px',
                              height: '8px',
                              borderRadius: '50%',
                              background: '#d53a9d',
                              marginTop: '6px',
                              flexShrink: 0,
                            }}
                          />
                          <div className="flex-1">
                            <div className="flex flex-wrap align-items-center gap-2">
                              <span className="text-sm" style={{ color: '#ccc' }}>{role.title}</span>
                              {role.startDate && (
                                <span className="text-xs" style={{ color: '#666' }}>
                                  {formatDate(role.startDate)}
                                  {role.endDate ? ` — ${formatDate(role.endDate)}` : ' — Present'}
                                </span>
                              )}
                            </div>
                            {role.description && (
                              <p className="text-sm text-blue-400 mt-1 mb-0" style={{ fontFamily: '"Space Grotesk", sans-serif' }}>
                                {role.description}
                              </p>
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </Card>
          </div>
        );
      })}
    </div>
  );
}
