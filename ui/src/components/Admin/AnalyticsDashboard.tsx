import { useState, useEffect } from 'react';
import { Select, Spin, message, Row, Col, Typography } from 'antd';
import type { AnalyticsData } from '../../models/types';
import { fetchAnalytics } from '../../services/api';
import SummaryCards from './SummaryCards';
import DailyBarChart from './DailyBarChart';
import CountryPieChart from './CountryPieChart';
import TopBlogsList from './TopBlogsList';
import { settings } from '../../data/settings';

interface Props {
  token: string;
}

export default function AnalyticsDashboard({ token }: Props) {
  const [data, setData] = useState<AnalyticsData | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedBlog, setSelectedBlog] = useState<string | null>(null);

  const blogOptions = data?.topBlogs.map(b => ({
    label: b.title,
    value: b.id,
  })) ?? [];

  const load = (blogId?: string) => {
    setLoading(true);
    fetchAnalytics(token, blogId)
      .then(result => {
        if (result) {
          setData(result);
        } else {
          message.error('Failed to load analytics');
        }
      })
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    load(selectedBlog ?? undefined);
  }, [selectedBlog]);

  return (
    <>
      <div style={{ maxWidth: 400, marginBottom: 16 }}>
        <Select
          value={selectedBlog}
          options={[{ label: 'All Blogs', value: '' }, ...blogOptions]}
          onChange={value => setSelectedBlog(value || null)}
          placeholder="All Blogs"
          style={{ width: '100%' }}
          allowClear
        />
      </div>
      {loading ? (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', margin: '32px auto', gap: 12 }}>
          <Spin size="large" />
          <Typography.Text type="secondary">Loading analytics...</Typography.Text>
        </div>
      ) : data ? (
        <div className="space-y-6">
          <SummaryCards
            totalViews={data.totalViews}
            totalLikes={data.totalLikes}
            uniqueVisitors={data.uniqueVisitors}
          />

          <DailyBarChart title="Daily Views" data={data.viewsByDate} color={settings.themeColor} />
          <DailyBarChart title="Daily Likes" data={data.likesByDate} color="#22c55e" />

          <Row gutter={[16, 16]}>
            <Col xs={24} md={12}>
              <CountryPieChart title="Views by Country" data={data.viewsByCountry} />
            </Col>
            <Col xs={24} md={12}>
              <CountryPieChart title="Likes by Country" data={data.likesByCountry} />
            </Col>
          </Row>

          <TopBlogsList blogs={data.topBlogs} />
        </div>
      ) : (
        <p style={{ color: 'var(--ant-color-text-tertiary)', textAlign: 'center', padding: 32 }}>
          Failed to load analytics.
        </p>
      )}
    </>
  );
}