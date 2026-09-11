import { Card, Row, Col, Space } from 'antd';
import LazyImage from '../shared/LazyImage';
import { getUserMainStats } from '../../services/stats';

export default function StatsSection() {
  return (
    <Row gutter={[16, 16]}>
      <Col xs={24}>
        <Space direction="vertical" style={{ width: '100%' }}>
          <Card className="glow-card" style={{ textAlign: 'center', borderRadius: 12, transition: 'box-shadow 0.3s' }}>
            <LazyImage src={getUserMainStats()} alt="Profile stats" />
          </Card>
        </Space>
      </Col>
    </Row>
  );
}