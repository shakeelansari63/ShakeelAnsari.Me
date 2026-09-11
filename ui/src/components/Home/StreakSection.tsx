import { Card, Row, Col, Space } from 'antd';
import LazyImage from '../shared/LazyImage';
import { getStreaks, getProductiveTime } from '../../services/stats';

export default function StreakSection() {
  return (
    <Row gutter={[16, 16]}>
      <Col xs={24} md={12}>
        <Space direction="vertical" style={{ width: '100%' }}>
            <Card className="glow-card" style={{ textAlign: 'center', borderRadius: 12, height: '100%', transition: 'box-shadow 0.3s' }}>
              <LazyImage src={getStreaks()} alt="Streaks" />
          </Card>
        </Space>
      </Col>
      <Col xs={24} md={12}>
        <Space direction="vertical" style={{ width: '100%' }}>
            <Card className="glow-card" style={{ textAlign: 'center', borderRadius: 12, height: '100%', transition: 'box-shadow 0.3s' }}>
              <LazyImage src={getProductiveTime()} alt="Productive time" />
          </Card>
        </Space>
      </Col>
    </Row>
  );
}