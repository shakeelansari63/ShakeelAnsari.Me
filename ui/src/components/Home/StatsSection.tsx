import { Card, Row, Col, Space } from 'antd';
import LazyImage from '../shared/LazyImage';
import { getUserMainStats } from '../../services/stats';

export default function StatsSection() {
  return (
    <Row gutter={[16, 16]}>
      <Col xs={24}>
        <Space direction="vertical" style={{ width: '100%' }}>
          <Card style={{ textAlign: 'center', borderRadius: 12 }}>
            <LazyImage src={getUserMainStats()} alt="Profile stats" />
          </Card>
        </Space>
      </Col>
    </Row>
  );
}