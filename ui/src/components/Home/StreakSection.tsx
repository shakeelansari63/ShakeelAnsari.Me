import { Row, Col } from 'antd';
import ProductiveTimeCard from '../Stats/ProductiveTimeCard';
import StreakStatsCard from '../Stats/StreakStatsCard';

export default function StreakSection() {
  return (
    <Row gutter={[16, 16]}>
      <Col xs={24} md={12}>
        <StreakStatsCard />
      </Col>
      <Col xs={24} md={12}>
        <ProductiveTimeCard />
      </Col>
    </Row>
  );
}
