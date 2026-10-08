import { Row, Col } from 'antd';
import ProfileStatsCard from '../Stats/ProfileStatsCard';

export default function StatsSection() {
  return (
    <Row gutter={[16, 16]}>
      <Col xs={24}>
        <ProfileStatsCard />
      </Col>
    </Row>
  );
}
