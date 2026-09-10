import type { GitProfile } from '../../models/types';
import { Row, Col } from 'antd';
import HeroUserDetail from './HeroUserDetail';
import HeroUserSkills from './HeroUserSkills';

interface Props {
  profile: GitProfile | null;
}

export default function Hero({ profile }: Props) {
  return (
    <Row gutter={[24, 24]} style={{ marginTop: 16 }}>
      <Col xs={24} md={12}>
        <HeroUserDetail profile={profile} />
      </Col>
      <Col xs={24} md={12}>
        <HeroUserSkills profile={profile} />
      </Col>
    </Row>
  );
}