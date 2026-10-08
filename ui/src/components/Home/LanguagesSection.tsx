import { Row, Col } from 'antd';
import ReposLanguageCard from '../Stats/ReposLanguageCard';
import CommitLanguageCard from '../Stats/CommitLanguageCard';

export default function LanguagesSection() {
  return (
    <Row gutter={[16, 16]}>
      <Col xs={24} md={12}>
        <ReposLanguageCard />
      </Col>
      <Col xs={24} md={12}>
        <CommitLanguageCard />
      </Col>
    </Row>
  );
}
