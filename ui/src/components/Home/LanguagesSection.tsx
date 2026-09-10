import { Card, Row, Col, Space } from 'antd';
import LazyImage from '../shared/LazyImage';
import { getTopLanguageByRepo, getTopLanguageByCommit } from '../../services/stats';

export default function LanguagesSection() {
  return (
    <>
      <Row gutter={[16, 16]}>
        <Col xs={24} md={12}>
          <Space direction="vertical" style={{ width: '100%' }}>
            <Card style={{ textAlign: 'center', borderRadius: 12, height: '100%' }}>
              <LazyImage src={getTopLanguageByRepo()} alt="Languages by repo" />
            </Card>
          </Space>
        </Col>
        <Col xs={24} md={12}>
          <Space direction="vertical" style={{ width: '100%' }}>
            <Card style={{ textAlign: 'center', borderRadius: 12, height: '100%' }}>
              <LazyImage src={getTopLanguageByCommit()} alt="Languages by commit" />
            </Card>
          </Space>
        </Col>
      </Row>
    </>
  );
}