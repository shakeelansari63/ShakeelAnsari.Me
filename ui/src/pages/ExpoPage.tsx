import { Helmet } from 'react-helmet-async';
import { Row, Col } from 'antd';
import ToolBar from '../components/shared/ToolBar';
import PageFooter from '../components/shared/PageFooter';
import ExpoCard from '../components/Expo/ExpoCard';
import { userData } from '../data/profile';
import { seo } from '../data/seo';

export default function ExpoPage() {
  return (
    <>
      <Helmet>
        <title>{`Expo — ${seo.name}`}</title>
        <meta name="description" content={`Projects and experiments by ${seo.name}.`} />
        <meta property="og:title" content={`Expo — ${seo.name}`} />
        <meta property="og:description" content={`Projects and experiments by ${seo.name}.`} />
        <meta property="og:url" content={`https://${seo.domain}/expo`} />
      </Helmet>
      <ToolBar />
      <div className="app-container">
        <h1 className="text-3xl font-bold mb-6" style={{ color: 'var(--ant-color-text)' }}>Expo</h1>
        <Row gutter={[16, 16]}>
          {userData.expo.map(item => (
            <Col key={item.name} xs={24} md={12}>
              <ExpoCard item={item} />
            </Col>
          ))}
        </Row>
        <PageFooter />
      </div>
    </>
  );
}
