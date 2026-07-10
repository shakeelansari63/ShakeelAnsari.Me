import { Helmet } from 'react-helmet-async';
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
      <div className="app-container pb-4">
        <h1 className="text-white text-3xl font-bold mb-4">Expo</h1>
        <div className="grid">
          {userData.expo.map((item) => (
            <ExpoCard key={item.name} item={item} />
          ))}
        </div>
        <PageFooter />
      </div>
    </>
  );
}
