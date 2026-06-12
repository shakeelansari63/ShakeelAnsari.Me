import { useEffect } from 'react';
import ToolBar from '../components/shared/ToolBar';
import ExpoCard from '../components/Expo/ExpoCard';
import { userData } from '../data/profile';

export default function ExpoPage() {
  useEffect(() => { document.title = "Expo — Shakeel Ansari"; }, []);
  return (
    <>
      <ToolBar />
      <div className="app-container pb-4">
        <h1 className="text-white text-3xl font-bold mb-4">Expo</h1>
        <div className="grid">
          {userData.expo.map((item) => (
            <ExpoCard key={item.name} item={item} />
          ))}
        </div>
      </div>
    </>
  );
}
