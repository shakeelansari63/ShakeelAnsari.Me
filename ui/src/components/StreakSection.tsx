import { getStreaks, getProductiveTime, getLocalStarDevStats, getGlobalStarDevStats } from '../services/stats';

export default function StreakSection() {
  return (
    <>
      <div className="grid">
        <div className="md:col-6 col-12">
          <div className="portfolio-card w-full text-center">
            <img src={getStreaks()} className="max-w-full" alt="Streaks" />
          </div>
        </div>
        <div className="md:col-6 col-12">
          <div className="portfolio-card w-full text-center">
            <img src={getProductiveTime()} className="max-w-full" alt="Productive time" />
          </div>
        </div>
      </div>
      <div className="grid mt-2">
        <div className="md:col-6 col-12">
          <img src={getLocalStarDevStats()} className="max-w-full border-round" alt="Local rank" />
        </div>
        <div className="md:col-6 col-12">
          <img src={getGlobalStarDevStats()} className="max-w-full border-round" alt="Global rank" />
        </div>
      </div>
    </>
  );
}
