import { getUserMainStats, getUserGithubStats, getUserGithubStars } from '../services/stats';

export default function StatsSection() {
  return (
    <>
      <div className="grid">
        <div className="col-12">
          <div className="portfolio-card w-full text-center">
            <img src={getUserMainStats()} className="max-w-full" alt="Profile stats" />
          </div>
        </div>
      </div>
      <div className="grid mt-2">
        <div className="md:col-6 col-12">
          <div className="portfolio-card w-full text-center">
            <img src={getUserGithubStats()} className="max-w-full" alt="GitHub stats" />
          </div>
        </div>
        <div className="md:col-6 col-12">
          <div className="portfolio-card w-full text-center">
            <img src={getUserGithubStars()} className="max-w-full" alt="GitHub stars" />
          </div>
        </div>
      </div>
    </>
  );
}
