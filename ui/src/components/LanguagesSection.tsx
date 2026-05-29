import { getTopLanguageByRepo, getTopLanguageByCommit, getTopLanguages } from '../services/stats';

export default function LanguagesSection() {
  return (
    <>
      <div className="grid">
        <div className="md:col-6 col-12">
          <div className="portfolio-card w-full text-center">
            <img src={getTopLanguageByRepo()} className="max-w-full" alt="Languages by repo" />
          </div>
        </div>
        <div className="md:col-6 col-12">
          <div className="portfolio-card w-full text-center">
            <img src={getTopLanguageByCommit()} className="max-w-full" alt="Languages by commit" />
          </div>
        </div>
      </div>
      <div className="grid mt-2">
        <div className="col-12">
          <div className="portfolio-card w-full text-center">
            <img src={getTopLanguages()} className="max-w-full" alt="Top languages" />
          </div>
        </div>
      </div>
    </>
  );
}
