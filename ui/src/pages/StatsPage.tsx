import { useEffect, useState } from 'react';
import { Helmet } from 'react-helmet-async';
import type { GitProject } from '../models/types';
import { fetchUserProjects } from '../services/api';
import { seo } from '../data/seo';
import ToolBar from '../components/shared/ToolBar';
import SectionTitle from '../components/Home/SectionTitle';
import StatsSection from '../components/Home/StatsSection';
import ProjectsSection from '../components/Home/ProjectsSection';
import LanguagesSection from '../components/Home/LanguagesSection';
import StreakSection from '../components/Home/StreakSection';
import ActivitySection from '../components/Home/ActivitySection';
import PageFooter from '../components/shared/PageFooter';

export default function StatsPage() {
  const [projects, setProjects] = useState<GitProject[]>([]);

  useEffect(() => {
    fetchUserProjects().then(setProjects);
  }, []);

  return (
    <>
      <Helmet>
        <title>{`${seo.name} — Stats`}</title>
        <meta name="description" content={`GitHub Stats of ${seo.name}`} />
        <meta property="og:title" content={`${seo.name} — Stats`} />
        <meta property="og:description" content={`GitHub Stats of ${seo.name}`} />
        <meta property="og:url" content={`https://${seo.domain}/stats`} />
        <meta property="og:type" content="website" />
        <meta property="og:image" content={seo.avatarUrl} />
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:title" content={seo.name} />
        <meta name="twitter:description" content={`GitHub Stats of ${seo.name}`} />
      </Helmet>
      <ToolBar />
      <div className="app-container pb-4">
        <h1 className="text-3xl font-bold mb-6" style={{ color: 'var(--ant-color-text)' }}>GitHub Stats</h1>
        <div id="stats" className="mb-6 h-5" />
        <StatsSection />

        <SectionTitle anchor="languages">GitHub Languages</SectionTitle>
        <LanguagesSection />

        <SectionTitle anchor="contributions">Streaks & Productivity</SectionTitle>
        <StreakSection />

        <SectionTitle anchor="activity">Activity Graph</SectionTitle>
        <ActivitySection />

        <SectionTitle anchor="projects">GitHub Projects</SectionTitle>
        <ProjectsSection projects={projects} />

        <SectionTitle hideTitle />
        <PageFooter />
      </div>
    </>
  );
}