import { useEffect, useState } from "react";
import { Helmet } from "react-helmet-async";
import type { GitProject } from "../models/types";
import { fetchUserProjects } from "../services/api";
import { seo } from "../data/seo";
import ToolBar from "../components/shared/ToolBar";
import SectionTitle from "../components/Home/SectionTitle";
import StatsSection from "../components/Home/StatsSection";
import ProjectsSection from "../components/Home/ProjectsSection";
import LanguagesSection from "../components/Home/LanguagesSection";
import StreakSection from "../components/Home/StreakSection";
import ActivitySection from "../components/Home/ActivitySection";
import FooterSection from "../components/Home/FooterSection";

export default function PortfolioPage() {
  const [projects, setProjects] = useState<GitProject[]>([]);

  useEffect(() => {
    fetchUserProjects().then(setProjects);
  }, []);

  return (
    <>
      <Helmet>
        <title>{`${seo.name} — Portfolio`}</title>
        <meta name="description" content={`GitHub Portfolio of ${seo.name}`} />
        <meta property="og:title" content={`${seo.name} — Portfolio`} />
        <meta
          property="og:description"
          content={`GitHub Portfolio of ${seo.name}`}
        />
        <meta property="og:url" content={`https://${seo.domain}/portfolio`} />
        <meta property="og:type" content="website" />
        <meta property="og:image" content={seo.avatarUrl} />
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:title" content={seo.name} />
        <meta
          name="twitter:description"
          content={`GitHub Portfolio of ${seo.name}`}
        />
      </Helmet>
      <ToolBar />
      <div className="app-container">
        <div id="stats" className="mb-6 h-2rem"></div>
        <StatsSection />

        <SectionTitle anchor="languages">GitHub Languages</SectionTitle>
        <LanguagesSection />

        <SectionTitle anchor="contributions">
          Streaks &amp; Productivity
        </SectionTitle>
        <StreakSection />

        <SectionTitle anchor="activity">Activity Graph</SectionTitle>
        <ActivitySection />

        <SectionTitle anchor="projects">GitHub Projects</SectionTitle>
        <ProjectsSection projects={projects} />

        <SectionTitle hideTitle />
        <FooterSection />
      </div>
    </>
  );
}
