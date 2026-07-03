import { useEffect, useState } from "react";
import { Helmet } from "react-helmet-async";
import type { GitProfile, GitProject } from "../models/types";
import { fetchUserProfile, fetchUserProjects } from "../services/api";
import { seo } from "../data/seo";
import ToolBar from "../components/shared/ToolBar";
import Hero from "../components/Home/Hero";
import WorkSection from "../components/Home/WorkSection";
import SectionTitle from "../components/Home/SectionTitle";
import StatsSection from "../components/Home/StatsSection";
import ProjectsSection from "../components/Home/ProjectsSection";
import AlsoSeeSection from "../components/Home/AlsoSeeSection";
import LanguagesSection from "../components/Home/LanguagesSection";
import StreakSection from "../components/Home/StreakSection";
import ActivitySection from "../components/Home/ActivitySection";
import FooterSection from "../components/Home/FooterSection";

export default function MainPage() {
  const [profile, setProfile] = useState<GitProfile | null>(null);
  const [projects, setProjects] = useState<GitProject[]>([]);

  useEffect(() => {
    fetchUserProfile().then(setProfile);
    fetchUserProjects().then(setProjects);
  }, []);

  return (
    <>
      <Helmet>
        <title>{`${seo.name} — ${seo.title}`}</title>
        <meta name="description" content={seo.description} />
        <meta property="og:title" content={`${seo.name} — ${seo.title}`} />
        <meta property="og:description" content={seo.description} />
        <meta property="og:url" content={`https://${seo.domain}/`} />
        <meta property="og:type" content="website" />
        <meta property="og:image" content={seo.avatarUrl} />
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:title" content={seo.name} />
        <meta name="twitter:description" content={seo.descriptionShort} />
      </Helmet>
      <ToolBar />
      <div className="app-container">
        <Hero profile={profile} />

        <SectionTitle anchor="experience">Work Experience</SectionTitle>
        <WorkSection />

        <SectionTitle anchor="stats">GitHub Statistics</SectionTitle>
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

        <SectionTitle anchor="see-also">Explore Further</SectionTitle>
        <AlsoSeeSection />

        <SectionTitle hideTitle />
        <FooterSection />
      </div>
    </>
  );
}
