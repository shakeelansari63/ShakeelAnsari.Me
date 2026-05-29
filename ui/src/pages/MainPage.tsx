import { useEffect, useState } from 'react';
import type { GitProfile, GitProject } from '../models/types';
import { fetchUserProfile, fetchUserProjects } from '../services/api';
import ToolBar from '../components/ToolBar';
import Hero from '../components/Hero';
import SectionTitle from '../components/SectionTitle';
import StatsSection from '../components/StatsSection';
import ProjectsSection from '../components/ProjectsSection';
import LanguagesSection from '../components/LanguagesSection';
import StreakSection from '../components/StreakSection';
import ActivitySection from '../components/ActivitySection';
import FooterSection from '../components/FooterSection';

export default function MainPage() {
  const [profile, setProfile] = useState<GitProfile | null>(null);
  const [projects, setProjects] = useState<GitProject[]>([]);

  useEffect(() => {
    fetchUserProfile().then(setProfile);
    fetchUserProjects().then(setProjects);
  }, []);

  return (
    <>
      <ToolBar />
      <div className="app-container">
        <Hero profile={profile} />

        <SectionTitle anchor="stats">GitHub Statistics</SectionTitle>
        <StatsSection />

        <SectionTitle anchor="projects">GitHub Projects</SectionTitle>
        <ProjectsSection projects={projects} />

        <SectionTitle anchor="languages">GitHub Languages</SectionTitle>
        <LanguagesSection />

        <SectionTitle anchor="contributions">Streaks &amp; Ranks</SectionTitle>
        <StreakSection />

        <SectionTitle anchor="activity">Activity Graph</SectionTitle>
        <ActivitySection />

        <SectionTitle hideTitle />
        <FooterSection />
      </div>
    </>
  );
}
