import { useEffect, useState } from "react";
import type { GitProfile, GitProject } from "../models/types";
import { fetchUserProfile, fetchUserProjects } from "../services/api";
import ToolBar from "../components/shared/ToolBar";
import Hero from "../components/Home/Hero";
import SectionTitle from "../components/Home/SectionTitle";
import StatsSection from "../components/Home/StatsSection";
import ProjectsSection from "../components/Home/ProjectsSection";
import LanguagesSection from "../components/Home/LanguagesSection";
import StreakSection from "../components/Home/StreakSection";
import ActivitySection from "../components/Home/ActivitySection";
import FooterSection from "../components/Home/FooterSection";

export default function MainPage() {
    useEffect(() => {
        document.title =
            "Shakeel Ansari — Data Engineer, GenAI Engineer & Full-Stack Developer";
    }, []);
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
