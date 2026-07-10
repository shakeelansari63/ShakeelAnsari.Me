import { useEffect, useState } from "react";
import { Helmet } from "react-helmet-async";
import type { GitProfile } from "../models/types";
import { fetchUserProfile } from "../services/api";
import { seo } from "../data/seo";
import ToolBar from "../components/shared/ToolBar";
import Hero from "../components/Home/Hero";
import SkillsSection from "../components/Home/SkillsSection";
import WorkSection from "../components/Home/WorkSection";
import SectionTitle from "../components/Home/SectionTitle";
import AlsoSeeSection from "../components/Home/AlsoSeeSection";
import PageFooter from "../components/shared/PageFooter";

export default function MainPage() {
  const [profile, setProfile] = useState<GitProfile | null>(null);

  useEffect(() => {
    fetchUserProfile().then(setProfile);
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

        <SectionTitle anchor="skills">Skills &amp; Expertise</SectionTitle>
        <SkillsSection />

        <SectionTitle anchor="experience">Work Experience</SectionTitle>
        <WorkSection />

        <SectionTitle anchor="see-also">Explore Further</SectionTitle>
        <AlsoSeeSection />

        <SectionTitle hideTitle />
        <PageFooter />
      </div>
    </>
  );
}
