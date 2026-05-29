import type { GitProfile } from '../models/types';
import HeroUserDetail from './HeroUserDetail';
import HeroUserSkills from './HeroUserSkills';

interface Props {
  profile: GitProfile | null;
}

export default function Hero({ profile }: Props) {
  return (
    <div className="hero-grid mt-4">
      <HeroUserDetail profile={profile} />
      <HeroUserSkills profile={profile} />
    </div>
  );
}
