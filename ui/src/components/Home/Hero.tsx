import type { GitProfile } from '../../models/types';
import HeroUserDetail from './HeroUserDetail';
import HeroUserSkills from './HeroUserSkills';

interface Props {
  profile: GitProfile | null;
}

export default function Hero({ profile }: Props) {
  return (
    <div className="grid mt-4">
      <div className="md:col-6 col-12">
        <HeroUserDetail profile={profile} />
      </div>
      <div className="md:col-6 col-12">
        <HeroUserSkills profile={profile} />
      </div>
    </div>
  );
}
