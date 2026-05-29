import type { GitProfile } from '../models/types';
import { userData } from '../services/data';

interface Props {
  profile: GitProfile | null;
}

export default function HeroUserDetail({ profile }: Props) {
  return (
    <div className="portfolio-card h-full">
      <div className="flex flex-column align-items-center p-3">
        <div className="mb-2">
          <img
            src={profile?.avatar_url ?? ''}
            alt="Avatar"
            className="avatar-img"
            width={150}
            height={150}
          />
        </div>
        <div className="flex align-items-center justify-content-center mb-2 text-center">
          <span className="text-orange-300">{profile?.bio ?? ''}</span>
        </div>
      </div>
      <div className="card-footer">
        <div className="flex align-items-center justify-content-center flex-wrap gap-2">
          <a href={userData.github} target="_blank" className="social-link" title="GitHub">GH</a>
          <a href={userData.leetcode} target="_blank" className="social-link" title="LeetCode">LC</a>
          <a href={userData.linkedIn} target="_blank" className="social-link" title="LinkedIn">LI</a>
          <a href={userData.facebook} target="_blank" className="social-link" title="Facebook">FB</a>
          <a href={userData.twitter} target="_blank" className="social-link" title="Twitter">TW</a>
        </div>
        <div className="flex align-items-center justify-content-center mt-2">
          <a
            href={userData.badges}
            target="_blank"
            className="badge-link"
          >
            Check my badges
          </a>
        </div>
      </div>
    </div>
  );
}
