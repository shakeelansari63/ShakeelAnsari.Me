import type { GitProject } from '../models/types';
import { getProjectCard } from '../services/stats';
import { userData } from '../services/data';

interface Props {
  projects: GitProject[];
}

export default function ProjectsSection({ projects }: Props) {
  const openProject = (name: string) => {
    window.open(`https://github.com/${userData.githubUser}/${name}`, '_blank');
  };

  return (
    <>
      <div className="grid">
        {projects.map((project) => (
          <div key={project.id} className="md:col-6 col-12">
            <div
              className="portfolio-card h-full w-full text-center cursor-pointer"
              onClick={() => project.name && openProject(project.name)}
            >
              <img
                src={project.name ? getProjectCard(project.name) : ''}
                className="max-w-full"
                alt={project.name ?? ''}
              />
            </div>
          </div>
        ))}
      </div>
      <div className="flex align-items-center justify-content-center mt-3">
        <a
          href={`https://github.com/${userData.githubUser}?tab=repositories`}
          target="_blank"
          className="badge-link"
        >
          See all Projects
        </a>
      </div>
    </>
  );
}
