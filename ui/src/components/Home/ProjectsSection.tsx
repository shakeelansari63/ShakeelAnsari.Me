import { Card } from 'primereact/card';
import { Button } from 'primereact/button';
import type { GitProject } from '../../models/types';
import { userData } from '../../services/data';

interface Props {
  projects: GitProject[];
}

export default function ProjectsSection({ projects }: Props) {
  const openProject = (name: string) => {
    window.open(
      `https://github.com/${userData.githubUser}/${name}`,
      '_blank',
    );
  };

  return (
    <>
      <div className="grid">
        {projects.map((project) => (
          <div key={project.id} className="md:col-6 col-12">
            <Card
              className="cursor-pointer h-full"
              onClick={() => project.name && openProject(project.name)}
            >
              <div className="flex align-items-center mb-2">
                <i className="pi pi-book mr-2 text-pink-400" />
                <span className="font-bold text-lg text-pink-400">{project.name ?? ''}</span>
              </div>
              {project.description && (
                <p className="text-sm text-blue-400 m-0" style={{ fontFamily: 'SpaceMono, monospace' }}>{project.description}</p>
              )}
            </Card>
          </div>
        ))}
      </div>
      <div className="flex align-items-center justify-content-center mt-3">
        <Button
          label="See all Projects"
          icon="pi pi-external-link"
          rounded
          className="border-gradient-purple text-white"
          onClick={() => window.open(
            `https://github.com/${userData.githubUser}?tab=repositories`,
            '_blank',
          )}
          style={{ outline: 'none', boxShadow: 'none' }}
        />
      </div>
    </>
  );
}
