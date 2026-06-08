import { useNavigate } from 'react-router-dom';
import { Card } from 'primereact/card';

export default function AlsoSeeSection() {
  const navigate = useNavigate();

  const items = [
    {
      icon: 'pi-book',
      title: 'Blogs',
      description: 'Thoughts on data engineering, AI, and software development.',
      route: '/blog',
    },
    {
      icon: 'pi-star',
      title: 'Expo',
      description: 'Showcase of projects and experiments I have built.',
      route: '/expo',
    },
  ];

  return (
    <div className="grid">
      {items.map((item) => (
        <div key={item.route} className="md:col-6 col-12">
          <Card
            className="cursor-pointer h-full"
            onClick={() => navigate(item.route)}
          >
            <div className="flex flex-column align-items-center text-center gap-2">
              <i className={`pi ${item.icon} text-4xl text-pink-400`} />
              <span className="font-bold text-xl text-pink-400">{item.title}</span>
              <p className="text-sm text-blue-400 m-0" style={{ fontFamily: '"Space Grotesk", sans-serif' }}>
                {item.description}
              </p>
            </div>
          </Card>
        </div>
      ))}
    </div>
  );
}
