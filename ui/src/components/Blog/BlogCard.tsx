import { useNavigate } from 'react-router-dom';
import { Card } from 'primereact/card';
import { Chip } from 'primereact/chip';
import type { BlogPost } from '../../models/types';

interface Props {
  post: BlogPost;
}

export default function BlogCard({ post }: Props) {
  const navigate = useNavigate();

  const footer = (
    <div className="flex justify-content-end gap-3 text-sm text-gray-400">
      <span className="flex align-items-center gap-1"><i className="pi pi-eye" />{post.views}</span>
      <span className="flex align-items-center gap-1"><i className="pi pi-heart" />{post.likes}</span>
    </div>
  );

  return (
    <div className="col-12">
      <Card className="cursor-pointer" footer={footer} onClick={() => navigate(`/blog/${post.id}`)}>
        <div className="flex flex-column">
          <span className="font-bold text-xl text-pink-400 mb-2">{post.title}</span>
          <span className="text-sm text-gray-400 mb-2">{post.date} &middot; {post.readTime}</span>
          <p className="text-blue-400 m-0 mb-2" style={{ fontFamily: '"Space Grotesk", sans-serif' }}>{post.excerpt}</p>
          <div className="flex gap-2 flex-wrap">
            {post.tags.map((tag) => (
              <Chip key={tag} label={tag} className="text-sm" />
            ))}
          </div>
        </div>
      </Card>
    </div>
  );
}
