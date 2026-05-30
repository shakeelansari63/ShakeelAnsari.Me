import { Chip } from 'primereact/chip';
import type { BlogPost } from '../../data/blogs';

interface Props {
  post: BlogPost;
  isLight?: boolean;
}

export default function ArticleHeader({ post, isLight }: Props) {
  return (
    <>
      <h1 className={`text-3xl font-bold mb-2 ${isLight ? 'text-pink-600' : 'text-pink-400'}`}>{post.title}</h1>
      <div className={`flex align-items-center gap-2 text-sm mb-3 ${isLight ? 'text-gray-500' : 'text-gray-400'}`}>
        <span>{post.date}</span>
        <span>&middot;</span>
        <span>{post.readTime}</span>
      </div>
      <div className="flex gap-2 flex-wrap mb-4">
        {post.tags.map((tag) => (
          <Chip key={tag} label={tag} className="text-sm" />
        ))}
      </div>
    </>
  );
}
