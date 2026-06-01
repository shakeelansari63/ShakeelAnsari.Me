import { Chip } from 'primereact/chip';
import { Button } from 'primereact/button';
import type { BlogPost } from '../../data/blogs';

interface Props {
  post: BlogPost;
  isLight?: boolean;
  stats: { views: number; likes: number; liked?: boolean };
  liking?: boolean;
  onLike?: () => void;
}

export default function ArticleHeader({ post, isLight, stats, liking, onLike }: Props) {
  return (
    <>
      <h1 className={`text-3xl font-bold mb-2 ${isLight ? 'text-pink-600' : 'text-pink-400'}`}>{post.title}</h1>
      <div className={`flex align-items-center justify-content-between mb-3 text-base ${isLight ? 'text-gray-500' : 'text-gray-400'}`}>
        <div className="flex align-items-center gap-2">
          <span>{post.date}</span>
          <span>&middot;</span>
          <span>{post.readTime}</span>
        </div>
        <div className="flex align-items-center gap-2">
          <span><i className="pi pi-eye mr-1" />{stats.views}</span>
          <Button
            text
            severity="secondary"
            className={`p-0 ${isLight ? '' : 'text-pink-500'}`}
            icon={stats.liked ? 'pi pi-heart-fill' : 'pi pi-heart'}
            label={String(stats.likes)}
            loading={liking}
            onClick={onLike}
            style={{ outline: 'none', boxShadow: 'none', color: isLight ? '#d53a9d' : undefined }}
          />
        </div>
      </div>
      <div className="flex gap-2 flex-wrap mb-4">
        {post.tags.map((tag) => (
          <Chip key={tag} label={tag} className="text-sm" />
        ))}
      </div>
    </>
  );
}
