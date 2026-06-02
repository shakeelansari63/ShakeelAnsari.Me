import { Chip } from 'primereact/chip';
import { Button } from 'primereact/button';
import type { BlogPost } from '../../models/types';

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
      <h1 className={`text-2xl md:text-3xl font-bold mb-2 ${isLight ? 'text-pink-600' : 'text-pink-400'}`}>{post.title}</h1>
      <div className={`flex flex-column sm:flex-row align-items-start sm:align-items-center justify-content-between mb-3 text-sm md:text-base gap-1 sm:gap-0 ${isLight ? 'text-gray-500' : 'text-gray-400'}`}>
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
          <Button
            text
            severity="secondary"
            className={`p-0 ${isLight ? '' : 'text-pink-500'}`}
            icon="pi pi-share-alt"
            onClick={() => {
              const url = window.location.href;
              if (navigator.share) {
                navigator.share({ title: post.title, url }).catch(() => {
                  navigator.clipboard.writeText(url);
                });
              } else {
                navigator.clipboard.writeText(url);
              }
            }}
            tooltip="Share"
            tooltipOptions={{ position: 'top' }}
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
