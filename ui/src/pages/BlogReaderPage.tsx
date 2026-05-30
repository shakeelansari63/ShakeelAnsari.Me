import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Button } from 'primereact/button';
import ToolBar from '../components/shared/ToolBar';
import ArticleHeader from '../components/BlogReader/ArticleHeader';
import ArticleContent from '../components/BlogReader/ArticleContent';
import { fetchBlogPost } from '../services/api';
import type { BlogPost } from '../data/blogs';

export default function BlogReaderPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [post, setPost] = useState<BlogPost | null>(null);
  const [loading, setLoading] = useState(true);
  const [isLight, setIsLight] = useState(false);

  useEffect(() => {
    if (!id) return;
    fetchBlogPost(id).then((data) => {
      setPost(data);
      setLoading(false);
    });
  }, [id]);

  if (loading) {
    return (
      <>
        <ToolBar />
        <div className="app-container">
          <p className="text-gray-400 mt-8">Loading post...</p>
        </div>
      </>
    );
  }

  if (!post) {
    return (
      <>
        <ToolBar />
        <div className="app-container">
          <div className="text-center mt-8">
            <h2 className="text-white">Post not found</h2>
            <Button label="Back to Blog" text severity="secondary" className="text-pink-500" onClick={() => navigate('/blog')} />
          </div>
        </div>
      </>
    );
  }

  return (
    <>
      <ToolBar />
      <div
        style={{
          minHeight: '100vh',
          background: isLight ? '#f5f5f5' : 'transparent',
          color: isLight ? '#1a1a2e' : 'inherit',
        }}
      >
        <div className="app-container">
          <div className="mb-3 flex align-items-center justify-content-between">
            <Button
              icon="pi pi-arrow-left"
              label="Back"
              text
              severity="secondary"
              className={isLight ? '' : 'text-pink-500'}
              onClick={() => navigate('/blog')}
              style={{ outline: 'none', boxShadow: 'none', color: isLight ? '#d53a9d' : undefined }}
            />
            <Button
              text
              severity="secondary"
              className={isLight ? '' : 'text-pink-500'}
              icon={isLight ? 'pi pi-moon' : 'pi pi-sun'}
              onClick={() => setIsLight(!isLight)}
              tooltip={isLight ? 'Dark Mode' : 'Light Mode'}
              tooltipOptions={{ position: 'bottom' }}
              style={{ outline: 'none', boxShadow: 'none', color: isLight ? '#d53a9d' : undefined }}
            />
          </div>
          <article>
            <ArticleHeader post={post} isLight={isLight} />
            <ArticleContent content={post.content} isLight={isLight} />
          </article>
        </div>
      </div>
    </>
  );
}
