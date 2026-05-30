import { useState, useEffect } from 'react';
import { Button } from 'primereact/button';
import ToolBar from '../components/shared/ToolBar';
import BlogCard from '../components/Blog/BlogCard';
import { fetchBlogPosts } from '../services/api';
import type { BlogPost } from '../data/blogs';

export default function BlogPage() {
  const [posts, setPosts] = useState<BlogPost[]>([]);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const limit = 5;

  useEffect(() => {
    setLoading(true);
    fetchBlogPosts(page, limit).then((res) => {
      setPosts(res.data);
      setTotalPages(res.totalPages);
      setLoading(false);
    });
  }, [page]);

  return (
    <>
      <ToolBar />
      <div className="app-container">
        <h1 className="text-white text-3xl font-bold mb-4">Blog</h1>
        {loading ? (
          <p className="text-gray-400">Loading posts...</p>
        ) : (
          <>
            <div className="grid">
              {posts.map((post) => (
                <BlogCard key={post.id} post={post} />
              ))}
            </div>
            {totalPages > 1 && (
              <div className="flex align-items-center justify-content-center gap-3 mt-4">
                <Button
                  icon="pi pi-chevron-left"
                  label="Previous"
                  text
                  severity="secondary"
                  className="text-pink-500"
                  disabled={page <= 1}
                  onClick={() => setPage((p) => p - 1)}
                  style={{ outline: 'none', boxShadow: 'none' }}
                />
                <span className="text-gray-400">
                  Page {page} of {totalPages}
                </span>
                <Button
                  icon="pi pi-chevron-right"
                  label="Next"
                  iconPos="right"
                  text
                  severity="secondary"
                  className="text-pink-500"
                  disabled={page >= totalPages}
                  onClick={() => setPage((p) => p + 1)}
                  style={{ outline: 'none', boxShadow: 'none' }}
                />
              </div>
            )}
          </>
        )}
      </div>
    </>
  );
}
