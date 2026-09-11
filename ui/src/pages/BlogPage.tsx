import { useState, useEffect } from 'react';
import { Helmet } from 'react-helmet-async';
import { Pagination } from 'antd';
import ToolBar from '../components/shared/ToolBar';
import PageFooter from '../components/shared/PageFooter';
import BlogCard from '../components/Blog/BlogCard';
import SkeletonCard from '../components/Blog/SkeletonCard';
import { fetchBlogPosts } from '../services/api';
import type { BlogPost } from '../models/types';
import { seo } from '../data/seo';

export default function BlogPage() {
  const [posts, setPosts] = useState<BlogPost[]>([]);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const limit = 6;

  useEffect(() => {
    setLoading(true);
    fetchBlogPosts(page, limit).then(res => {
      setPosts(res.data);
      setTotalPages(res.totalPages);
      setLoading(false);
    });
  }, [page]);

  return (
    <>
      <Helmet>
        <title>{`Blogs — ${seo.name}`}</title>
        <meta name="description" content={`Read blogs by ${seo.name} on data engineering, AI, and software development.`} />
        <meta property="og:title" content={`Blogs — ${seo.name}`} />
        <meta property="og:description" content={`Read blogs by ${seo.name} on data engineering, AI, and software development.`} />
        <meta property="og:url" content={`https://${seo.domain}/blog`} />
      </Helmet>
      <ToolBar />
      <div className="app-container">
        <h1 className="text-3xl font-bold mb-6" style={{ color: 'var(--ant-color-text)' }}>Blogs</h1>
        {loading ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '24px' }}>
            <SkeletonCard />
            <SkeletonCard />
            <SkeletonCard />
          </div>
        ) : (
          <>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: '24px', marginBottom: 32 }}>
              {posts.map(post => (
                <BlogCard key={post.id} post={post} />
              ))}
            </div>
            {totalPages > 1 && (
              <div className="mt-8 flex justify-center" style={{ display: 'flex', justifyContent: 'center', marginTop: 8 }}>
                <Pagination
                  current={page}
                  total={totalPages * limit}
                  pageSize={limit}
                  showSizeChanger={false}
                  showQuickJumper={true}
                  onChange={setPage}
                  itemRender={(current, type, original) => {
                    if (type === 'page') return current;
                    if (type === 'prev') return 'Previous';
                    if (type === 'next') return 'Next';
                    return original;
                  }}
                />
              </div>
            )}
          </>
        )}
        <PageFooter />
      </div>
    </>
  );
}
