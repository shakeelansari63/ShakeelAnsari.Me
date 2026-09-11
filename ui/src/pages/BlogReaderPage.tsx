import { useState, useEffect } from 'react';
import { Helmet } from 'react-helmet-async';
import { useParams, useNavigate } from 'react-router-dom';
import { Button, Skeleton, Spin } from 'antd';
import { LeftOutlined } from '@ant-design/icons';
import ToolBar from '../components/shared/ToolBar';
import ArticleHeader from '../components/BlogReader/ArticleHeader';
import ArticleContent from '../components/BlogReader/ArticleContent';
import AlsoReadSection from '../components/BlogReader/AlsoReadSection';
import PageFooter from '../components/shared/PageFooter';
import {
  fetchBlogPost,
  fetchBlogContent,
  fetchBlogStats,
  fetchRelatedBlogPosts,
  recordBlogView,
  likeBlog,
} from '../services/api';
import type { BlogPost } from '../models/types';
import type { BlogStats } from '../services/api';
import { seo } from '../data/seo';

export default function BlogReaderPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [post, setPost] = useState<BlogPost | null>(null);
  const [loading, setLoading] = useState(true);
  const [content, setContent] = useState<string | null>(null);
  const [contentLoading, setContentLoading] = useState(true);
  const [stats, setStats] = useState<BlogStats>({ views: 0, likes: 0 });
  const [liking, setLiking] = useState(false);
  const [related, setRelated] = useState<BlogPost[]>([]);

  useEffect(() => {
    if (!id) return;
    fetchBlogPost(id).then(data => {
      setPost(data);
      setLoading(false);
    });
    fetchBlogContent(id)
      .then(setContent)
      .finally(() => setContentLoading(false));
    fetchBlogStats(id).then(setStats);
    fetchRelatedBlogPosts(id).then(setRelated);
    recordBlogView(id).then(res => {
      if (res?.views !== undefined) {
        setStats(s => ({ ...s, views: res.views }));
      } else {
        fetchBlogStats(id).then(setStats);
      }
    });
  }, [id]);

  const metaTitle = post ? `${post.title} — ${seo.name}` : `Blog — ${seo.name}`;
  const metaDesc = post?.excerpt || seo.description;

  return (
    <>
      <Helmet>
        <title>{metaTitle}</title>
        <meta name="description" content={metaDesc} />
        <meta property="og:title" content={metaTitle} />
        <meta property="og:description" content={metaDesc} />
        <meta property="og:url" content={`https://${seo.domain}/blog/${id}`} />
        <meta property="og:type" content="article" />
        <meta name="twitter:title" content={metaTitle} />
        <meta name="twitter:description" content={metaDesc} />
      </Helmet>
      <ToolBar />
      <div className="app-container">
        <div className="mb-6">
          <Button
            type="text"
            icon={<LeftOutlined />}
            onClick={() => navigate('/blog')}
            style={{ color: 'var(--ant-color-primary)' }}
          >
            Back to Blogs
          </Button>
        </div>
        {loading ? (
          <div className="space-y-4">
            <Skeleton active avatar paragraph={{ rows: 4 }} />
          </div>
        ) : !post ? (
          <div className="text-center py-12">
            <h2 style={{ color: 'var(--ant-color-text)' }}>Post not found</h2>
            <Button
              type="text"
              onClick={() => navigate('/blog')}
              style={{ marginTop: 16, color: 'var(--ant-color-primary)' }}
            >
              Back to Blog
            </Button>
          </div>
        ) : (
          <article>
            <ArticleHeader
              post={post}
              stats={stats}
              liking={liking}
              onLike={async () => {
                setLiking(true);
                const res = await likeBlog(id!);
                if (res) {
                  setStats(s => ({ ...s, likes: res.likes, liked: res.liked }));
                }
                setLiking(false);
              }}
            />
            {contentLoading ? (
              <div className="flex justify-center my-8">
                <Spin size="large" />
              </div>
            ) : content ? (
              <ArticleContent content={content} />
            ) : null}
            {post && <AlsoReadSection posts={related} />}
            {post && <PageFooter />}
          </article>
        )}
      </div>
    </>
  );
}