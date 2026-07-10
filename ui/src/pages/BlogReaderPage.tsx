import { useState, useEffect } from "react";
import { Helmet } from "react-helmet-async";
import { useParams, useNavigate } from "react-router-dom";
import { Button } from "primereact/button";
import { Skeleton } from "primereact/skeleton";
import ToolBar from "../components/shared/ToolBar";
import ArticleHeader from "../components/BlogReader/ArticleHeader";
import ArticleContent from "../components/BlogReader/ArticleContent";
import PageFooter from "../components/shared/PageFooter";
import {
    fetchBlogPost,
    fetchBlogContent,
    fetchBlogStats,
    recordBlogView,
    likeBlog,
} from "../services/api";
import type { BlogPost } from "../models/types";
import type { BlogStats } from "../services/api";
import { seo } from "../data/seo";
import LoadingSpinner from "../components/shared/LoadingSpinner";

export default function BlogReaderPage() {
    const { id } = useParams();
    const navigate = useNavigate();
    const [post, setPost] = useState<BlogPost | null>(null);
    const [loading, setLoading] = useState(true);
    const [content, setContent] = useState<string | null>(null);
    const [contentLoading, setContentLoading] = useState(true);
    const [isLight, setIsLight] = useState(false);
    const [stats, setStats] = useState<BlogStats>({ views: 0, likes: 0 });
    const [liking, setLiking] = useState(false);

    useEffect(() => {
        if (!id) return;
        fetchBlogPost(id).then((data) => {
            setPost(data);
            setLoading(false);
        });
        fetchBlogContent(id)
            .then(setContent)
            .finally(() => setContentLoading(false));
        fetchBlogStats(id).then(setStats);
        recordBlogView(id).then((res) => {
            if (res?.views !== undefined) {
                setStats((s) => ({ ...s, views: res.views }));
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
            <ToolBar
                isLight={isLight}
                onToggleTheme={() => setIsLight((p) => !p)}
            />
            <div
                style={{
                    minHeight: "100vh",
                    background: isLight ? "#f5f5f5" : "transparent",
                    color: isLight ? "#1a1a2e" : "inherit",
                }}
            >
                <div className="app-container">
                    <div className="mb-3">
                        <Button
                            icon="pi pi-arrow-left"
                            label="Back"
                            text
                            severity="secondary"
                            className={isLight ? "" : "text-pink-500"}
                            onClick={() => navigate("/blog")}
                            style={{
                                outline: "none",
                                boxShadow: "none",
                                color: isLight ? "#d53a9d" : undefined,
                            }}
                            pt={{
                                root: { style: { background: "transparent" } },
                            }}
                        />
                    </div>
                    {loading ? (
                        <div className="flex flex-column gap-3">
                            <Skeleton width="60%" height="2rem" />
                            <Skeleton width="30%" height="1rem" />
                            <div className="flex gap-2">
                                <Skeleton
                                    width="4rem"
                                    height="1.5rem"
                                    borderRadius="16px"
                                />
                                <Skeleton
                                    width="5rem"
                                    height="1.5rem"
                                    borderRadius="16px"
                                />
                            </div>
                            <div className="mt-4">
                                <Skeleton
                                    width="100%"
                                    height="1rem"
                                    className="mb-2"
                                />
                                <Skeleton
                                    width="100%"
                                    height="1rem"
                                    className="mb-2"
                                />
                                <Skeleton width="75%" height="1rem" />
                            </div>
                        </div>
                    ) : !post ? (
                        <div className="text-center mt-8">
                            <h2 className="text-white">Post not found</h2>
                            <Button
                                label="Back to Blog"
                                text
                                severity="secondary"
                                className="text-pink-500"
                                onClick={() => navigate("/blog")}
                            />
                        </div>
                    ) : (
                        <article>
                            <ArticleHeader
                                post={post}
                                isLight={isLight}
                                stats={stats}
                                liking={liking}
                                onLike={async () => {
                                    setLiking(true);
                                    const res = await likeBlog(id!);
                                    if (res) {
                                        setStats((s) => ({
                                            ...s,
                                            likes: res.likes,
                                            liked: res.liked,
                                        }));
                                    }
                                    setLiking(false);
                                }}
                            />
                            {contentLoading ? (
                                <div className="flex justify-content-center mt-4">
                                    <LoadingSpinner />
                                </div>
                            ) : content ? (
                                <ArticleContent
                                    content={content}
                                    isLight={isLight}
                                />
                            ) : null}
                            {post && <PageFooter isLight={isLight} />}
                        </article>
                    )}
                </div>
            </div>
        </>
    );
}
