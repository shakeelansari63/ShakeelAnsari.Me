import { useState, useEffect } from "react";
import { Helmet } from "react-helmet-async";
import { useParams, useNavigate } from "react-router-dom";
import { Button } from "primereact/button";
import { Skeleton } from "primereact/skeleton";
import ToolBar from "../components/shared/ToolBar";
import MarkdownRenderer from "../components/shared/MarkdownRenderer";
import ReaderFooter from "../components/shared/ReaderFooter";
import { fetchChapterContent } from "../services/api";
import { seo } from "../data/seo";

export default function ChapterReaderPage() {
  const { subjectId, chapterId } = useParams<{
    subjectId: string;
    chapterId: string;
  }>();
  const navigate = useNavigate();
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [loading, setLoading] = useState(true);
  const [notFound, setNotFound] = useState(false);
  const [isLight, setIsLight] = useState(false);

  useEffect(() => {
    if (subjectId && chapterId) {
      fetchChapterContent(subjectId, chapterId)
        .then((data) => {
          if (data) {
            setTitle(data.title);
            setContent(data.content);
          } else {
            setNotFound(true);
          }
        })
        .finally(() => setLoading(false));
    }
  }, [subjectId, chapterId]);

  const metaTitle = title ? `${title} — Learn — ${seo.name}` : `Learn — ${seo.name}`;

  return (
    <>
      <Helmet>
        <title>{metaTitle}</title>
        <meta name="description" content={title ? `${title} — ${seo.name}` : seo.description} />
        <meta property="og:title" content={metaTitle} />
        <meta property="og:url" content={`https://${seo.domain}/learn/${subjectId}/${chapterId}`} />
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
              className="text-pink-500"
              onClick={() => navigate(`/learn/${subjectId}`)}
              style={{ outline: "none", boxShadow: "none" }}
            />
          </div>

          {loading ? (
            <div className="flex flex-column gap-3">
              <Skeleton width="60%" height="2rem" />
              <Skeleton width="100%" height="1rem" className="mb-2" />
              <Skeleton width="100%" height="1rem" className="mb-2" />
              <Skeleton width="75%" height="1rem" />
            </div>
          ) : notFound ? (
            <div className="text-center mt-8">
              <h2 className={isLight ? "text-gray-800" : "text-white"}>Chapter not found</h2>
              <Button
                label="Back to Subject"
                text
                severity="secondary"
                className="text-pink-500"
                onClick={() => navigate(`/learn/${subjectId}`)}
              />
            </div>
          ) : (
            <article>
              <h1 className="text-2xl md:text-3xl font-bold mb-6 text-pink-400">
                {title}
              </h1>
              <MarkdownRenderer content={content} isLight={isLight} />
              <ReaderFooter isLight={isLight} />
            </article>
          )}
        </div>
      </div>
    </>
  );
}
