import { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Card } from "primereact/card";
import { Button } from "primereact/button";
import { Skeleton } from "primereact/skeleton";
import ToolBar from "../components/shared/ToolBar";
import { fetchSubjectChapters } from "../services/api";
import type { LearnChapter } from "../models/types";

export default function SubjectPage() {
  const { subjectId } = useParams<{ subjectId: string }>();
  const navigate = useNavigate();
  const [chapters, setChapters] = useState<LearnChapter[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    document.title = "Learn — [{#SEO-NAME#}]";
    if (subjectId) {
      fetchSubjectChapters(subjectId)
        .then(setChapters)
        .finally(() => setLoading(false));
    }
  }, [subjectId]);

  const subjectTitle = subjectId
    ? subjectId
        .replace(/^\d+-/, "")
        .replace(/[-_]/g, " ")
        .replace(/\b\w/g, (c) => c.toUpperCase())
    : "";

  return (
    <>
      <ToolBar />
      <div className="app-container">
        <div className="mb-3">
          <Button
            icon="pi pi-arrow-left"
            label="Back"
            text
            severity="secondary"
            className="text-pink-500"
            onClick={() => navigate("/learn")}
            style={{ outline: "none", boxShadow: "none" }}
          />
        </div>

        <h1 className="text-white text-2xl font-bold mb-4">{subjectTitle}</h1>

        {loading ? (
          <div className="flex flex-column gap-2">
            {[1, 2, 3].map((i) => (
              <Skeleton key={i} width="100%" height="3rem" />
            ))}
          </div>
        ) : chapters.length === 0 ? (
          <div className="text-center mt-8">
            <i className="pi pi-inbox text-4xl text-gray-500 mb-3" />
            <p className="text-gray-400 m-0">No chapters yet.</p>
          </div>
        ) : (
          <div className="flex flex-column gap-2">
            {chapters.map((chapter) => (
              <Card
                key={chapter.id}
                className="cursor-pointer"
                onClick={() =>
                  navigate(`/learn/${subjectId}/${chapter.id}`)
                }
              >
                <div className="flex align-items-center gap-3">
                  <span className="text-sm text-gray-500">
                    {`Topic: ${parseInt(chapter.chapter_id.replace(/^ch(\d+).*$/, "$1"), 10)}`}
                  </span>
                  <span className="text-pink-400 font-bold">
                    {chapter.title}
                  </span>
                </div>
              </Card>
            ))}
          </div>
        )}
      </div>
    </>
  );
}
