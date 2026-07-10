import { useState, useEffect } from "react";
import { Helmet } from "react-helmet-async";
import { useNavigate } from "react-router-dom";
import { Card } from "primereact/card";
import { Skeleton } from "primereact/skeleton";
import ToolBar from "../components/shared/ToolBar";
import PageFooter from "../components/shared/PageFooter";
import { fetchLearnSubjects } from "../services/api";
import type { LearnSubject } from "../models/types";
import { seo } from "../data/seo";

export default function LearnPage() {
  const navigate = useNavigate();
  const [subjects, setSubjects] = useState<LearnSubject[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchLearnSubjects()
      .then(setSubjects)
      .finally(() => setLoading(false));
  }, []);

  return (
    <>
      <Helmet>
        <title>{`Learn — ${seo.name}`}</title>
        <meta name="description" content={`Tutorials and learning resources by ${seo.name}.`} />
        <meta property="og:title" content={`Learn — ${seo.name}`} />
        <meta property="og:description" content={`Tutorials and learning resources by ${seo.name}.`} />
        <meta property="og:url" content={`https://${seo.domain}/learn`} />
      </Helmet>
      <ToolBar />
      <div className="app-container pb-4">
        <h1 className="text-white text-3xl font-bold mb-4 mt-3">Learn</h1>

        {loading ? (
          <div className="grid">
            {[1, 2, 3].map((i) => (
              <div key={i} className="md:col-6 col-12">
                <Card className="h-full">
                  <Skeleton width="100%" height="8rem" className="mb-2" />
                  <Skeleton width="60%" height="1.5rem" />
                </Card>
              </div>
            ))}
          </div>
        ) : subjects.length === 0 ? (
          <div className="text-center mt-8">
            <i className="pi pi-inbox text-4xl text-gray-500 mb-3" />
            <p className="text-gray-400 m-0">No subjects available yet.</p>
          </div>
        ) : (
          <div className="grid">
            {subjects.map((subject) => (
              <div key={subject.id} className="md:col-6 col-12">
                <Card
                  className="cursor-pointer h-full overflow-hidden"
                  onClick={() => navigate(`/learn/${subject.id}`)}
                  pt={{ body: { className: 'p-0' }, content: { className: 'p-0' } }}
                >
                  <img
                    src={`/api/learn/images/${subject.folder}/${subject.thumbnail}`}
                    alt={subject.title}
                    style={{
                      width: '100%',
                      aspectRatio: '16 / 9',
                      objectFit: 'contain',
                      display: 'block',
                      background: '#1a1a1a',
                    }}
                    loading="lazy"
                  />
                  <div className="px-3 pb-3 pt-2">
                    <span className="font-bold text-xl text-pink-400">
                      {subject.title}
                    </span>
                  </div>
                </Card>
              </div>
            ))}
          </div>
        )}
        <PageFooter />
      </div>
    </>
  );
}
