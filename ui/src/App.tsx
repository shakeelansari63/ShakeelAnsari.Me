import { Suspense, lazy } from "react";
import { Routes, Route } from "react-router-dom";
import ScrollShrink from "./components/shared/ScrollShrink";
import ScrollToTop from "./components/shared/ScrollToTop";
import LoadingSpinner from "./components/shared/LoadingSpinner";
import { settings } from "./data/settings";

// Eager Load Main Page
import MainPage from "./pages/MainPage";

// Lazy Load other pages
const BlogPage = lazy(() => import("./pages/BlogPage"));
const BlogReaderPage = lazy(() => import("./pages/BlogReaderPage"));
const ExpoPage = lazy(() => import("./pages/ExpoPage"));
const LearnPage = lazy(() => import("./pages/LearnPage"));
const SubjectPage = lazy(() => import("./pages/SubjectPage"));
const ChapterReaderPage = lazy(() => import("./pages/ChapterReaderPage"));
const ProductPage = lazy(() => import("./pages/ProductPage"));
const PortfolioPage = lazy(() => import("./pages/PortfolioPage"));
const AdminPage = lazy(() => import("./pages/AdminPage"));
const NotFoundPage = lazy(() => import("./pages/NotFoundPage"));

function App() {
    return (
        <>
            <ScrollShrink />
            <ScrollToTop />
            <Suspense fallback={<LoadingSpinner />}>
                <Routes>
                    <Route path="/" element={<MainPage />} />
                    {settings.showBlogs && (
                        <>
                            <Route path="/blog" element={<BlogPage />} />
                            <Route path="/blog/:id" element={<BlogReaderPage />} />
                        </>
                    )}
                    {settings.showExpo && (
                        <>
                            <Route path="/expo" element={<ExpoPage />} />
                            <Route path="/product/:id" element={<ProductPage />} />
                        </>
                    )}
                    {settings.showTutorial && (
                        <>
                            <Route path="/learn" element={<LearnPage />} />
                            <Route path="/learn/:subjectId" element={<SubjectPage />} />
                            <Route
                                path="/learn/:subjectId/:chapterId"
                                element={<ChapterReaderPage />}
                            />
                        </>
                    )}
                    <Route path="/portfolio" element={<PortfolioPage />} />
                    <Route path="/admin" element={<AdminPage />} />
                    <Route path="*" element={<NotFoundPage />} />
                </Routes>
            </Suspense>
        </>
    );
}

export default App;
