import { Routes, Route } from 'react-router-dom';
import MainPage from './pages/MainPage';
import BlogPage from './pages/BlogPage';
import BlogReaderPage from './pages/BlogReaderPage';
import ExpoPage from './pages/ExpoPage';
import AdminPage from './pages/AdminPage';
import NotFoundPage from './pages/NotFoundPage';
import ScrollShrink from './components/shared/ScrollShrink';
import ScrollToTop from './components/shared/ScrollToTop';

function App() {
  return (
    <>
      <ScrollShrink />
      <ScrollToTop />
      <Routes>
        <Route path="/" element={<MainPage />} />
        <Route path="/blog" element={<BlogPage />} />
        <Route path="/blog/:id" element={<BlogReaderPage />} />
        <Route path="/expo" element={<ExpoPage />} />
        <Route path="/admin" element={<AdminPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Routes>
    </>
  );
}

export default App;
