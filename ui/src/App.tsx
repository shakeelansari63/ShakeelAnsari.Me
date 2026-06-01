import { Routes, Route } from 'react-router-dom';
import MainPage from './pages/MainPage';
import BlogPage from './pages/BlogPage';
import BlogReaderPage from './pages/BlogReaderPage';
import ExpoPage from './pages/ExpoPage';
import AdminPage from './pages/AdminPage';
import ScrollShrink from './ScrollShrink';
import ScrollToTop from './ScrollToTop';

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
      </Routes>
    </>
  );
}

export default App;
