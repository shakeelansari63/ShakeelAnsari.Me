import { Routes, Route } from 'react-router-dom';
import MainPage from './pages/MainPage';
import BlogPage from './pages/BlogPage';
import BlogReaderPage from './pages/BlogReaderPage';

function App() {
  return (
    <Routes>
      <Route path="/" element={<MainPage />} />
      <Route path="/blog" element={<BlogPage />} />
      <Route path="/blog/:id" element={<BlogReaderPage />} />
    </Routes>
  );
}

export default App;
