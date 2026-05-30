import { useEffect } from 'react';

const SCROLL_THRESHOLD = 80;

export default function ScrollShrink() {
  useEffect(() => {
    const onScroll = () => {
      const scrolled = window.scrollY > SCROLL_THRESHOLD;
      document.documentElement.classList.toggle('scrolled', scrolled);
    };

    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  return null;
}
