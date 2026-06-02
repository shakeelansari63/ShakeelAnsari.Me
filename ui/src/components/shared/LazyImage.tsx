import { useState } from 'react';

interface Props {
  src: string;
  alt?: string;
  maxWidth?: number;
  maxHeight?: number;
}

export default function LazyImage({ src, alt, maxWidth = 800, maxHeight = 400 }: Props) {
  const [loaded, setLoaded] = useState(false);

  return (
    <div className="flex justify-content-center my-3" style={{ position: 'relative' }}>
      {!loaded && (
        <div
          className="skeleton-pulse"
          style={{ width: '100%', maxWidth: `${maxWidth}px`, height: '300px', borderRadius: '8px' }}
        />
      )}
      <div
        style={{
          maxWidth: `${maxWidth}px`,
          maxHeight: `${maxHeight}px`,
          width: '100%',
          display: loaded ? 'flex' : 'none',
          alignItems: 'center',
          justifyContent: 'center',
          overflow: 'hidden',
        }}
      >
        <img
          src={src}
          alt={alt}
          loading="lazy"
          onLoad={() => setLoaded(true)}
          style={{ maxWidth: '100%', maxHeight: `${maxHeight}px`, width: 'auto', height: 'auto', borderRadius: '8px' }}
        />
      </div>
    </div>
  );
}
