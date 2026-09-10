import { useState } from 'react';
import { Skeleton } from 'antd';

interface Props {
  src: string;
  alt?: string;
  maxWidth?: number;
  maxHeight?: number;
  aspectRatio?: string;
  rounded?: boolean;
  className?: string;
}

export default function LazyImage({
  src,
  alt,
  maxWidth,
  maxHeight,
  aspectRatio = '2 / 1',
  rounded = false,
  className = '',
}: Props) {
  const [loaded, setLoaded] = useState(false);

  const borderRadius = rounded ? '50%' : '8px';

  return (
    <div
      className={`flex justify-center my-3 ${className}`}
      style={{
        position: 'relative',
        ...(maxWidth !== undefined && { maxWidth: `${maxWidth}px` }),
        marginLeft: 'auto',
        marginRight: 'auto',
      }}
    >
      {!loaded && (
        <Skeleton
          active
          avatar={{ size: 'large', shape: rounded ? 'circle' : 'square' }}
          style={{
            width: '100%',
            aspectRatio,
            borderRadius,
          }}
        />
      )}
      <div
        style={{
          position: 'relative',
          maxWidth: '100%',
          ...(maxHeight !== undefined && { maxHeight: `${maxHeight}px` }),
          overflow: 'hidden',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          borderRadius,
        }}
      >
        <img
          src={src}
          alt={alt}
          loading="lazy"
          onLoad={() => setLoaded(true)}
          className={loaded ? 'image-entrance' : ''}
          style={{
            maxWidth: '100%',
            ...(maxHeight !== undefined && { maxHeight: `${maxHeight}px` }),
            width: 'auto',
            height: 'auto',
            display: 'block',
            opacity: loaded ? 1 : 0,
          }}
        />
      </div>
    </div>
  );
}