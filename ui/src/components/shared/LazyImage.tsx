import { useState } from "react";

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
  aspectRatio = "2 / 1",
  rounded = false,
  className = "",
}: Props) {
  const [loaded, setLoaded] = useState(false);

  const borderRadius = rounded ? "50%" : "8px";

  return (
    <div
      className={`flex justify-content-center my-3 ${className}`}
      style={{
        position: "relative",
        ...(maxWidth !== undefined && { maxWidth: `${maxWidth}px` }),
        marginLeft: "auto",
        marginRight: "auto",
      }}
    >
      {!loaded && (
        <div
          className="skeleton-image"
          style={{
            width: "100%",
            aspectRatio,
            borderRadius,
          }}
        >
          <svg
            width="48"
            height="48"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="skeleton-image-icon"
          >
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
            <circle cx="8.5" cy="8.5" r="1.5" />
            <polyline points="21 15 16 10 5 21" />
          </svg>
        </div>
      )}
      <div
        style={{
          position: "relative",
          maxWidth: "100%",
          ...(maxHeight !== undefined && { maxHeight: `${maxHeight}px` }),
          overflow: "hidden",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          borderRadius,
        }}
      >
        <img
          src={src}
          alt={alt}
          loading="lazy"
          onLoad={() => setLoaded(true)}
          className={loaded ? "image-entrance" : ""}
          style={{
            maxWidth: "100%",
            ...(maxHeight !== undefined && { maxHeight: `${maxHeight}px` }),
            width: "auto",
            height: "auto",
            display: "block",
            opacity: loaded ? 1 : 0,
          }}
        />
      </div>
    </div>
  );
}
