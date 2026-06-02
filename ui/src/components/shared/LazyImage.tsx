import { useState } from "react";

interface Props {
  src: string;
  alt?: string;
  maxWidth?: number;
  maxHeight?: number;
}

export default function LazyImage({
  src,
  alt,
  maxWidth = 800,
  maxHeight = 400,
}: Props) {
  const [loaded, setLoaded] = useState(false);

  return (
    <div
      className="flex justify-content-center my-3"
      style={{
        position: "relative",
        maxWidth: `${maxWidth}px`,
        marginLeft: "auto",
        marginRight: "auto",
      }}
    >
      {!loaded && (
        <div
          className="skeleton-pulse"
          style={{ width: "100%", height: "300px", borderRadius: "8px" }}
        />
      )}
      <div
        style={{
          maxWidth: "100%",
          maxHeight: `${maxHeight}px`,
          overflow: "hidden",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          borderRadius: "8px",
        }}
      >
        <img
          src={src}
          alt={alt}
          loading="lazy"
          onLoad={() => setLoaded(true)}
          style={{
            maxWidth: "100%",
            maxHeight: `${maxHeight}px`,
            width: "auto",
            height: "auto",
            display: "block",
            opacity: loaded ? 1 : 0,
            transition: "opacity 0.5s ease-in",
          }}
        />
      </div>
    </div>
  );
}
