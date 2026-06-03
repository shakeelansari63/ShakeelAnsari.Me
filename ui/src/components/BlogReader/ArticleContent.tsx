import ReactMarkdown from "react-markdown";
import CodeBlock from "./CodeBlock";
import LazyImage from "../shared/LazyImage";

interface Props {
  content: string;
  isLight?: boolean;
}

export default function ArticleContent({ content, isLight }: Props) {
  return (
    <div
      className="blog-content"
      style={{
        fontFamily: "'Space Grotesk', sans-serif",
        lineHeight: "1.8",
        color: isLight ? "#1a1a2e" : "#ffffff",
      }}
    >
      <ReactMarkdown
        components={{
          h1: ({ children }) => (
            <h1
              className={`text-2xl md:text-3xl font-bold mt-8 mb-4 ${isLight ? "text-pink-700" : "text-pink-400"}`}
            >
              {children}
            </h1>
          ),
          h2: ({ children }) => (
            <h2
              className={`text-xl md:text-2xl mt-6 mb-2 ${isLight ? "text-pink-600" : "text-pink-300"}`}
            >
              {children}
            </h2>
          ),
          h3: ({ children }) => (
            <h3
              className={`text-lg md:text-xl mt-5 mb-1 ${isLight ? "text-pink-500" : "text-pink-200"}`}
            >
              {children}
            </h3>
          ),
          p: ({ children }) => <p className="m-0 mb-2">{children}</p>,
          blockquote: ({ children }) => (
            <blockquote
              className="m-0 mb-2 p-2"
              style={{
                borderLeft: "3px solid #d53a9d",
                background: isLight
                  ? "rgba(213, 58, 157, 0.05)"
                  : "rgba(213, 58, 157, 0.1)",
              }}
            >
              {children}
            </blockquote>
          ),
          pre: ({ children }) => <>{children}</>,
          strong: ({ children }) => (
            <strong className={isLight ? "text-orange-600" : "text-orange-300"}>
              {children}
            </strong>
          ),
          li: ({ children }) => (
            <li className="m-0 mb-2" style={{ listStyle: "none" }}>
              <span style={{ color: "#d53a9d", marginRight: "0.5rem" }}>•</span>
              {children}
            </li>
          ),
          a: ({ href, children }) => (
            <a href={href} target="_blank" rel="noopener noreferrer">
              {children}
            </a>
          ),
          img: ({ src, alt }) => (
            <LazyImage src={src!} alt={alt} maxWidth={800} maxHeight={400} />
          ),
          ul: ({ children }) => (
            <ul className="m-0 mb-2 pl-3" style={{ listStyle: "none" }}>
              {children}
            </ul>
          ),
          code: ({ className, children }) => {
            const match = /language-(\w+)/.exec(className ?? "");
            if (match) {
              const code = String(children).replace(/\n$/, "");
              return (
                <CodeBlock code={code} language={match[1]} isLight={isLight} />
              );
            }
            return (
              <code
                style={{
                  background: isLight
                    ? "rgba(0,0,0,0.06)"
                    : "rgba(255,255,255,0.1)",
                  padding: "2px 6px",
                  borderRadius: "4px",
                  fontSize: "0.9em",
                  wordBreak: "break-word",
                }}
              >
                {children}
              </code>
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
