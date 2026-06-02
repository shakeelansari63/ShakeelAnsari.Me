import ReactMarkdown from 'react-markdown';
import LazyImage from '../shared/LazyImage';

interface Props {
  content: string;
  isLight?: boolean;
}

export default function ArticleContent({ content, isLight }: Props) {
  return (
    <div style={{ fontFamily: 'SpaceMono, monospace', lineHeight: '1.8', color: isLight ? '#1a1a2e' : '#ffffff' }}>
      <ReactMarkdown
        components={{
          h2: ({ children }) => (
            <h2 className={`text-2xl mt-4 mb-2 ${isLight ? 'text-pink-600' : 'text-pink-300'}`}>{children}</h2>
          ),
          h3: ({ children }) => (
            <h3 className={`text-xl mt-3 mb-1 ${isLight ? 'text-pink-500' : 'text-pink-200'}`}>{children}</h3>
          ),
          p: ({ children }) => <p className="m-0 mb-2">{children}</p>,
          blockquote: ({ children }) => (
            <blockquote
              className="m-0 mb-2 p-2"
              style={{ borderLeft: '3px solid #d53a9d', background: isLight ? 'rgba(213, 58, 157, 0.05)' : 'rgba(213, 58, 157, 0.1)' }}
            >
              {children}
            </blockquote>
          ),
          strong: ({ children }) => (
            <strong className={isLight ? 'text-orange-600' : 'text-orange-300'}>{children}</strong>
          ),
          li: ({ children }) => <li className="m-0 mb-1">{children}</li>,
          img: ({ src, alt }) => <LazyImage src={src!} alt={alt} />,
          ul: ({ children }) => <ul className="m-0 mb-2 pl-3" style={{ listStyle: 'none' }}>{children}</ul>,
          code: ({ children }) => (
            <code style={{ background: isLight ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.1)', padding: '2px 6px', borderRadius: '4px', fontSize: '0.9em' }}>
              {children}
            </code>
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
