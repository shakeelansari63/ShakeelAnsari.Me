import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import CodeBlock from '../shared/CodeBlock';
import MermaidRenderer from '../shared/MermaidRenderer';
import LazyImage from '../shared/LazyImage';

interface Props {
  content: string;
}

export default function MarkdownRenderer({ content }: Props) {
  return (
    <div
      className="blog-content"
      style={{
        fontFamily: "'Space Grotesk', sans-serif",
        lineHeight: '1.8',
        fontSize: '1.125rem',
        color: 'var(--ant-color-text)',
      }}
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({ children }) => (
            <h1 className="text-2xl md:text-3xl font-bold mt-8 mb-4" style={{ color: 'var(--ant-color-primary)' }}>
              {children}
            </h1>
          ),
          h2: ({ children }) => (
            <h2 className="text-xl md:text-2xl mt-6 mb-2" style={{ color: 'var(--ant-color-primary)' }}>
              {children}
            </h2>
          ),
          h3: ({ children }) => (
            <h3 className="text-lg md:text-xl mt-5 mb-1" style={{ color: 'var(--ant-color-primary)' }}>
              {children}
            </h3>
          ),
          p: ({ children }) => (
            <p className="mx-0 mt-2 mb-3">{children}</p>
          ),
          blockquote: ({ children }) => (
            <blockquote
              className="mx-0 my-3 p-4 rounded-r-lg"
              style={{
                borderLeft: '3px solid var(--ant-color-primary)',
                background: 'rgba(213, 58, 157, 0.08)',
              }}
            >
              {children}
            </blockquote>
          ),
          pre: ({ children }) => <>{children}</>,
          strong: ({ children }) => (
            <strong style={{ color: '#f59e0b' }}>
              {children}
            </strong>
          ),
          li: ({ children }) => (
            <li className="m-0 mb-2">{children}</li>
          ),
          a: ({ href, children }) => (
            <a
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              style={{ color: 'var(--ant-color-primary)', textDecoration: 'underline' }}
            >
              {children}
            </a>
          ),
          img: ({ src, alt }) => (
            <LazyImage
              src={src!}
              alt={alt}
              maxWidth={800}
              maxHeight={400}
            />
          ),
          table: ({ children }) => (
            <div className="m-0 mb-2" style={{ overflowX: 'auto' }}>
              <table
                style={{
                  width: '100%',
                  borderCollapse: 'collapse',
                  fontSize: '0.9rem',
                }}
              >
                {children}
              </table>
            </div>
          ),
          thead: ({ children }) => (
            <thead style={{ borderBottom: '2px solid var(--ant-color-primary)' }}>
              {children}
            </thead>
          ),
          tbody: ({ children }) => <tbody>{children}</tbody>,
          tr: ({ children }) => (
            <tr style={{ borderBottom: '1px solid var(--ant-color-border)' }}>
              {children}
            </tr>
          ),
          th: ({ children }) => (
            <th
              style={{
                padding: '0.75rem 1rem',
                textAlign: 'left',
                fontWeight: 'bold',
                color: 'var(--ant-color-primary)',
                whiteSpace: 'nowrap',
              }}
            >
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td
              style={{
                padding: '0.75rem 1rem',
                textAlign: 'left',
                color: 'var(--ant-color-text)',
              }}
            >
              {children}
            </td>
          ),
          hr: () => (
            <hr
              style={{
                width: '300px',
                margin: '2rem auto',
                border: 'none',
                height: '1px',
                background: 'linear-gradient(to right, transparent, var(--ant-color-primary) 45%, var(--ant-color-primary) 55%, transparent)',
                borderRadius: '2px',
              }}
            />
          ),
          ul: ({ children }) => (
            <ul className="m-0 mb-2 pl-0">{children}</ul>
          ),
          ol: ({ children }) => (
            <ol className="m-0 mb-2 pl-3 list-decimal">
              {children}
            </ol>
          ),
          code: ({ className, children }) => {
            const match = /language-(\w+)/.exec(className ?? '');
            if (match) {
              const code = String(children).replace(/\n$/, '');
              if (match[1] === 'mermaid') {
                return <MermaidRenderer chart={code} />;
              }
              return (
                <CodeBlock code={code} language={match[1]} />
              );
            }
            return (
              <code
                style={{
                  background: 'var(--ant-color-fill-secondary)',
                  padding: '2px 6px',
                  borderRadius: '4px',
                  fontSize: '0.9rem',
                  wordBreak: 'break-word',
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