import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import CodeBlock from '../shared/CodeBlock';
import LazyImage from '../shared/LazyImage';
import { Suspense, lazy } from 'react';
import { settings } from '../../data/settings';
import { useTheme } from '../../context/ThemeContext';

const MermaidRenderer = lazy(() => import('../shared/MermaidRenderer'));

interface Props {
  content: string;
}

export default function MarkdownRenderer({ content }: Props) {
  const { theme } = useTheme();
  const isLight = theme === 'light';
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
            <h1 className="text-2xl md:text-3xl font-bold mt-20 mb-4" style={{ color: settings.themeColor }}>
              {children}
            </h1>
          ),
          h2: ({ children }) => (
            <h2 className="text-xl md:text-2xl mt-16 mb-2" style={{ color: `color-mix(in srgb, ${settings.themeColor} 85%, white)` }}>
              {children}
            </h2>
          ),
          h3: ({ children }) => (
            <h3 className="text-lg md:text-xl mt-12 mb-1" style={{ color: `color-mix(in srgb, ${settings.themeColor} 75%, white)` }}>
              {children}
            </h3>
          ),
          h4: ({ children }) => (
            <h4 className="text-base md:text-lg mt-8 mb-1" style={{ color: `color-mix(in srgb, ${settings.themeColor} 60%, white)` }}>
              {children}
            </h4>
          ),
          p: ({ children }) => (
            <p className="mx-0 mt-2 mb-3">{children}</p>
          ),
          blockquote: ({ children }) => (
            <blockquote
              className="mx-0 my-3 px-4 py-2"
              style={{
                borderLeft: '3px solid #d53a9d',
                background: isLight
                  ? 'rgba(213, 58, 157, 0.05)'
                  : 'rgba(213, 58, 157, 0.1)',
              }}
            >
              {children}
            </blockquote>
          ),
          pre: ({ children }) => <>{children}</>,
          strong: ({ children }) => (
            <strong style={{ color: isLight ? '#ea580c' : '#fdba74' }}>
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
            <thead
              style={{
                borderBottom: '2px solid var(--md-h3)',
              }}
            >
              {children}
            </thead>
          ),
          tbody: ({ children }) => <tbody>{children}</tbody>,
          tr: ({ children }) => (
            <tr
              style={{
                borderBottom: `1px solid ${isLight ? 'rgba(0,0,0,0.1)' : 'rgba(255,255,255,0.1)'}`,
              }}
            >
              {children}
            </tr>
          ),
          th: ({ children }) => (
            <th
              style={{
                padding: '0.75rem 1rem',
                textAlign: 'left',
                fontWeight: 'bold',
                color: 'var(--md-h3)',
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
                color: isLight ? '#1a1a2e' : '#e0e0e0',
              }}
            >
              {children}
            </td>
          ),
          hr: () => (
            <hr
              style={{
                width: '300px',
                margin: '3rem auto',
                border: 'none',
                height: '1px',
                background: `linear-gradient(to right, transparent, ${isLight ? '#ea580c' : '#fdba74'} 45%, ${isLight ? '#ea580c' : '#fdba74'} 55%, transparent)`,
                borderRadius: '2px',
              }}
            />
          ),
          ul: ({ children }) => (
            <ul className="m-0 mb-2 ml-0 pl-0 list-disc list-inside">{children}</ul>
          ),
          ol: ({ children }) => (
            <ol className="m-0 mb-2 ml-0 pl-0 list-decimal list-inside">
              {children}
            </ol>
          ),
          code: ({ className, children }) => {
            const match = /language-(\w+)/.exec(className ?? '');
            if (match) {
              const code = String(children).replace(/\n$/, '');
              if (match[1] === 'mermaid') {
                return (
                  <Suspense fallback={<div style={{ textAlign: 'center', padding: '1rem' }}>Loading diagram…</div>}>
                    <MermaidRenderer chart={code} />
                  </Suspense>
                );
              }
              return (
                <CodeBlock
                  code={code}
                  language={match[1]}
                  isLight={isLight}
                />
              );
            }
            return (
              <code
                style={{
                  background: isLight
                    ? 'rgba(0,0,0,0.06)'
                    : 'rgba(255,255,255,0.1)',
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
