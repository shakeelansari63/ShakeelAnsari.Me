import { useState } from 'react';
import { PrismLight as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark, oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism';
import tsx from 'react-syntax-highlighter/dist/esm/languages/prism/tsx';
import typescript from 'react-syntax-highlighter/dist/esm/languages/prism/typescript';
import python from 'react-syntax-highlighter/dist/esm/languages/prism/python';
import sql from 'react-syntax-highlighter/dist/esm/languages/prism/sql';
import bash from 'react-syntax-highlighter/dist/esm/languages/prism/bash';
import json from 'react-syntax-highlighter/dist/esm/languages/prism/json';
import yaml from 'react-syntax-highlighter/dist/esm/languages/prism/yaml';
import php from 'react-syntax-highlighter/dist/esm/languages/prism/php';
import go from 'react-syntax-highlighter/dist/esm/languages/prism/go';
import scala from 'react-syntax-highlighter/dist/esm/languages/prism/scala';
import cypher from 'react-syntax-highlighter/dist/esm/languages/prism/cypher';
import { CopyOutlined, CheckOutlined } from '@ant-design/icons';

SyntaxHighlighter.registerLanguage('tsx', tsx);
SyntaxHighlighter.registerLanguage('typescript', typescript);
SyntaxHighlighter.registerLanguage('python', python);
SyntaxHighlighter.registerLanguage('sql', sql);
SyntaxHighlighter.registerLanguage('bash', bash);
SyntaxHighlighter.registerLanguage('sh', bash);
SyntaxHighlighter.registerLanguage('shell', bash);
SyntaxHighlighter.registerLanguage('json', json);
SyntaxHighlighter.registerLanguage('yaml', yaml);
SyntaxHighlighter.registerLanguage('php', php);
SyntaxHighlighter.registerLanguage('go', go);
SyntaxHighlighter.registerLanguage('golang', go);
SyntaxHighlighter.registerLanguage('scala', scala);
SyntaxHighlighter.registerLanguage('cypher', cypher);

interface Props {
  code: string;
  language: string;
  isLight?: boolean;
}

export default function CodeBlock({ code, language, isLight }: Props) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="code-block" style={{ position: 'relative', margin: '1rem 0' }}>
      <button
        onClick={handleCopy}
        className="code-block-copy"
        style={{
          position: 'absolute',
          top: '0.5rem',
          right: '0.5rem',
          border: 'none',
          cursor: 'pointer',
          padding: '0.5rem',
          borderRadius: '4px',
          background: 'var(--ant-color-fill-secondary)',
          color: 'var(--ant-color-text)',
          zIndex: 1,
          lineHeight: 0,
        }}
        aria-label="Copy code"
      >
        {copied ? <CheckOutlined style={{ fontSize: '0.9rem', color: '#22c55e' }} /> : <CopyOutlined style={{ fontSize: '0.9rem' }} />}
      </button>
      <SyntaxHighlighter
        style={isLight ? oneLight : oneDark}
        language={language}
        PreTag="div"
        customStyle={{
          fontSize: '0.95rem',
          borderRadius: '8px',
        }}
      >
        {code}
      </SyntaxHighlighter>
    </div>
  );
}