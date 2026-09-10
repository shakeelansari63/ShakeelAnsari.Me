import { useEffect, useRef } from 'react';
import mermaid from 'mermaid';
import { useTheme } from '../../context/ThemeContext';
import { settings } from '../../data/settings';

const lightTheme = {
  primaryColor: settings.themeColor,
  primaryTextColor: '#1f1f1f',
  primaryBorderColor: settings.themeColor,
  lineColor: '#8c8c8c',
  secondaryColor: '#f5f5f5',
  tertiaryColor: '#ffffff',
};

const darkTheme = {
  primaryColor: settings.themeColor,
  primaryTextColor: '#ffffff',
  primaryBorderColor: settings.themeColor,
  lineColor: '#a6a6a6',
  secondaryColor: '#1f1f1f',
  tertiaryColor: '#141414',
};

function getThemeVariables(theme: 'light' | 'dark') {
  return theme === 'dark' ? darkTheme : lightTheme;
}

export default function MermaidRenderer({ chart }: { chart: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const { theme } = useTheme();

  useEffect(() => {
    if (ref.current) {
      mermaid.initialize({
        theme: 'base',
        themeVariables: getThemeVariables(theme),
        flowchart: { useMaxWidth: true, htmlLabels: false },
        securityLevel: 'strict',
      });

      ref.current.removeAttribute('data-processed');
      mermaid
        .run({
          nodes: [ref.current],
          suppressErrors: true,
        })
        .catch(() => {});
    }
  }, [chart, theme]);

  const bgColor = theme === 'dark' ? '#1f1f1f' : '#f5f5f5';

  return (
    <div
      className="mermaid"
      ref={ref}
      style={{
        textAlign: 'center',
        margin: '1.5rem 0',
        overflowX: 'auto',
        background: bgColor,
        borderRadius: '8px',
        padding: '1rem',
      }}
    >
      {chart}
    </div>
  );
}