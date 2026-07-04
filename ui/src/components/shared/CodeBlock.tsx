import { useState } from "react";
import { PrismLight as SyntaxHighlighter } from "react-syntax-highlighter";
import {
  oneDark,
  materialLight,
} from "react-syntax-highlighter/dist/esm/styles/prism";
import tsx from "react-syntax-highlighter/dist/esm/languages/prism/tsx";
import typescript from "react-syntax-highlighter/dist/esm/languages/prism/typescript";
import python from "react-syntax-highlighter/dist/esm/languages/prism/python";
import sql from "react-syntax-highlighter/dist/esm/languages/prism/sql";
import bash from "react-syntax-highlighter/dist/esm/languages/prism/bash";
import json from "react-syntax-highlighter/dist/esm/languages/prism/json";
import yaml from "react-syntax-highlighter/dist/esm/languages/prism/yaml";
import php from "react-syntax-highlighter/dist/esm/languages/prism/php";
import go from "react-syntax-highlighter/dist/esm/languages/prism/go";
import scala from "react-syntax-highlighter/dist/esm/languages/prism/scala";
import cypher from "react-syntax-highlighter/dist/esm/languages/prism/cypher";

SyntaxHighlighter.registerLanguage("tsx", tsx);
SyntaxHighlighter.registerLanguage("typescript", typescript);
SyntaxHighlighter.registerLanguage("python", python);
SyntaxHighlighter.registerLanguage("sql", sql);
SyntaxHighlighter.registerLanguage("bash", bash);
SyntaxHighlighter.registerLanguage("sh", bash);
SyntaxHighlighter.registerLanguage("shell", bash);
SyntaxHighlighter.registerLanguage("json", json);
SyntaxHighlighter.registerLanguage("yaml", yaml);
SyntaxHighlighter.registerLanguage("php", php);
SyntaxHighlighter.registerLanguage("go", go);
SyntaxHighlighter.registerLanguage("golang", go);
SyntaxHighlighter.registerLanguage("scala", scala);
SyntaxHighlighter.registerLanguage("cypher", cypher);

interface Props {
  code: string;
  language: string;
  isLight?: boolean;
}

export default function CodeBlock({ code, language, isLight }: Props) {
  const [copied, setCopied] = useState(false);
  const [hover, setHover] = useState(false);

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div
      style={{ position: "relative", margin: "1rem 0" }}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
    >
      <button
        onClick={handleCopy}
        style={{
          position: "absolute",
          top: "0.5rem",
          right: "0.5rem",
          border: "none",
          cursor: "pointer",
          padding: "0.5rem",
          borderRadius: "4px",
          background: isLight ? "rgba(0,0,0,0.08)" : "rgba(255,255,255,0.12)",
          color: isLight ? "#1a1a2e" : "#fff",
          zIndex: 1,
          lineHeight: 0,
          opacity: hover ? 1 : 0,
          transition: "opacity 0.2s",
        }}
      >
        <i
          className={`pi ${copied ? "pi-check" : "pi-clone"}`}
          style={{ fontSize: "0.9rem", color: copied ? "#22c55e" : undefined }}
        />
      </button>
      <SyntaxHighlighter
        style={isLight ? materialLight : oneDark}
        language={language}
        PreTag="div"
        customStyle={{
          fontSize: "0.95rem",
          borderRadius: "8px",
        }}
      >
        {code}
      </SyntaxHighlighter>
    </div>
  );
}
