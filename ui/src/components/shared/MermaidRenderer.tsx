import { useEffect, useRef } from "react";
import mermaid from "mermaid";

mermaid.initialize({
  theme: "base",
  themeVariables: {
    primaryColor: "#d53a9d",
    primaryTextColor: "#fff",
    primaryBorderColor: "#d53a9d",
    lineColor: "#d53a9d",
    secondaryColor: "#1a1a2e",
    tertiaryColor: "#2d2d5e",
  },
  flowchart: { useMaxWidth: true, htmlLabels: true },
  securityLevel: "loose",
});

interface Props {
  chart: string;
}

export default function MermaidRenderer({ chart }: Props) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (ref.current) {
      ref.current.removeAttribute("data-processed");
      mermaid
        .run({
          nodes: [ref.current],
          suppressErrors: true,
        })
        .catch(() => {});
    }
  }, [chart]);

  return (
    <div
      className="mermaid"
      ref={ref}
      style={{
        textAlign: "center",
        margin: "1.5rem 0",
        overflowX: "auto",
        background: "rgba(255,255,255,0.03)",
        borderRadius: "8px",
        padding: "1rem",
      }}
    >
      {chart}
    </div>
  );
}
