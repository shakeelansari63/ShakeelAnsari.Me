import MarkdownRenderer from "../shared/MarkdownRenderer";

interface Props {
  content: string;
  isLight?: boolean;
}

export default function ArticleContent({ content, isLight }: Props) {
  return <MarkdownRenderer content={content} isLight={isLight} />;
}
