import MarkdownRenderer from '../shared/MarkdownRenderer';

interface Props {
  content: string;
}

export default function ArticleContent({ content }: Props) {
  return <MarkdownRenderer content={content} />;
}