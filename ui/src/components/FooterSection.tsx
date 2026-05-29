export default function FooterSection() {
  return (
    <div className="toolbar border-none bg-transparent mt-4">
      <span className="text-sm">
        <span className="mr-1">GitHub Portfolio by</span>
        <strong>@shakeelansari63</strong>
      </span>
      <div className="flex gap-2">
        <a
          href="https://github.com/shakeelansari63/portfolio"
          target="_blank"
          className="toolbar-link text-sm flex align-items-center gap-1"
        >
          Star
        </a>
        <a
          href="https://github.com/shakeelansari63/portfolio/fork"
          target="_blank"
          className="toolbar-link text-sm flex align-items-center gap-1"
        >
          Fork
        </a>
      </div>
    </div>
  );
}
