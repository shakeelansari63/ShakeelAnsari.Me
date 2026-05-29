import { userData } from '../services/data';

export default function ToolBar() {
  return (
    <nav className="toolbar">
      <div className="toolbar-start">
        <a href="/" className="toolbar-logo">
          @{userData.devUsername}
        </a>
      </div>
      <div className="toolbar-end">
        <a href="#projects" className="toolbar-link">
          Projects
        </a>
        <a href="#stats" className="toolbar-link">
          Stats
        </a>
        <a href="#contributions" className="toolbar-link">
          Contributions
        </a>
      </div>
    </nav>
  );
}
