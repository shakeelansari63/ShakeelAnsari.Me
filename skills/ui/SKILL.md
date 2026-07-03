# UI Development Skill — `shakeelansari.me`

## Context
This is a **React 18 SPA** with **TypeScript**, **Vite**, **PrimeReact** (lara-dark-pink theme), and **PrimeFlex** utilities. No Tailwind CSS. No Redux. No class components.

## Architecture Rules

### Routing & Code Splitting
- Routes are defined in `ui/src/App.tsx` using `react-router-dom` v7 (`BrowserRouter`).
- **All pages except `MainPage` must use `React.lazy()` + `Suspense`** with `LoadingSpinner` fallback:
  ```tsx
  const BlogPage = React.lazy(() => import('./pages/BlogPage'));
  ```
- Route visibility is gated by feature flags from `ui/src/data/settings.ts`:
  - `settings.showBlogs` → `/blog`, `/blog/:id`
  - `settings.showExpo` → `/expo`, `/product/:id`
  - `settings.showTutorial` → `/learn`, `/learn/:subjectId`, `/learn/:subjectId/:chapterId`
- Use `historyApiFallback: true` in `vite.config.js` for SPA fallback (already configured).
- Catch-all `*` route renders `NotFoundPage`.

### Component Conventions
- **Functional components only** with hooks. No class components.
- Define an interface for component props (named `Props` or `{ComponentName}Props`).
- Use `useState` + `useEffect` for state management — no Redux, Zustand, or Context API for global state.
- API calls use `.then()` pattern inside `useEffect`, not async/await in effects. Loading tracked with boolean state:
  ```tsx
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState<Type | null>(null);

  useEffect(() => {
      apiFunction()
          .then(setData)
          .finally(() => setLoading(false));
  }, []);
  ```
- Every API call must be wrapped in try/catch and return safe fallback values on failure.

### Styling Rules
- **Use PrimeFlex utility classes FIRST** for layouts and styling:
  - Grid: `grid`, `col-12`, `md:col-6`, `lg:col-4`
  - Flex: `flex`, `flex-column`, `align-items-center`, `justify-content-between`
  - Spacing: `p-3`, `m-2`, `gap-3`, `pt-4`, `mx-auto`
  - Typography: `text-lg`, `font-bold`, `text-pink-400`, `text-gray-300`, `text-center`
  - Display: `hidden`, `flex`, `block`, `md:flex`, `lg:block`
- **Use custom SCSS (`App.scss`) ONLY when PrimeFlex cannot achieve the desired effect** (animations, keyframes, complex selectors, PrimeReact component deep overrides).
- **Inline styles ONLY for:**
  - Dynamic values (e.g., `style={{ width: `${percent}%` }}`)
  - PrimeReact component prop overrides (e.g., `pt={{ body: { className: 'p-0' } }}`)
  - Removing PrimeReact focus outlines: `style={{ outline: 'none', boxShadow: 'none' }}`
- **No Tailwind CSS** — never import or generate Tailwind classes.

### PrimeReact Component Patterns
| Component | Usage Pattern |
|-----------|--------------|
| Button | `<Button text severity="secondary" className="text-pink-500" style={{ outline: 'none', boxShadow: 'none' }} />` |
| Card | `<Card className="cursor-pointer h-full" pt={{ body: { className: 'p-0' } }}>` |
| Toolbar | `<Toolbar className="border-none bg-transparent" />` |
| Skeleton | `<Skeleton className="mb-2" width="100%" height="2rem" />` |
| Chip | `<Chip label="React" className="text-pink-400" />` |
| Sidebar | `<Sidebar visible={visible} onHide={handler} className="w-full md:w-20rem">` |
| InputText | `<InputText value={val} onChange={e => setVal(e.target.value)} className="w-full" />` |
| Password | `<Password value={val} onChange={e => setVal(e.target.value)} feedback={false} />` |
| Toast | `<Toast ref={toast} position="bottom-right" />` |

### TypeScript Models (`ui/src/models/types.ts`)
All API response types are defined here. Key interfaces:
- `BlogPost` — id, title, excerpt, content?, date, readTime, tags, bannerImage?, views, likes
- `LearnSubject` — id, title, folder, sort_order, thumbnail
- `LearnChapter` — id, chapter_id, title, sort_order
- `AnalyticsData` — viewsByDate, likesByDate, viewsByCountry, likesByCountry, topBlogs, etc.
- `GitProfile` — avatar_url, bio, company, location, followers, following, public_repos, hireable, name, login, html_url
- `GitProject` — id, name, description, html_url, stargazers_count, language, fork
- `ExpoProject` — title, description, techStack, githubUrl, liveUrl, image?
- `WorkExperience` — company, roles[{title, period, description?}]

### API Service Layer (`ui/src/services/api.ts`)
- All API paths are relative (`/api/blogs`, `/api/learn/subjects`, etc.) — never hardcode full URLs.
- Every function is a `fetch` wrapper with try/catch returning typed fallback (empty array, null, etc.).
- Each function has a typed return matching its API response.
- Use `fetch` directly — no Axios or other HTTP libraries.

### Static Data Layer (`ui/src/data/`)
These files are intentionally static (no API calls):
- `profile.ts` — central user profile object re-exporting skills, work, expo, social links.
- `settings.ts` — feature toggles `{ showExpo, showBlogs, showTutorial }`.
- `skills.ts` — string array of skill names.
- `work.ts` — work experience timeline entries.
- `expo.ts` — portfolio project entries.
- **Do NOT create API endpoints for these** — they are intentionally client-side only.

### Markdown Rendering
- `MarkdownRenderer` wraps `react-markdown` with `remark-gfm`.
- Code blocks use `react-syntax-highlighter` with PrismLight (registered languages: tsx, typescript, python, sql, bash, json, yaml, php, go, scala).
- Mermaid diagrams detected via `language-mermaid` code class → rendered with `MermaidRenderer`.
- Images rendered via `LazyImage` (lazy loading + skeleton + fade-in).
- Tables, blockquotes, and headings use custom styled components.

### Performance Constraints
- Keep third-party additions separate or lazy-loaded — do not bloat the main vendor bundle.
- Route elements must be `React.lazy()` wrapped (except MainPage).
- Images should use lazy loading (`LazyImage` component).

### Critical Guardrails
- ❌ **Never modify `[{#SEO-*#}]` placeholders** in `ui/index.html` or anywhere in source. These are replaced by CI/CD pipeline.
- ❌ **No Tailwind CSS** — use PrimeFlex utilities.
- ❌ **No Redux/Zustand** — use local state with hooks.
- ❌ **No class components** — use functional components with hooks.
- ❌ **No hardcoded backend URLs** — always use relative paths `/api/*`.
- ❌ **No Axios** — use native `fetch`.
