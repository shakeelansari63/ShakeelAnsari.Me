# UI Development Skill

## Context
This is a **React 18 SPA** with **TypeScript**, **Vite**, **Ant Design (AntD) v5**, and **Ant Design CSS utilities**. No Tailwind CSS. No Redux. No class components.

## Architecture Rules

### Routing & Code Splitting
- Routes are defined in `ui/src/App.tsx` using `react-router-dom` v7 (`BrowserRouter`).
- **All pages except `MainPage` must use `React.lazy()` + `Suspense`** with `Spin` or `Skeleton` fallback:
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
- **Use Ant Design CSS utilities and design tokens FIRST** for layouts and styling:
  - Grid: `Row`/`Col` components with `span`, `xs`, `sm`, `md`, `lg`, `xl`, `xxl` props
  - Flex: `d-flex`, `flex-column`, `align-items-center`, `justify-content-between`, `flex-wrap`
  - Spacing: `margin-*`, `padding-*`, `gap-*` utility classes or `space`/`gap` props on components
  - Typography: `text-*` color utilities, `font-weight-*`, `text-center`, `text-truncate`
  - Display: `d-none`, `d-flex`, `d-block`, `d-md-flex`, `d-lg-block`
- **Use Ant Design ConfigProvider theme tokens** for consistent design (colors, spacing, border-radius, etc.)
- **Use custom SCSS (`App.scss`) ONLY when Ant Design utilities cannot achieve the desired effect** (animations, keyframes, complex selectors, deep overrides).
- **Inline styles ONLY for:**
  - Dynamic values (e.g., `style={{ width: `${percent}%` }}`)
  - Component prop style overrides (e.g., `style={{ borderRadius: '8px' }}`)
- **No Tailwind CSS** — never import or generate Tailwind classes.
- **Use Ant Design Icons** from `@ant-design/icons` — never use other icon libraries.

### Ant Design Component Patterns
| Component | Usage Pattern |
|-----------|--------------|
| Button | `<Button type="primary" danger={false} htmlType="button" />` |
| Card | `<Card hoverable className="h-100" style={{ borderRadius: 12 }} />` |
| Layout | `<Layout><Sider /><Layout><Header /><Content /><Footer /></Layout></Layout>` |
| Skeleton | `<Skeleton active avatar paragraph={{ rows: 3 }} />` |
| Tag | `<Tag color="blue">React</Tag>` |
| Drawer | `<Drawer open={open} onClose={handler} placement="right" width={500} />` |
| Input | `<Input value={val} onChange={e => setVal(e.target.value)} allowClear />` |
| Input.Password | `<Input.Password value={val} onChange={e => setVal(e.target.value)} />` |
| message / notification | `message.success('Done')` / `notification.open({ message: 'Title', description: '...' })` |
| Table | `<Table columns={columns} dataSource={data} rowKey="id" pagination={{ pageSize: 10 }} />` |
| Modal | `<Modal open={open} onOk={handleOk} onCancel={handleCancel} title="Title" />` |
| Select | `<Select options={options} placeholder="Select" allowClear />` |
| Avatar | `<Avatar src={url} alt="Name" shape="circle" size="large" />` |
| Breadcrumb | `<Breadcrumb items={items} />` |
| Pagination | `<Pagination total={total} pageSize={10} showSizeChanger />` |

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

### SEO / Head Management (`react-helmet-async`)
- The app uses `react-helmet-async` for per-page `<title>`, meta tags (OG, Twitter, description), and robots directives **during client-side navigation**.
- **Production initial HTML** is server-rendered per-URL by `api/public/page.php` (root `.htaccess` rewrites non-file GETs to it). Crawlers that don't execute JS receive unique server `<head>` (title, description, canonical, OG/Twitter, JSON-LD) straight from the API content sources. Helmet then hydrates it — keep client and server titles consistent (both `${title} — ${seo.name}`).
- **Setup:** `main.tsx` wraps the app in `<HelmetProvider>`.
- **Config:** Placeholder values live in `ui/src/data/seo.ts` mirroring the `[{#SEO-*#}]` tokens that CI/CD replaces at build time.
  - Import `seo` from `"../data/seo"` and use `seo.name`, `seo.domain`, etc. — never hardcode these values.
- **Pattern:**
  ```tsx
  import { Helmet } from "react-helmet-async";
  import { seo } from "../data/seo";

  <Helmet>
    <title>{`Page — ${seo.name}`}</title>
    <meta name="description" content="..." />
    <meta property="og:title" content={`Page — ${seo.name}`} />
    <meta property="og:url" content={`https://${seo.domain}/page-path`} />
  </Helmet>
  ```
- For pages with dynamic data (e.g. `BlogReaderPage`), compute meta tags reactively:
  ```tsx
  const metaTitle = post ? `${post.title} — ${seo.name}` : `Blog — ${seo.name}`;
  ```
- **Do NOT** use `document.title = ...` directly — Helmet handles it.
- **Do NOT** manually create/append `<meta>` elements — use `<Helmet>` instead.
- The `[{#SEO-*#}]` placeholders in `ui/index.html` remain as the build template for `page.php`; `page.php` overrides the head at runtime from `api/.env` `SEO_*` vars.

### Static Data Layer (`ui/src/data/`)
These files are intentionally static (no API calls):
- `profile.ts` — central user profile object re-exporting skills, work, expo, social links.
- `settings.ts` — feature toggles `{ showExpo, showBlogs, showTutorial }`.
- `skills.ts` — string array of skill names.
- `work.ts` — work experience timeline entries.
- `expo.ts` — portfolio project entries.
- `seo.ts` — SEO placeholder values (`[{#SEO-*#}]` tokens).
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
- Use Ant Design's tree-shaking imports (import from `antd/es/...` or `antd/lib/...`) for smaller bundles.

### Critical Guardrails
- ❌ **Never modify `[{#SEO-*#}]` placeholders** in `ui/index.html` or anywhere in source. These are replaced by CI/CD pipeline.
- ❌ **No `document.title = ...`** — use `<Helmet>` from `react-helmet-async` instead.
- ❌ **No manual `document.createElement('meta')`** — use `<Helmet>` with `<meta>` children.
- ❌ **No Tailwind CSS** — use Ant Design utilities and design tokens.
- ❌ **No Redux/Zustand** — use local state with hooks.
- ❌ **No class components** — use functional components with hooks.
- ❌ **No hardcoded backend URLs** — always use relative paths `/api/*`.
- ❌ **No Axios** — use native `fetch`.
- ❌ **No PrimeReact or PrimeFlex** — use Ant Design components and utilities.
- ❌ **No other icon libraries** — use `@ant-design/icons` only.

(End of file)