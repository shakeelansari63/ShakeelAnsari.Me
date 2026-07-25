# Skills and Agent SDKs — Anthropic Skills, AGENTS.md, OpenAI Apps SDK

> MCP says "what tools exist." Skills say "how to do a task." The 2026 stack layers both. Anthropic's Agent Skills ship as SKILL.md with progressive disclosure. OpenAI's Apps SDK is MCP plus widget metadata. AGENTS.md sits at the repo root as project-level agent context.

**Type:** Learn
**Languages:** Python (stdlib, SKILL.md parser and loader)
**Prerequisites:** Phase 13 · 07
**Time:** ~45 minutes

## Learning Objectives
- Distinguish the three layers: AGENTS.md (project context), SKILL.md (reusable know-how), MCP (tools)
- Write a SKILL.md with YAML frontmatter and progressive disclosure
- Load skills filesystem-style into an agent runtime
- Compose a skill with an MCP server and an AGENTS.md for cross-agent portability

## The Problem

An engineer distills a release-notes-writing workflow. They want to use it from Claude Code, Cursor, and Codex CLI. Pre-2026: copy the workflow three times. AGENTS.md + SKILL.md fix this.

## The Concept

### AGENTS.md

One file at repo root. Every coding agent in 2026 supports it.

```markdown
# Project: my-service

## Conventions
- TypeScript with strict mode.
- Tests run with `pnpm test`.

## Build and run
- `pnpm dev` for local dev server.
- `pnpm build` for production bundle.
```

### SKILL.md format

Anthropic's Agent Skills (open standard, December 2025):

```markdown
---
name: release-notes-writer
description: Write changelog entry for latest merged PRs.
---

# Release notes writer

1. List PRs merged since last tag. Use `gh pr list --base main --state merged`.
2. Group by label: feature, fix, chore, docs.
3. For each PR: `- <title> (#<num>)`.
4. Draft in CHANGELOG.md.

## Notes
- Never include commits without a PR.
- Skip "chore" entries from public changelog.
```

### Progressive disclosure

Skills reference sub-resources fetched only when needed:

```
skills/release-notes-writer/
  SKILL.md
  style-guide.md
  template.md
  scripts/generate.sh
```

SKILL.md says "see style-guide.md." Agent pulls it only when the skill runs, avoiding prompt bloat.

### Filesystem discovery

Agent runtimes scan: `~/.anthropic/skills/*/SKILL.md`, `./skills/*/SKILL.md`, `~/.claude/skills/*/SKILL.md`.

### The three-layer stack

| Layer | File | Loaded when | Purpose |
|-------|------|-------------|---------|
| AGENTS.md | repo root | session start | Project conventions |
| SKILL.md | skills/ directory | skill invoked | Reusable workflow |
| MCP server | external process | tools needed | Callable actions |

### Cross-agent portability via SkillKit

Tools like SkillKit translate a single SKILL.md into the native format of 32+ AI agents. One source of truth, many consumers.

### OpenAI Apps SDK

Launched October 2025. Built directly on MCP: an MCP server (tools, resources, prompts) plus widget metadata for ChatGPT's UI, plus optional `ui://` resources.

## Use It

`code/main.py` ships a stdlib SKILL.md parser and loader. It discovers skills under `./skills/`, parses YAML frontmatter plus markdown body, and simulates an agent loop invoking a skill by name.

## Exercises

1. Add a second skill under `skills/` and confirm the loader picks it up.
2. Write an AGENTS.md for this course repo.
3. Port a multi-step workflow from your team's docs into a SKILL.md.
4. Translate the skill into Cursor's and Codex's native formats.
5. Read the Anthropic Agent Skills blog post; identify one SDK feature the loader doesn't cover.

## Key Terms

| Term | What it actually means |
|------|------------------------|
| SKILL.md | YAML frontmatter + markdown body, loaded by agent runtime |
| AGENTS.md | Repo-root project conventions, read on session start |
| Progressive disclosure | Lazy-load sub-resources from skill directory |
| Frontmatter | YAML metadata block in `---` delimiters |
| Claude Agent SDK | Anthropic's skill runtime (`@anthropic-ai/claude-agent-sdk`) |
| OpenAI Apps SDK | MCP + widget metadata for ChatGPT UI |
| SkillKit | Cross-agent translator covering 32+ agent formats |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/13-tools-and-protocols/22-skills-and-agent-sdks)
