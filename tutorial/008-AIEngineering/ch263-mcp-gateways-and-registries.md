# MCP Gateways and Registries — Enterprise Control Planes

> Enterprises cannot let every dev install random MCP servers. A gateway centralizes auth, RBAC, audit, rate limiting, caching, and tool-poisoning detection, then exposes the merged tool surface as a single MCP endpoint.

**Type:** Learn
**Languages:** Python (stdlib, minimal gateway)
**Prerequisites:** Phase 13 · 15, 16
**Time:** ~45 minutes

## Learning Objectives
- Explain where an MCP gateway sits (between clients and backend MCP servers)
- Implement the five gateway responsibilities: auth, RBAC, audit, rate limit, policy
- Enforce a pinned-tool-hash manifest at the gateway layer
- Differentiate the Official MCP Registry from metaregistries

## The Problem

A Fortune 500 has 30 approved MCP servers, 5000 developers, compliance requirements, and a security team wanting centralized policy. The gateway pattern: single Streamable HTTP endpoint, gateway holds backend credentials, every request authenticated and scoped.

## The Concept

### Five gateway responsibilities

1. **Auth**: OAuth 2.1 to identify developer, maps to roles
2. **RBAC**: per-user policy — which servers, tools, scopes
3. **Audit**: every call logged with who, what, when, result
4. **Rate limit**: per-user/per-tool/per-server caps
5. **Policy**: reject poisoned descriptions, enforce Rule of Two, redact PII

### Tool-hash pinning at the gateway

```python
# Gateway holds manifest of approved tool description hashes
PINNED_MANIFEST = {"notes::notes_search": "sha256:a1b2...", ...}
# At discovery, compare server's tools/list against manifest
# Remove any tool whose hash changed
```

### Namespace merging

Gateways merge tool namespaces with prefix-on-collision: `github.open_pr`, `notes.search`.

### Registries

| Registry | Description |
|----------|-------------|
| Official MCP Registry | `registry.modelcontextprotocol.io`, namespace-verified, reverse-DNS |
| Glama | Search-centric metaregistry |
| MCPMarket | Commercial-leaning directory |
| MCP.so | Community directory |
| Smithery | Package-manager-style installation |
| LobeHub | UI-integrated in LobeChat |

### Vendor survey (April 2026)

| Vendor | Strength |
|--------|----------|
| Cloudflare MCP Portals | Edge-hosted, OAuth integrated |
| Kong AI Gateway | K8s-native, fine-grained policy |
| IBM ContextForge | Enterprise IAM, compliance |
| TrueFoundry | DevOps-leaning, metrics-first |

## Use It

`code/main.py` ships a minimal gateway in ~150 lines: authenticates by Bearer token, holds per-user RBAC policy, routes to two backends, logs every call to an audit log, enforces rate limits, and rejects backend tools whose description hashes don't match the pinned manifest.

## Exercises

1. Call as allowed user, disallowed user, and rate-limit-exceeded burst. Verify all three.
2. Add PII redaction from tool results.
3. Extend audit log to emit OpenTelemetry GenAI spans.
4. Design RBAC for 50 developers with 5 backends.
5. Read Cloudflare's enterprise MCP post; identify one feature this gateway doesn't have.

## Key Terms

| Term | What it actually means |
|------|------------------------|
| Gateway | Centralizing proxy between MCP clients and backends |
| Credential vaulting | Developers never see upstream tokens |
| Tool-hash pinning | SHA256 of approved descriptions; blocks rug-pulls centrally |
| RBAC | Per-user access control for tools and servers |
| Audit log | Append-only event log for compliance |
| Rate limit | Per-user token bucket preventing abuse |
| Official MCP Registry | Canonical upstream, namespace-verified |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/13-tools-and-protocols/17-mcp-gateways-and-registries)
