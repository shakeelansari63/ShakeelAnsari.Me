# MCP Security II — OAuth 2.1, Resource Indicators, Incremental Scopes

> Remote MCP servers need authorization, not just authentication. The 2025-11-25 spec aligns with OAuth 2.1 + PKCE + resource indicators (RFC 8707) + protected-resource metadata (RFC 9728). SEP-835 adds incremental scope consent.

**Type:** Build
**Languages:** Python (stdlib, OAuth state machine simulator)
**Prerequisites:** Phase 13 · 09, 15
**Time:** ~75 minutes

## Learning Objectives
- Distinguish resource server from authorization server responsibilities
- Walk the PKCE-protected OAuth 2.1 authorization code flow
- Use `resource` (RFC 8707) to prevent confused-deputy attacks
- Implement step-up authorization: 403 with WWW-Authenticate asking for higher scope

## The Problem

Pre-2025 MCP shipped remote servers with ad-hoc API keys or no auth. Three needs: ordinary OAuth for remote servers, scope escalation (step-up), and confused-deputy prevention via resource indicators.

## The Concept

### Authorization code + PKCE

```
1. Client: code_verifier + code_challenge (SHA256)
2. Client redirects to /authorize?code_challenge=...&resource=https://notes.example.com
3. User consents; redirect to callback with code
4. Client POSTs to /token with code + code_verifier + resource
5. AS validates verifier hash, issues access token
6. Client: Authorization: Bearer ... on every request
```

### Protected-resource metadata (RFC 9728)

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:read", "notes:write", "notes:delete"]
}
```

### Resource indicators (RFC 8707)

`resource` parameter pins the token's intended audience. Every request: server checks `token.aud == self.resource_url`. Mismatch = 401. This stops cross-server token replay.

### Step-up authorization (SEP-835)

```
User grants notes:read. Later, agent needs to delete a note.
Server responds:
  HTTP 403 Forbidden
  WWW-Authenticate: Bearer error="insufficient_scope",
      scope="notes:delete", resource="https://notes.example.com"
Client prompts user for additional scope, re-does mini OAuth flow, retries.
```

### Confused-deputy prevention

Token binds to `aud`. Client binds to `client_id`. Every request validated against both. The spec bans the old "pass-the-token" pattern.

### Short-lived tokens

Access tokens: 1 hour default. Refresh tokens rotate on every refresh. Client handles silent refresh in background.

## Use It

`code/main.py` simulates the full OAuth 2.1 step-up flow as a state machine: PKCE generation, authorization code flow with resource indicator, protected-resource metadata endpoint, token validation with audience check, and step-up on `insufficient_scope`.

## Exercises

1. Trace the two-scope step-up flow and note which hops repeat.
2. Add refresh-token rotation with theft detection.
3. Implement the protected-resource metadata as a real HTTP endpoint.
4. Design a scope hierarchy for a GitHub MCP server with step-up between levels.
5. Read RFC 8707 and RFC 9728; identify the one field in 9728 MCP uses differently.

## Key Terms

| Term | What it actually means |
|------|------------------------|
| OAuth 2.1 | Consolidated RFC mandating PKCE, forbidding implicit |
| PKCE | Code verifier + challenge defeating authorization-code interception |
| Resource indicator | RFC 8707 `resource` pinning token to one server |
| Protected-resource metadata | RFC 9728 `.well-known/oauth-protected-resource` |
| Step-up authorization | SEP-835 for adding scopes on demand |
| Confused deputy | Token reuse across services |
| Scope hierarchy | Graduated scope set with step-up between levels |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/13-tools-and-protocols/16-mcp-security-oauth-2-1)
