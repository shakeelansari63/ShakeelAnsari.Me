# MCP Auth in Production — Enrollment, JWKS Refresh, Audience-Pinned Tokens

> Lesson 16 stood up the OAuth 2.1 state machine in memory. By 2026, every MCP server you ship to a real org sits behind production auth: client enrollment that scales, authorization-server metadata discovery, JWKS cache refresh that does not break 3 a.m. token validation, and audience-pinned tokens that refuse cross-resource replay.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 13 · 16, 17
**Time:** ~90 minutes

## Learning Objectives
- Discover an authorization server through RFC 8414 metadata
- Implement RFC 7591 dynamic client registration
- Cache and refresh JWKS keys on a schedule
- Pin tokens to a single MCP resource using RFC 8707 resource indicators
- Separate authorization server, resource server, and client roles

## The Problem

Lesson 16 ran OAuth 2.1 in memory. Production has three gaps: enrollment (thousands of clients, no manual registration), key rotation (JWKS refresh), and audience binding (preventing token replay).

## The Concept

### RFC 8414 — Authorization Server Metadata

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "registration_endpoint": "https://auth.example.com/register",
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"]
}
```

Contract checks: `S256` must be present (no PKCE = refuse), `grant_types_supported` must exclude `password`/`implicit`.

### Client ID Metadata Documents (CIMD — recommended default)

The client uses an HTTPS URL it controls as its `client_id`. The authorization server fetches the metadata document from that URL. Trust rooted in DNS.

```json
{
  "client_id": "https://app.example.com/oauth/client.json",
  "client_name": "Example MCP Client",
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none"
}
```

### RFC 7591 — Dynamic Client Registration (fallback)

```python
POST /register
{"redirect_uris": ["http://127.0.0.1:7333/callback"],
 "grant_types": ["authorization_code", "refresh_token"],
 "client_name": "Cursor", "software_id": "com.cursor.cursor"}
# Response:
{"client_id": "c_3e7f1a", "registration_access_token": "regt_b2..."}
```

Rate-limit by source IP. Hash the `registration_access_token` at rest.

### JWKS refresh pattern

```python
# Resource server schedules a periodic job:
def refresh_jwks():
    jwks = requests.get(f"{issuer}/.well-known/jwks.json").json()
    cache[issuer] = {"keys": jwks["keys"], "fetched_at": time.time()}

# On cache miss during validation:
def resolve_key(kid):
    if kid not in cache[issuer]["keys"]:
        refresh_jwks()  # re-fetch, NEVER rotate-and-mint
    return cache[issuer]["keys"][kid]
```

The fall-back must be a re-fetch, never a rotate-and-mint (that would never produce the missing `kid` and creates a DoS vector).

### Audience validation

```python
def validate(bearer_token, required_scope):
    payload = decode_jwt(bearer_token, jwks_cache)
    if payload["aud"] != "https://notes.example.com":
        return {"valid": False, "status": 401, "error": "audience mismatch"}
    if required_scope not in payload["scope"]:
        return {"valid": False, "status": 403, "error": "insufficient_scope"}
    return {"valid": True, "sub": payload["sub"]}
```

### Mix-up attacks

Client must validate RFC 9207 `iss` parameter against the issuer it recorded before redirecting. PKCE alone does not stop mix-up (client hands `code_verifier` to whatever endpoint it's steered to).

### IdP capability matrix

| IdP | AS metadata | CIMD | DCR | PKCE | Notes |
|-----|-------------|------|-----|------|-------|
| Keycloak | yes | emerging | yes | yes | Reference IdP |
| Entra ID | yes | emerging | premium tiers | yes | Verify tenant |
| Okta | yes | emerging | Auth0/CIC | yes | Classic orgs need pre-reg |

## Use It

`code/main.py` walks the full production flow with three roles: `AuthorizationServer`, `ResourceServer`, and `Client`. Includes RFC 8414 metadata, DCR enrollment, PKCE code flow, JWKS cache refresh, key rotation, and audience-replay rejection.

## Exercises

1. Trace the flow and note how key rotation works with overlap window.
2. Add a new IdP to `authorization_servers` and confirm acceptance, then an unlisted IdP and confirm rejection.
3. Add rate-limit to `register_client`.
4. Identify two fields RFC 7591's `/register` handler doesn't validate; add them.
5. Add a CIMD path and confirm enrollment without `register_client`.
6. Prove the DoS fix: send a random `kid` and confirm key count doesn't grow.
7. Implement client-side RFC 9207 `iss` check.

## Key Terms

| Term | What it actually means |
|------|------------------------|
| CIMD | Client ID Metadata Document — HTTPS URL as client_id; AS pulls metadata |
| DCR | RFC 7591 dynamic client registration (MAY fallback since 2025-11-25) |
| JWKS | JSON Web Key Set from `jwks_uri`, indexed by `kid` |
| Rotate vs refresh | AS rotates (mints/retires keys); resource server refreshes (re-fetches) |
| Resource indicator | RFC 8707 `resource` pinning token to one server |
| Audience replay | Token for Server A presented to Server B; defended by `aud` validation |
| Mix-up attack | Client steered to wrong token endpoint; defended by RFC 9207 `iss` |

[Reference](https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/phases/13-tools-and-protocols/18-mcp-auth-production)
