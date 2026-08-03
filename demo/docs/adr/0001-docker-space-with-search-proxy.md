# Convert the static Space into a Docker app with a search proxy

To give the model a web search tool, the executor (which runs in the browser of a
public Space) needs a search key. A `sdk: static` Space serves files as-is with no
runtime process, so it cannot hold a secret the browser uses without exposing it.
We convert the Space from `sdk: static` to `sdk: docker`: a single container runs a
small server (FastAPI + uvicorn) that both serves the existing front-end *unchanged*
and exposes a same-origin `/search` proxy holding `SERPER_API_KEY` server-side. The
whole app lives in that one container; the s2s speech-to-speech backend stays the
separate load-balanced service it already is.

## Considered options

- **Stay static, user-supplied key only** — no owner key; search works only if each
  user pastes their own. Rejected as the default because it leaves the deployed demo
  with no working search.
- **Separate proxy service** — same secrecy, but splits the app across two deploys.
  Rejected: the app must live in one container.
- **Key baked into client JS at build time** — would be readable in the served
  bundle. Rejected: defeats the point.

## Consequences

- Deployment is no longer static: there is a Dockerfile and a server process; the
  README front-matter changes from `sdk: static` to `sdk: docker`.
- The client calls `/search` same-origin; the server reads the key from env. A user
  may still supply their own key as a fallback, sent per-request to the proxy.
- The front-end, audio pipeline, and s2s handshake are untouched — only the hosting
  shape and the new route are added.
