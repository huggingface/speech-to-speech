# Per-request context extensions

A deployment can supply a read-only context provider to the Realtime server. A provider can return up to four short strings before the preparation deadline. The server formats them as reference data in a request-local copy of the eligible user message. A miss, error, expired deadline or saturated provider allows an active response to proceed. The default wait is 25 ms with four concurrent lookups per server and one per connection.

For automatic activation, place a UTF-8 binding at `<sys.prefix>/etc/speech-to-speech/context.json` or set `S2S_CONTEXT_BINDING` to an absolute file path. The file names an installed distribution, entry point, and absolute deployment configuration file. Both flag-based and YAML `serve` use the binding. The package can configure it through its own setup command. Example binding:

```json
{"format":1,"distribution":"speech-mds","entry":"mds","config":"/srv/speech/mds.json"}
```

Register a Python entry point in `speech_to_speech.context_extensions.v1`. Its callable receives the config path and returns `speech_to_speech.context_extension.ContextIntegration`. The context factory opens an async context manager on the server loop. Its provider implements `async prepare_context(request)`. Inputs include bounded original text, language hints, response/session IDs and an absolute monotonic deadline. Providers must use bounded cooperative or off-loop execution. The host’s timeout stops waiting; a native worker must cap and account for work that survives coroutine cancellation. Close underlying resources on lifespan exit.

For an ordinary CLI launch, a trusted integration can pass an ASGI middleware factory as the optional `scope_middleware` field of `ContextIntegration`. The host installs it for both transports. In protected deployments, this authentication middleware may set a server-owned `ContextScope` in ASGI `scope["state"]["speech_to_speech.context_scope"]` for the WebSocket handshake and WebRTC offer. The scope carries an exact tenant, issuer, subject, monotonic expiry, and a revocation event. Supply a new session on person or audience change. Shared installations can omit the scope and serve explicitly public material. Client-provided identifiers do not authenticate a person.

Text-only Transformers, MLX-LM, Responses and Chat Completions accept projection. Direct audio, out-of-band requests and tool continuations bypass retrieval. The server checks the input anchor and session ownership at dispatch. Data already delivered to a model or played to a client cannot be withdrawn. Publishing, durable storage, user enrollment and tool authorization belong to the deployment.
