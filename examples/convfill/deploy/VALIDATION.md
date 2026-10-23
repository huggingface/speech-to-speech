# Validation scope

The example, backend registry and realtime pipeline builder pass 157 Python
regression tests, including 52 example tests. The current demo passes six UI
tests and 32 Agents SDK adapter/playback/timing/viseme tests. Ruff check/format,
launcher syntax and CLI option checks pass.

These tests use fake models and controlled HTTP streams. They cover raw prompt
preservation, incremental sentence delivery, control-token filtering, empty
chunks, final buffered text, HTTP/SSE failures, deadlines, cancellation,
interrupted-history handling, private-Space protocol negotiation, single-session
admission, reconnect and disabling camera access in speech-only mode.

Live inference was also exercised separately. Deployment addresses, topology,
resource identifiers, runtime telemetry and operational settings are intentionally
kept out of this public example. Live validation scripts are included for
operators to run against their own configured services.

A fresh container build and Space deployment of this port have not been
validated. Complete speech/browser inference should be checked after deployment.
Dispatched audio is not proof of physical playback, and interrupted history has
no exact heard-prefix guarantee.
