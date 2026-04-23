## Summary

This is intended as the **last cleanup-focused PR** before the next release. The diff looks large, but **it does not change core speech-to-speech behavior**; it raises **project quality and packaging** toward a production baseline and sets us up for a PyPI-oriented layout.

### Entry point / deployment note

The app is now invoked as:

```bash
uv run speech-to-speech
```

instead of calling the interpreter under `.venv/...` directly. **This may affect deployment scripts or docs** that still assume the old path. After merge, I’ll **communicate the new entrypoint** to anyone who owns deploy or ops.

---

## What’s included

- **mypy** fixes **across the repository** so static checking is consistent with the current package layout.
- **Stronger typing** on functions (broader annotations to match real usage and catch regressions earlier).
- **Lightweight CI** running **mypy**, **ruff**, and **pytest**. **Merge-blocking checks are not enabled yet**; this is mainly to establish a baseline and iterate.
- **Reworked folder structure** to align with common Python packaging conventions and to **prepare for a future PyPI release** (clearer `src`/package boundaries and standard layout).

---

## Review focus

Please focus on: **packaging**, **imports/paths**, **CI config**, and **anything that might break deploy** (especially the `uv run speech-to-speech` entrypoint). Core pipeline logic should be unchanged.
