---
trigger: always_on
---

# Workspace Hygiene Rule

All agents MUST preserve repository structure and leave no disposable artifacts behind.

1. All one-off files MUST be created under:
   `.tmp-agent/<task-id>/`

2. One-off includes:
   plans, probes, debug scripts, logs, dumps, screenshots, temporary JSON,
   HTML, JS, Python, TXT, Markdown, patches, fixtures, copied source snapshots,
   generated reports, and experimental files.

3. NEVER place disposable files beside production source files.

4. Do not create plan.md / notes.md / debug.py / test.js / temp.json unless
   explicitly required. Prefer in-memory reasoning.

5. Before finishing:
   - inspect `git status`
   - identify every file created by this task
   - promote permanent files to canonical project locations
   - delete all disposable files and task temp directories
   - verify `.tmp-agent/<task-id>/` is removed

6. A task is NOT complete if agent-created temporary files remain.

7. Never delete pre-existing user files merely because they look temporary.

8. If cleanup fails, report it explicitly with exact paths.

9. If a temporary artifact must survive for handoff, place it under
   `.tmp-agent/<task-id>/handoff/` and explicitly list it in the final report.

10. Keep repository root clean. No orphan files, numbered experiments,
    duplicate plans, ad-hoc exports, or unnamed scratch files.