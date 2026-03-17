---
name: issue-worktree-orchestrator
description: Create and execute an issue-driven development workflow in an isolated git worktree. Use when the user provides a GitHub issue link and asks to solve it safely: create a `<type>/<issue-id>-<slug>` branch in a dedicated worktree, run affected-module build and test gates before push, open a PR against `dev` with a standard template, and keep the sibling Pluck-It.wiki repo updated.
---

# Issue Worktree Orchestrator

Use this skill when the user provides a GitHub issue link and asks for code changes to be completed.

Primary requirements:

- create a dedicated worktree for the issue branch,
- enforce a branch naming format of `<type>/<issue-id>-<branch-name>`,
- never push before module build and tests pass,
- create a PR to `dev`,
- and update the wiki repository (`../Pluck-It.wiki` by default).
- close the worktree once PR creation succeeds (or report why it was kept open).

## 0) Inputs

Required:
- `ISSUE_URL`: GitHub issue URL (`https://github.com/org/repo/issues/123`)

Optional:
- `BASE_BRANCH` (default: `dev`)
- `WORKTREE_BASE` (default: `<repo>/./worktrees`)
- `WIKI_PATH` (default: `<repo>/../Pluck-It.wiki`)
- `BRANCH_TYPE` (optional: `feature`, `bugfix`, `chore`, `refactor`, `ci`, `docs`, `hotfix`, `test`)

### 0a) Wiki page selection policy
- Find and update the most specific existing wiki page; do not create synthetic `Issue-<topic>` files.
- Run in priority order:
  1. Use changed files from `git diff --name-only` after implementation.
  2. Map touched domains:
     - `PluckIt.Functions` / .NET API files -> `backend/dotnet` markdown pages (prefer explicit match by filename tokens).
       - `StylistFunctions.cs` -> `backend/dotnet/Dotnet-Stylist-Functions.md`
       - `CollectionFunctions.cs` -> `backend/dotnet/Dotnet-Collection-Functions.md`
       - `AuthFunctions.cs` -> `backend/dotnet/Dotnet-Auth-Functions.md`
       - `ProfileFunctions.cs` -> `backend/dotnet/Dotnet-Profile-Functions.md`
       - `WardrobeFunctions.cs` -> `backend/dotnet/Dotnet-Wardrobe-Functions.md`
     - `PluckIt.Processor` / Python backend files -> `backend/python` markdown pages (prefer explicit match by filename tokens).
       - `scraper`/`crawl`/`source` -> `backend/python/Python-Scraper-Functions.md`
       - `mood` -> `backend/python/Python-Moods-Functions.md`
       - `digest` -> `backend/python/Python-Digest-and-Analytics-Functions.md`
       - `chat`/`memory` -> `backend/python/Python-Chat-and-Memory-Functions.md`
       - `media` -> `backend/python/Python-Media-and-Metadata-Functions.md`
       - `auth` -> `backend/python/Python-Auth-Functions.md`
       - `health` -> `backend/python/Python-Health-Functions.md`
       - `serverless`/`hosted`/`trigger` -> `backend/python/Python-Serverless-Functions.md`
     - `PluckIt.Client` or Angular `src/app` files -> `frontend/PluckIt-Client.md`.
     - auth/session/token/security changes -> `security/Authentication-and-Identity.md`.
  3. Score existing wiki files under `WIKI_PATH` by overlap with:
     - touched file names/path segments,
     - issue title/slug tokens,
     - labels from `ISSUE_URL`.
  4. Pick the highest-scoring page; append an in-page section for this issue.
  5. If no page is above threshold, append to `Home.md` under a "Recent changes" section.

## 1) Resolve issue metadata

1. Validate tooling:
- `git status --short` (clean preferred, but not required if user is aware),
- `gh --version`,
- `git rev-parse --show-toplevel`.

2. Resolve issue details from URL:
- call `gh issue view <ISSUE_URL> --json number,title,body,url,labels`.
- use `number` as `ISSUE_ID`.
- build branch slug from title:
  - lowercase,
  - spaces and punctuation -> `-`,
  - collapse repeated `-`,
  - trim leading/trailing `-`.

3. Resolve branch type:
- set `BRANCH_TYPE` if the user provided one,
- else infer from issue title/labels:
  - label/title includes `bug`, `bugfix`, `fix` → `bugfix`
  - includes `refactor` → `refactor`
  - includes `chore` → `chore`
  - includes `ci` or `build` → `ci`
  - includes `docs` → `docs`
  - includes `test` → `test`
- default to `feature` when no signal exists,
- normalize invalid values to `feature`.
4. Build branch name:
- `<branch_type>/<ISSUE_ID>-<slug>`.

## 2) Create isolated worktree branch

1. Ensure a clean worktree base:
- create directory `<repo>/.worktrees` if missing.

2. Ensure base branch exists locally:
- prefer local `BASE_BRANCH`,
- fetch remote if needed.

3. Add worktree with issue branch:
- `git worktree add <worktree-path> -b <branch> <BASE_BRANCH>`
- `<branch>` must be exactly `<branch_type>/<ISSUE_ID>-<slug>`.

4. Enter the worktree and switch to the new branch before any edits.

5. Never edit files in the original repo while the worktree is used.

## 3) Implement the issue

1. Read the issue context again inside the worktree and capture acceptance criteria.
2. Make code changes in the worktree only.
3. Update/add tests in the affected module.
4. Keep each commit focused; include module-specific changes only.

## 4) Determine affected module and run required checks

Before any commit:

1. Inspect changed files (`git diff --name-only` in worktree).
2. Map touched paths to module build/test commands.
3. Run build command(s) and test command(s).  

If module commands are not known:
- ask for exact commands before proceeding, or
- infer from repo conventions where possible, then confirm with the user.

Never proceed to push when either build or tests fail.

### Common module defaults (only if repo conventions match)

- Python module: `python -m pytest`
- Node module: `npm run build` then `npm test`
- Go module: `go test ./...`

## 5) Commit and pre-push validation

1. Re-run checks after final edits.
2. Record exact commands and outputs (pass/fail).
3. Commit only after clean evidence that:
- affected-module build passes,
- affected-module tests pass fully.
4. Push with upstream tracking:
- `git push -u origin <branch>`.

## 6) Open PR to `dev`

Create PR only after validated checks pass.

Use this command flow immediately after push:

```bash
PR_TITLE="$(printf "%s: %s" "$BRANCH_TYPE" "$SHORT_DESCRIPTION")"
PR_BODY="$(cat <<'EOF'
## Summary
- Issue: <ISSUE_URL>
- Branch: <branch_type>/<ISSUE_ID>-<slug>

## What changed
- ...

## Validation
- Build: `<command>` ✅ pass
- Tests: `<command>` ✅ pass

## Files changed
- ...

## Wiki sync
- Wiki path used: <WIKI_PATH>
- Updated files: ...
EOF
)"
PR_URL="$(gh pr create --base dev --head "<branch_type>/<ISSUE_ID>-<slug>" --title "$PR_TITLE" --body "$PR_BODY")"
```

If the PR is not created, do not run cleanup and report the exact PR command error.

Use this template body:

```markdown
## Summary
- Issue: <ISSUE_URL>
- Branch: <branch_type>/<ISSUE_ID>-<slug>

## What changed
- ...

## Validation
- Build: `<command>` ✅ pass
- Tests: `<command>` ✅ pass

## Files changed
- ...

## Wiki sync
- Wiki path used: <WIKI_PATH>
- Updated files: ...
```

Create PR:
- base: `dev`
- head: branch name

## 7) Wiki updates

After PR creation (or immediately before merge, if easier for team process):

1. open the wiki repo at `WIKI_PATH` (default sibling `../Pluck-It.wiki`),
2. pull latest changes,
3. determine destination page using 0a) Wiki page selection policy,
4. add/update a section for the issue containing:
   - branch name,
   - PR link,
   - summary,
   - validation commands/results,
   - files changed.
5. commit and push wiki changes.

Keep this update idempotent and short; do not overwrite unrelated wiki history.

## 8) Close issue worktree

After PR URL is captured successfully:

1. Remove the worktree directory:

```bash
git worktree remove --force "<worktree-path>"
```

2. Verify it is gone:

```bash
git worktree list | rg "<worktree-path>"
```

3. If removal fails, keep worktree for follow-up and report the blocker.

## 9) Completion report

Respond with:
- worktree path,
- branch name,
- build/test commands and pass status,
- PR URL,
- wiki commit URL (if created),
- and any follow-up tasks.

If any required gate fails, do not create PR or push. Fix and re-run all gates.

