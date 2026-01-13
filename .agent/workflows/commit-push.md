---
description: Commit and push code changes with documentation update
---

# Commit and Push Workflow

This workflow helps commit staged changes and push to origin.

## Steps

// turbo-all

1. Check current git status to see what files changed:
```bash
cd /Users/jinjing/workspace/delorean && git status
```

2. Stage all modified and new files:
```bash
cd /Users/jinjing/workspace/delorean && git add -A
```

3. Review staged changes:
```bash
cd /Users/jinjing/workspace/delorean && git diff --cached --stat
```

4. Commit with a descriptive message (replace MESSAGE with actual commit message):
```bash
cd /Users/jinjing/workspace/delorean && git commit -m "MESSAGE"
```

5. Push to remote origin:
```bash
cd /Users/jinjing/workspace/delorean && git push
```

## Commit Message Guidelines

Use conventional commit format:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation only
- `refactor:` - Code refactoring
- `test:` - Adding/updating tests
- `chore:` - Maintenance tasks

Example: `feat: add new volatility factor to strategy`

## Notes
- Always review `git status` before committing
- Use meaningful commit messages
- Run tests before pushing: `conda run -n quant python -m pytest tests/ -v`
