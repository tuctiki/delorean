---
description: Update project documentation (README, docstrings, comments)
---

# Documentation Update Workflow

This workflow guides updating project documentation to reflect current codebase state.

## Key Documentation Files

- `README.md` - Main project documentation
- `delorean/config.py` - Configuration constants (ETF list, model params)
- `delorean/data.py` - Factor definitions and data handlers
- `workflow_live_trading.md` - Live trading operation guide

## Steps

1. Review current README structure:
```bash
cd /Users/jinjing/workspace/delorean && head -100 README.md
```

2. Check current factor definitions for accuracy:
```bash
cd /Users/jinjing/workspace/delorean && grep -A 30 "get_custom_factors" delorean/data.py
```

3. Verify ETF universe is documented:
```bash
cd /Users/jinjing/workspace/delorean && grep -A 20 "ETF_LIST" delorean/config.py
```

4. Check model hyperparameters:
```bash
cd /Users/jinjing/workspace/delorean && grep -A 15 "MODEL_PARAMS" delorean/config.py
```

## Documentation Sections to Update

### README.md
- [ ] Project overview and purpose
- [ ] Factor model descriptions (sync with `data.py`)
- [ ] ETF universe list (sync with `config.py`)
- [ ] Model hyperparameters (sync with `config.py`)
- [ ] Performance metrics (update with latest backtest)
- [ ] Usage instructions

### Docstrings
- [ ] All public functions have docstrings
- [ ] Type hints match actual behavior
- [ ] Examples are up-to-date

## After Updates

Commit documentation changes:
```bash
cd /Users/jinjing/workspace/delorean && git add README.md && git commit -m "docs: update documentation"
```
