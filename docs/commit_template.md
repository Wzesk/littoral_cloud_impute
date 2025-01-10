# Commit Message Templates

## Repository Structure Changes

```
refactor: standardize repository structure and tooling

Research Goals:
| Goal | Description |
|------|-------------|
| Goal 1 | Infer the state of pixels obscured by clouds from adjacent clouds (adjacent in space or time) |
| Goal 2 | Infer the state of pixels obscured by clouds from alternative imagery products (comparing imagery from different sources) |
| Goal 3 | Superresolve the shoreline pixels potentially using the timeseries to train, but preserving fine grain changes to the shoreline over time |

Changes:
- Add proper directory structure (src/, docs/, data/, etc.)
- Configure development tools (hatch, pre-commit, commitizen)
- Enhance documentation with research goals
- Fix code quality issues and docstrings
- Add proper type hints and formatting

Closes #1
```

## Feature Addition Template

```
feat(module): add new feature description

Research Goals Addressed:
| Goal | Implementation |
|------|----------------|
| Goal X | How this feature contributes to the goal |

Changes:
- Detailed change 1
- Detailed change 2
- Impact on existing functionality

Related to #issue_number
```

## Bug Fix Template

```
fix(module): fix specific issue description

Impact:
- What was fixed
- Why it needed fixing
- How it affects functionality

Fixes #issue_number
