# Contributing to Memor

First off, thank you for considering contributing to Memor! It's people like you that make open-source such a great community.

## 1. Where to Start

- **Bug Reports**: If you notice a bug, please open an issue and include the steps to reproduce it.
- **Feature Requests**: We welcome ideas! Open an issue with a detailed explanation of why the feature would be useful and how it aligns with the bitemporal local-first philosophy of the engine.
- **Pull Requests (PRs)**: You are welcome to fork the repository and submit a PR for any open issues or improvements you've discussed with the maintainers.

## 2. Development Setup

1. Fork the repo and clone it locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Memor.git
   cd Memor
   ```
2. Create a fresh virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install the package in editable mode with development dependencies:
   ```bash
   pip install -e ".[dev,mcp]"
   ```

## 3. Architecture Overview

Memor is built on three core pillars:
1. **Local-First SQLite**: All state is persisted in a local `memory.db` file. We use Write-Ahead Logging (WAL) for concurrency.
2. **Bitemporal Tracking**: Facts are never mutated or silently overwritten. When a fact changes, we set `valid_to = NOW()` on the old record and insert a new record with `valid_to = NULL`.
3. **Dual-Path Resolution**: 
   - **Fast Path**: Single-valued predicates (e.g., `lives_in`) are immediately resolved.
   - **Slow Path**: Ambiguous predicates route to the LLM (via `config.py`) to determine if they are additive or contradictory.

If you are modifying the core resolution logic, please ensure you read through `memor/resolver.py` to understand this dual-path system.

## 4. Testing

We have a comprehensive internal test suite.

**Offline Unit Tests** (No API Key Required):
Tests core SQLite mechanics, pruning, and fast-path resolution.
```bash
pytest tests/test_pruning.py -v
```

**LLM Benchmarks** (Requires `GROQ_API_KEY`):
Evaluates the slow-path LLM conflict resolution using our rigorous `BaseTemporalMemoryBenchmark`.
```bash
export GROQ_API_KEY="your_api_key"
pytest tests/ -v
```
All PRs must pass the test suite before merging.

## 5. Style & Linting

- Keep the codebase lightweight. We explicitly avoid heavyweight frameworks (no massive ORMs or excessive dependencies) to ensure Memor remains easily embeddable.
- Ensure any new code has type hints for primary function signatures.
- Document any complex bitemporal logic thoroughly.

Thank you for contributing!
