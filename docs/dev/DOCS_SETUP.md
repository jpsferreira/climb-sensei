# Documentation Setup Summary

## What Was Created

### 1. MkDocs Configuration (`mkdocs.yml`)

- Configured Material theme with dark/light mode support
- Set up navigation with 8 main sections
- Enabled mkdocstrings for automatic API documentation
- Added code highlighting, math support (KaTeX), and search functionality
- Configured GitHub integration and social links

### 2. Documentation Files (`docs/`)

#### `docs/index.md` - Homepage

- Project overview
- Feature highlights
- Quick example code
- Project structure
- Requirements

#### `docs/installation.md` - Installation Guide

- uv installation (recommended)
- pip installation
- Source installation
- Development setup
- Dependency list

#### `docs/quickstart.md` - Quick Start Tutorial

- CLI usage examples
- Basic Python API usage
- Demo instructions
- Links to detailed guides

#### `docs/usage.md` - Usage Guide

- Complete ClimbingAnalysis documentation
- All available metrics explained
- Video processing examples
- Pose detection usage
- Visualization techniques
- Biomechanics functions
- Complete pipeline example

#### `docs/metrics.md` - Metrics Reference

- Detailed documentation of all 25+ metrics
- Core movement metrics
- Efficiency & technique metrics
- Joint angle documentation
- Summary statistics
- Time-series history
- Usage examples

#### `docs/api.md` - API Reference

- Auto-generated API docs using mkdocstrings
- ClimbingAnalysis class
- PoseEngine class
- VideoReader/VideoWriter classes
- Biomechanics functions
- Visualization functions
- Configuration module

#### `docs/architecture.md` - Architecture Overview

- Design principles
- Module responsibilities
- Data flow diagrams
- Testing strategy
- Design patterns used
- Extensibility guide
- Performance considerations

#### `docs/development.md` - Development Guide

- Setup instructions
- Makefile target documentation
- Code quality tools (ruff, black, pre-commit)
- Testing guide
- Documentation building
- CI/CD workflows
- Release process
- Code style guidelines
- Troubleshooting
- Contributing checklist

### 3. Simplified README.md

**Before**: 308 lines with extensive documentation
**After**: 120 lines focused on:

- Quick overview with badges
- Key features
- Quick start (installation, CLI, API)
- Links to full documentation
- Development basics
- Requirements

### 4. Updated Dependencies (`pyproject.toml`)

Added to `dev` dependencies:

- `mkdocs>=1.6.0` - Documentation generator
- `mkdocs-material>=9.5.0` - Material theme
- `mkdocstrings[python]>=0.26.0` - API doc generator

### 5. Makefile Targets

Added documentation commands:

- `make docs` - Serve documentation locally at <http://127.0.0.1:8000>
- `make docs-build` - Build static site to `site/`
- `make docs-deploy` - Deploy to GitHub Pages

### 6. GitHub Actions Workflow (`.github/workflows/docs.yaml`)

Automatically deploys documentation to GitHub Pages when:

- Changes to `docs/` directory
- Changes to `mkdocs.yml`
- Push to main branch
- Manual workflow dispatch

## How to Use

### Local Development

```bash
# Serve docs locally (auto-reload on changes)
make docs
# or
uv run mkdocs serve

# Open browser to http://127.0.0.1:8000
```

### Build Documentation

```bash
# Build static site
make docs-build
# or
uv run mkdocs build

# Output in site/ directory
```

### Deploy to GitHub Pages

```bash
# Manual deploy
make docs-deploy
# or
uv run mkdocs gh-deploy --force

# Or push to main (automatic via GitHub Actions)
git push origin main
```

## Documentation URLs

After deployment, documentation will be available at:

- **Main Site**: <https://jpsferreira.github.io/climb-sensei>
- **Installation**: <https://jpsferreira.github.io/climb-sensei/installation/>
- **Quick Start**: <https://jpsferreira.github.io/climb-sensei/quickstart/>
- **Usage**: <https://jpsferreira.github.io/climb-sensei/usage/>
- **Metrics**: <https://jpsferreira.github.io/climb-sensei/metrics/>
- **API**: <https://jpsferreira.github.io/climb-sensei/api/>
- **Architecture**: <https://jpsferreira.github.io/climb-sensei/architecture/>
- **Development**: <https://jpsferreira.github.io/climb-sensei/development/>

## Benefits

### For Users

- Clean, focused README for quick onboarding
- Comprehensive documentation with search
- Beautiful Material Design theme
- Responsive (works on mobile)
- Dark/light mode support

### For Developers

- Easy to maintain (Markdown files)
- Automatic API docs from docstrings
- Version controlled documentation
- Automatic deployment
- Local preview with hot reload

### For Contributors

- Clear development guide
- Architecture documentation
- Contributing guidelines
- Code style reference

## Next Steps

1. **Deploy**: Push to main to trigger automatic deployment
2. **Review**: Check the live site at <https://jpsferreira.github.io/climb-sensei>
3. **Iterate**: Update docs as needed (they auto-deploy on push)
4. **Share**: Update README badges once docs are live

## Maintenance

### Adding New Pages

1. Create `.md` file in `docs/`
2. Add to `nav:` section in `mkdocs.yml`
3. Commit and push (auto-deploys)

### Updating API Docs

- Mkdocstrings automatically generates from docstrings
- Update docstrings in source code
- Rebuild docs to see changes

### Customizing Theme

Edit `mkdocs.yml`:

- `theme.palette` - Colors
- `theme.features` - UI features
- `markdown_extensions` - Markdown features
- `extra` - Social links, analytics, etc.

## File Structure

```
climb-sensei/
├── mkdocs.yml                  # MkDocs configuration
├── docs/                       # Documentation source
│   ├── index.md               # Homepage
│   ├── installation.md        # Install guide
│   ├── quickstart.md          # Quick start
│   ├── usage.md               # Usage guide
│   ├── metrics.md             # Metrics reference
│   ├── api.md                 # API docs
│   ├── architecture.md        # Architecture
│   └── development.md         # Dev guide
├── site/                       # Built docs (gitignored)
├── .github/workflows/
│   └── docs.yaml              # Auto-deploy workflow
└── README.md                   # Simplified (120 lines)
```

## Verification

Documentation builds successfully:

- ✅ No errors or warnings
- ✅ All mkdocstrings references valid
- ✅ All internal links work
- ✅ Math rendering configured
- ✅ Code highlighting enabled
- ✅ Search functionality enabled
