# Automation Control Panel

A local automation system that monitors folders for new files (screenshots, audio memos, inbox items) and processes them using AI models via your Claude Code subscription.

## Features

- **Screenshot Auto-Rename**: Automatically renames screenshots with descriptive names
- **Audio Memo Processing**: Transcribes voice memos and creates Obsidian notes
- **Inbox Processing**: Categorizes and files misc documents
- **Model Flexibility**: Use Claude Code (subscription), OpenRouter (pay-per-use), or local models
- **Quality Auditing**: Periodic Opus reviews to ensure quality
- **Web Control Panel**: Monitor, configure, and A/B test from a browser
- **CLI Tools**: Quick stats, logs, and manual controls

## Installation

```bash
# Clone the repository
git clone https://github.com/breelowdermilk/automation-system.git
cd automation-system

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install the package
pip install -e .

# For audio transcription support
pip install -e ".[whisper]"
```

## Quick Start

```bash
# Run initial setup
automation setup

# Edit configuration (update obsidian_vault path)
automation config

# Start the daemon
automation start

# Open web control panel
automation web
# Then visit http://localhost:8420
```

## CLI Commands

```bash
automation start          # Start watching for files
automation stop           # Stop the daemon
automation status         # Show today's stats
automation logs           # View recent processing
automation stats          # Detailed statistics
automation audit          # Run manual quality audit
automation config         # Edit configuration
automation web            # Start control panel
automation test -m sonnet # Test a model
automation setup          # Initial setup
```

## How It Works

### File Processing Flow

1. File appears on Desktop (screenshot) or in inbox
2. Watchdog detects the new file
3. File is queued (batched for efficiency)
4. When batch is ready (10 files or 60 seconds):
   - Send to Claude Code CLI with --print flag
   - Get descriptive filename
   - Rename file
   - Log to database

### Model Routing

The system uses your Claude Code subscription as primary (no extra cost):

```
Primary: Claude Code CLI (sonnet/haiku/opus)
    ↓ (if fails)
Fallback: OpenRouter API (gemini-flash, gpt-4o-mini)
    ↓ (for text-only)
Local: Ollama (llama, qwen)
```

### Batching

Screenshots are batched to reduce API calls:
- Wait up to 60 seconds OR
- Batch 10 screenshots
- Whichever comes first

Audio is processed immediately (no batching).

## Configuration

Key settings in `~/.automation/config.yaml`:

```yaml
task_defaults:
  screenshot_rename:
    model: sonnet          # Claude Code model
    fallback: gemini-flash # OpenRouter fallback
    batch_size: 10
    batch_wait_seconds: 60

paths:
  obsidian_vault: ~/path/to/your/vault  # UPDATE THIS
```

## Web Control Panel

Access at `http://localhost:8420` after running `automation web`.

Features:
- **Dashboard**: Today's stats, model usage, recent activity
- **Models**: Switch primary/fallback models per task
- **Logs**: View all processing with filters
- **Audits**: Quality scores and improvement suggestions
- **Stats**: Charts and performance data

## Costs

- **Claude Code models**: $0 marginal cost (included in subscription)
- **OpenRouter (if used)**: ~$0.001-0.01 per call depending on model
- **Local models**: $0 (runs on your hardware)

## Project Structure

```
automation-system/
├── src/automation/
│   ├── cli.py           # CLI interface
│   ├── database.py      # SQLite storage
│   ├── main.py          # Main daemon
│   ├── model_router.py  # Model routing logic
│   ├── processors.py    # Task processors
│   ├── quality_audit.py # Quality auditing
│   ├── watchers.py      # File system watchers
│   └── web/
│       ├── app.py       # FastAPI web app
│       └── templates/   # Jinja2 templates
├── docs/                # Design documents
├── pyproject.toml
└── README.md
```

## License

MIT
