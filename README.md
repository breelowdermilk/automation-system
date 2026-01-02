# Automation System

macOS folder automation using **launchd + AppleScript**, with AI-powered file processing via Claude.

## Active Automations

| Watcher | Folder | Trigger | Action |
|---------|--------|---------|--------|
| **Screenshots** | `~/Desktop` | `Screenshot*.png` | Claude Sonnet renames to descriptive names |
| **Downloads** | `~/Downloads` | `*.heic` | Converts to JPG, moves to Desktop |
| **Voice Memos** | Voice Memos folder | `*.m4a` | Whisper transcribe → Claude categorize → Obsidian note |

## Quick Start

```bash
# Check running automations
launchctl list | grep automation

# View screenshot processing log
tail -f /tmp/launchd-screenshots.log

# Open web control panel
automation web
```

## Architecture

We use **launchd + AppleScript** instead of a Python daemon:

- **launchd** watches folders via `WatchPaths`
- **AppleScript/Finder** lists files (bypasses macOS security restrictions)
- **Claude CLI** processes files with AI

### Why AppleScript?

macOS security blocks `find` and `ls` from accessing Desktop/Downloads/Documents when run from launchd ("Operation not permitted"). **AppleScript/Finder bypasses this** because it has different permission handling.

## File Locations

### LaunchAgents (auto-start on login)
```
~/Library/LaunchAgents/
├── com.automation.screenshots.plist
├── com.automation.downloads.plist
└── com.automation.voicememos.plist
```

### Watch Scripts
```
~/.automation/scripts/
├── watch-screenshots.sh
├── watch-downloads.sh
└── watch-voicememos.sh
```

### Config
```
~/.automation/config.yaml
```

### Logs
```
/tmp/launchd-screenshots.log
/tmp/launchd-downloads.log
/tmp/launchd-voicememos.log
```

## Management Commands

```bash
# Check running automations
launchctl list | grep automation

# View logs
tail -f /tmp/launchd-screenshots.log

# Stop an automation
launchctl unload ~/Library/LaunchAgents/com.automation.screenshots.plist

# Restart an automation
launchctl unload ~/Library/LaunchAgents/com.automation.screenshots.plist
launchctl load ~/Library/LaunchAgents/com.automation.screenshots.plist

# Reprocess files (clear tracking)
rm /tmp/processed-screenshots.txt
```

## Configuration

### Change Screenshot Model

Edit `~/.automation/scripts/watch-screenshots.sh`:
```bash
CLAUDE_MODEL=sonnet  # or haiku
```

### Model Comparison

| Model | Speed | Quality |
|-------|-------|---------|
| Haiku | ~8 sec | Good |
| Sonnet | ~10 sec | Good |

Both produce quality names. **Sonnet is default.**

## CLI Commands

```bash
automation process screenshot ~/Desktop/file.png  # Process single file
automation status                                  # Show today's stats
automation logs                                    # View recent logs
automation web                                     # Start control panel
automation web --native                            # Native macOS window
```

## Web Control Panel

```bash
automation web           # Browser mode at http://localhost:8420
automation web --native  # Native macOS window (pywebview)
```

Features:
- **Dashboard**: Today's stats, recent activity
- **Models**: Configure models per task type
- **Settings**: Manage watched folders
- **Logs**: View processing history

## Project Structure

```
automation-system/
├── src/automation/
│   ├── cli.py              # CLI with `automation process` command
│   ├── processors.py       # Screenshot, Audio, Inbox processors
│   ├── model_router.py     # Model routing (Claude, OpenRouter, local)
│   └── web/                # Control panel (Flask + HTMX)
├── vendor/
│   └── claude-image-renamer/
│       └── claude-image-renamer.sh  # AI renaming script
└── README.md
```

## Skill for Creating New Automations

A reusable skill is installed at:
```
~/.claude/skills/macos-folder-automation/
```

Use this to set up new watchers following the same launchd + AppleScript pattern.

---

## Development Notes (2026-01-01)

### What We Built
1. Screenshot renaming with Claude (launchd + AppleScript)
2. HEIC→JPG conversion for Downloads
3. Voice memo transcription pipeline
4. Web control panel with native window option
5. `automation process` CLI command for Hazel/manual use
6. Automation skill for future use

### Approaches Tried
1. ~~Python watchdog daemon~~ - Unreliable, went stale
2. ~~Hazel~~ - Couldn't get it to trigger consistently
3. **launchd + AppleScript** ✓ - Works reliably

### Key Discoveries
- macOS blocks `find`/`ls` from launchd → use AppleScript instead
- Hazel was unreliable for our use case
- launchd WatchPaths + AppleScript Finder = reliable combo
- Claude CLI works but is slow (~8-10 sec per image)

### TODO
- [ ] Test voice memos automation end-to-end
- [ ] Test HEIC conversion end-to-end
- [ ] Web UI improvements

## License

MIT
