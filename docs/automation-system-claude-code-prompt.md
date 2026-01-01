# Claude Code Prompt: Local Automation System with Obsidian Integration

## Overview

Build me a local automation system that monitors folders for new files and processes them intelligently, routing to either local models (for cost savings) or cloud APIs (when vision/quality needed). The processed results should create or update notes in my Obsidian vault.

## My Setup

- **Main machine**: Mac (desktop with screenshots, audio memos)
- **Server**: Always-on Linux server (can run Whisper, Ollama)
- **Obsidian vault location**: [FILL IN YOUR VAULT PATH]
- **Sync mechanism between machines**: [FILL IN: Syncthing/Dropbox/etc., or same machine]

## Architecture Requirements

### 1. Folder Watching Daemon

Create a Python daemon using `watchdog` that monitors these folders:

```
~/Desktop/              → screenshots (*.png, *.jpg)
~/Desktop/inbox/        → misc files to process
[audio memo location]   → voice memos (*.m4a, *.mp3, *.wav)
```

The daemon should:
- Run as a background service (launchd on Mac, systemd on Linux)
- Queue files briefly (2-3 second debounce) to avoid processing partial writes
- Log all activity to a file
- Handle errors gracefully without crashing

### 2. Processing Pipeline

#### Screenshots (Vision Required)
```
New screenshot detected
    → Send to OpenRouter API (try Gemini 1.5 Flash first)
    → Prompt: "Describe this screenshot in 2-3 words suitable for a filename. 
       Be specific and descriptive. Examples: 'react-component-error', 
       'days-of-awe-research-pollack', 'home-renovation-wallpaper-sample',
       'obsidian-workflow-diagram'. Return ONLY the filename, no extension."
    → Rename file with returned name + timestamp suffix if collision
    → Optionally: create Obsidian note if screenshot seems research-related
```

#### Audio Memos (Local Processing)
```
New audio file detected
    → Send to local Whisper (whisper.cpp or faster-whisper)
    → Transcribe to text
    → Send transcript to local Ollama (Llama 3.2 or Qwen 2.5)
    → Prompt: "Categorize this voice memo and extract key points.
       Return JSON: {
         'title': 'brief descriptive title',
         'project': 'one of: days-of-awe, coven, republic, alignment, home, personal, other',
         'summary': '1-2 sentence summary',
         'action_items': ['list', 'of', 'todos'],
         'full_transcript': 'the transcript'
       }"
    → Create Obsidian note from result
```

#### Inbox Files (Hybrid)
```
New file in inbox
    → If image: process like screenshot
    → If PDF: extract text locally (pypdf), summarize with local LLM
    → If text/markdown: categorize with local LLM
    → Create/file Obsidian note appropriately
```

### 3. OpenRouter Integration

```python
# Use OpenRouter for all cloud API calls
# Base URL: https://openrouter.ai/api/v1
# Models to support (in preference order for vision):
#   - google/gemini-flash-1.5 (cheapest, try first)
#   - anthropic/claude-3-haiku (fallback)
#   - anthropic/claude-sonnet-4 (quality fallback)

# Store API key in environment variable: OPENROUTER_API_KEY
```

Create a simple abstraction that lets me easily swap models:

```python
def process_with_llm(prompt, image_path=None, model="google/gemini-flash-1.5"):
    """
    Unified interface for LLM calls.
    Automatically falls back to next model on failure.
    Logs cost per call.
    """
    pass
```

### 4. Local Model Setup (Server)

Provide setup instructions and scripts for:

```bash
# Whisper setup (faster-whisper recommended)
# Should expose a simple HTTP API or socket for the daemon to call

# Ollama setup
# Models to pull: llama3.2, qwen2.5:7b
# Expose on local network for daemon access
```

If my server isn't reachable, fall back to OpenRouter for everything.

### 5. Obsidian Integration

Create notes in these locations based on content:

```
Inbox/                          → unprocessed/uncategorized items
Projects/days-of-awe/research/  → Days of Awe related
Projects/coven/                 → Coven of Phoenix related  
Projects/republic/              → Republic related
Projects/alignment/             → Alignment related
Home/                           → home renovation
Voice-Memos/                    → transcribed audio (with date prefix)
Screenshots/                    → screenshots that warrant notes
```

Note template for voice memos:
```markdown
---
date: {{date}}
project: {{project}}
type: voice-memo
audio_file: {{original_filename}}
---

# {{title}}

## Summary
{{summary}}

## Action Items
{{action_items as checklist}}

## Full Transcript
{{transcript}}
```

Note template for research screenshots:
```markdown
---
date: {{date}}
type: screenshot
original_file: {{filename}}
---

# {{descriptive_title}}

![[{{filename}}]]

## Description
{{AI description if generated}}
```

### 6. Cost Tracking

Create a simple SQLite database or JSON log that tracks:
- Timestamp
- File processed
- Model used
- Tokens/cost
- Processing time

Include a simple CLI command to check usage:
```bash
python automation.py --stats           # show today's usage
python automation.py --stats --week    # show week's usage
```

### 7. Testing Mode

Before running live, I want to test the model quality. Create a test harness:

```bash
python automation.py --test-screenshots ./test-images/
# Processes folder of test images through Flash, Haiku, Sonnet
# Outputs comparison table so I can evaluate quality vs cost
```

### 8. Configuration

All settings in a single config file (YAML or TOML):

```yaml
watch_folders:
  screenshots: ~/Desktop
  audio: ~/Voice Memos  
  inbox: ~/Desktop/inbox

obsidian_vault: /path/to/vault

server:
  host: 192.168.x.x  # or localhost if same machine
  whisper_port: 8001
  ollama_port: 11434

openrouter:
  default_vision_model: google/gemini-flash-1.5
  default_text_model: anthropic/claude-3-haiku
  fallback_models:
    - anthropic/claude-sonnet-4

local_first: true  # prefer local models when possible

file_handling:
  rename_screenshots: true
  create_notes_for_screenshots: false  # only if research-related
  delete_audio_after_processing: false
```

## Deliverables

1. **Main Python package** with clear module structure
2. **Setup script** that installs dependencies, creates config template
3. **Launchd plist** (Mac) for running as daemon
4. **Systemd service file** (Linux) for server components
5. **README** with setup instructions
6. **Test suite** for the processing logic

## Notes

- I use `uv` for Python package management
- Prefer async where it makes sense (file watching + API calls)
- Keep it simple — I want to understand and modify this myself
- Add good logging so I can debug issues
- If something fails, it should notify me (simple desktop notification) rather than silently fail

## Questions to Answer Before Building

1. Where exactly is my Obsidian vault?
2. What's the sync situation between my Mac and server?
3. Do I want to run everything on one machine first, or set up the server immediately?
4. What's my audio memo source? (Voice Memos app, Wispr, other?)

---

*Start by asking me these questions, then build incrementally: get screenshot renaming working first, then add audio, then add the full Obsidian integration.*
