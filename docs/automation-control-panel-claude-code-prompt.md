# Claude Code Prompt: Automation Control Panel + File Processing System

## Overview

Build me a local automation system with a control panel that lets me monitor, configure, and A/B test different AI models for file processing tasks. The system should use my Claude Code subscription as the primary backend, with OpenRouter as a fallback for testing non-Claude models.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     CONTROL PANEL (Web UI)                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ Dashboard │  │ Model    │  │ Quality  │  │ Automation       │ │
│  │ & Stats   │  │ Switcher │  │ Audits   │  │ Logs             │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MODEL ROUTER                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Claude Code CLI │  │ OpenRouter API  │  │ Local (Ollama)  │  │
│  │ (subscription)  │  │ (pay-per-use)   │  │ (free)          │  │
│  │                 │  │                 │  │                 │  │
│  │ • Sonnet        │  │ • Gemini Flash  │  │ • Llama 3.2     │  │
│  │ • Opus          │  │ • GPT-4o-mini   │  │ • Qwen 2.5      │  │
│  │ • Haiku         │  │ • Claude models │  │                 │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AUTOMATION DAEMONS                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Screenshot      │  │ Audio Memo      │  │ Inbox           │  │
│  │ Processor       │  │ Processor       │  │ Processor       │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DATA LAYER                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ SQLite Database                                              ││
│  │ • Processing logs (file, model, latency, result, tokens)    ││
│  │ • Quality audits (original result vs audit result, score)   ││
│  │ • Model performance stats (accuracy, speed, cost)           ││
│  │ • Configuration (which model for which task)                ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## My Setup

- **Machine**: Mac (Apple Silicon)
- **Claude Code**: Already installed, authenticated with Max subscription
- **Obsidian vault**: [FILL IN PATH]
- **Optional**: Server for local models (can add later)

## Part 1: Model Router

Create a unified interface that can route to different backends:

```python
# model_router.py

from enum import Enum
from dataclasses import dataclass
from typing import Optional
import subprocess
import httpx

class Backend(Enum):
    CLAUDE_CODE = "claude_code"      # Uses subscription
    OPENROUTER = "openrouter"        # Pay per use
    LOCAL = "local"                  # Ollama

class Model(Enum):
    # Claude Code CLI models (subscription)
    SONNET = ("claude_code", "sonnet", "claude-sonnet-4")
    OPUS = ("claude_code", "opus", "claude-opus-4") 
    HAIKU = ("claude_code", "haiku", "claude-haiku-3")
    
    # OpenRouter models (pay per use)
    GEMINI_FLASH = ("openrouter", "gemini-flash", "google/gemini-flash-1.5")
    GPT4O_MINI = ("openrouter", "gpt4o-mini", "openai/gpt-4o-mini")
    CODESTRAL = ("openrouter", "codestral", "mistralai/codestral-latest")
    
    # Local models (free)
    LLAMA = ("local", "llama", "llama3.2")
    QWEN = ("local", "qwen", "qwen2.5:7b")

@dataclass
class ModelResponse:
    text: str
    model: str
    backend: str
    latency_ms: int
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None  # None for subscription/local

class ModelRouter:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        
    async def process(
        self, 
        prompt: str, 
        image_path: Optional[str] = None,
        model: Model = Model.SONNET,
        task_type: str = "default"
    ) -> ModelResponse:
        """
        Route request to appropriate backend.
        If model not specified, uses configured default for task_type.
        """
        pass
    
    def _call_claude_code(self, prompt: str, image_path: str, model: str) -> ModelResponse:
        """
        Shell out to claude CLI with --print flag.
        Uses subscription, no API cost.
        """
        cmd = [
            "claude", "-p", prompt,
            "--print",
            "--output-format", "json",
            "--model", model
        ]
        if image_path:
            # Claude Code can reference files directly
            cmd[2] = f"{prompt}\n\nAnalyze this image: {image_path}"
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        # Parse JSON response for tokens, etc.
        pass
    
    def _call_openrouter(self, prompt: str, image_path: str, model: str) -> ModelResponse:
        """
        Call OpenRouter API. Tracks cost.
        """
        pass
    
    def _call_local(self, prompt: str, model: str) -> ModelResponse:
        """
        Call local Ollama. No vision support.
        """
        pass
```

## Part 2: Configuration System

Store all configuration in a single YAML file that the control panel can edit:

```yaml
# ~/.automation/config.yaml

# Default models for each task type
task_defaults:
  screenshot_rename:
    model: sonnet           # Primary model
    fallback: gemini-flash  # If primary fails/unavailable
    batch_size: 10
    batch_wait_seconds: 60
    
  audio_transcription:
    model: local-whisper    # Always local
    
  audio_categorization:
    model: haiku            # Fast and cheap
    fallback: llama
    
  inbox_processing:
    model: sonnet
    fallback: gemini-flash
    
  quality_audit:
    model: opus             # Use best model for audits
    schedule: "0 */6 * * *" # Every 6 hours
    sample_size: 20         # Audit last 20 items

# A/B testing configuration
ab_tests:
  screenshot_rename:
    enabled: true
    variants:
      - model: sonnet
        weight: 70
      - model: gemini-flash  
        weight: 15
      - model: haiku
        weight: 15
    
# Model-specific settings
models:
  claude_code:
    # No API key needed - uses subscription
    timeout_seconds: 30
    
  openrouter:
    api_key_env: OPENROUTER_API_KEY
    timeout_seconds: 30
    
  local:
    ollama_host: "http://localhost:11434"
    whisper_model: "base.en"  # or medium.en, large-v3

# Paths
paths:
  watch_screenshots: ~/Desktop
  watch_audio: ~/Library/Group Containers/group.com.apple.VoiceMemos.shared/Recordings
  watch_inbox: ~/Desktop/inbox
  obsidian_vault: [FILL IN]
  database: ~/.automation/automation.db

# Batching
batching:
  screenshots:
    enabled: true
    max_batch_size: 10
    max_wait_seconds: 60
  audio:
    enabled: false  # Process immediately
  inbox:
    enabled: true
    max_batch_size: 5
    max_wait_seconds: 120
```

## Part 3: Database Schema

```sql
-- Processing log
CREATE TABLE processing_log (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    task_type TEXT NOT NULL,  -- screenshot_rename, audio_transcription, etc.
    file_path TEXT NOT NULL,
    file_hash TEXT,           -- For deduplication
    
    -- Model info
    model TEXT NOT NULL,
    backend TEXT NOT NULL,    -- claude_code, openrouter, local
    was_ab_test BOOLEAN DEFAULT FALSE,
    
    -- Results
    result TEXT,              -- The actual output
    success BOOLEAN,
    error_message TEXT,
    
    -- Metrics
    latency_ms INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL,            -- NULL for subscription/local
    
    -- For quality tracking
    audit_score REAL,         -- Filled in by quality audit
    audit_notes TEXT
);

-- Quality audits
CREATE TABLE quality_audits (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    processing_log_id INTEGER REFERENCES processing_log(id),
    
    -- Audit model (usually Opus)
    audit_model TEXT NOT NULL,
    
    -- Scores (1-5 scale)
    accuracy_score INTEGER,
    usefulness_score INTEGER,
    
    -- Audit findings
    suggested_improvement TEXT,
    would_rename BOOLEAN,     -- Would auditor rename differently?
    better_name TEXT,         -- If so, what?
    
    -- Meta
    audit_latency_ms INTEGER,
    audit_cost_usd REAL
);

-- Model performance aggregates (materialized view, refreshed periodically)
CREATE TABLE model_stats (
    model TEXT PRIMARY KEY,
    task_type TEXT,
    
    total_requests INTEGER,
    success_rate REAL,
    avg_latency_ms REAL,
    avg_audit_score REAL,
    total_cost_usd REAL,
    
    last_updated DATETIME
);

-- Configuration history (for tracking changes)
CREATE TABLE config_history (
    id INTEGER PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    config_yaml TEXT,
    changed_by TEXT  -- 'user' or 'ab_test_auto'
);
```

## Part 4: Control Panel UI

Build a simple web UI using FastAPI + HTMX (minimal JS, server-rendered):

### Dashboard View
```
┌─────────────────────────────────────────────────────────────────┐
│  AUTOMATION CONTROL PANEL                          [Settings ⚙️] │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  TODAY'S ACTIVITY                                               │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │ Screenshots  │ │ Audio Memos  │ │ Inbox Files  │            │
│  │     23       │ │      4       │ │      7       │            │
│  │  processed   │ │  processed   │ │  processed   │            │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
│                                                                  │
│  MODEL USAGE (THIS WEEK)                                        │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Sonnet (Claude Code)  ████████████████░░░░  847 calls       ││
│  │ Haiku (Claude Code)   ████░░░░░░░░░░░░░░░░  203 calls       ││
│  │ Gemini Flash          ██░░░░░░░░░░░░░░░░░░   89 calls       ││
│  │ Local Whisper         ███████░░░░░░░░░░░░░  312 calls       ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  RECENT PROCESSING                               [View All →]   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ 2 min ago   screenshot   sonnet   "days-of-awe-timeline"    ││
│  │ 5 min ago   screenshot   sonnet   "home-reno-paint-sample"  ││
│  │ 12 min ago  audio        whisper  "republic-melody-idea"    ││
│  │ 15 min ago  screenshot   gemini   "obsidian-workflow-test"  ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  QUALITY SCORES (Last 7 Days)                                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Model          │ Avg Score │ Accuracy │ Speed   │ Cost      ││
│  │ Sonnet         │ 4.2/5     │ 94%      │ 1.2s    │ $0 (sub)  ││
│  │ Gemini Flash   │ 3.8/5     │ 87%      │ 0.8s    │ $0.12     ││
│  │ Haiku          │ 3.9/5     │ 89%      │ 0.6s    │ $0 (sub)  ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  ⚠️  LAST AUDIT: 2 hours ago                    [Run Audit Now] │
│  Found 3 items that could be improved                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Model Switcher View
```
┌─────────────────────────────────────────────────────────────────┐
│  MODEL CONFIGURATION                                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  SCREENSHOT RENAMING                                            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Primary Model:  [Sonnet ▼]                                  ││
│  │ Fallback:       [Gemini Flash ▼]                            ││
│  │                                                              ││
│  │ ☑️ Enable A/B Testing                                        ││
│  │   Sonnet: 70%  Gemini: 15%  Haiku: 15%                      ││
│  │   [Adjust Weights...]                                        ││
│  │                                                              ││
│  │ Batching: ☑️ Enabled                                         ││
│  │   Max batch size: [10]  Wait time: [60] seconds             ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  AUDIO PROCESSING                                               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Transcription:  [Local Whisper ▼] (always local)            ││
│  │ Categorization: [Haiku ▼]                                   ││
│  │ Fallback:       [Local Llama ▼]                             ││
│  │                                                              ││
│  │ ☐ Enable A/B Testing                                        ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  QUALITY AUDITING                                               │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Audit Model:    [Opus ▼]                                    ││
│  │ Schedule:       [Every 6 hours ▼]                           ││
│  │ Sample Size:    [20] items per audit                        ││
│  │                                                              ││
│  │ [Run Manual Audit Now]                                       ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│                              [Save Changes]  [Reset to Defaults] │
└─────────────────────────────────────────────────────────────────┘
```

### Log Viewer
```
┌─────────────────────────────────────────────────────────────────┐
│  PROCESSING LOG                                                  │
├─────────────────────────────────────────────────────────────────┤
│  Filter: [All Types ▼] [All Models ▼] [Last 24h ▼] [Search...] │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ 📸 Screenshot • 2 min ago                                   ││
│  │ Original: Screenshot 2026-01-01 at 2.34.12 PM.png          ││
│  │ Renamed:  days-of-awe-pollack-research-timeline.png        ││
│  │ Model: Sonnet (Claude Code) • 1.2s • A/B: No               ││
│  │ Audit: 4/5 ⭐                                [View Details] ││
│  ├─────────────────────────────────────────────────────────────┤│
│  │ 🎤 Audio Memo • 12 min ago                                  ││
│  │ Original: Recording 47.m4a                                  ││
│  │ Created:  Voice-Memos/2026-01-01-republic-melody-idea.md   ││
│  │ Transcription: Whisper (local) • 3.4s                      ││
│  │ Categorization: Haiku • 0.6s                               ││
│  │ Audit: Not yet audited                      [View Details] ││
│  ├─────────────────────────────────────────────────────────────┤│
│  │ 📸 Screenshot • 15 min ago                    [A/B TEST]   ││
│  │ Original: Screenshot 2026-01-01 at 2.19.44 PM.png          ││
│  │ Renamed:  obsidian-mcp-workflow-diagram.png                ││
│  │ Model: Gemini Flash (OpenRouter) • 0.8s • $0.002           ││
│  │ Audit: 3/5 ⭐ "Could be more specific"      [View Details] ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  Showing 1-20 of 847                        [← Prev] [Next →]   │
└─────────────────────────────────────────────────────────────────┘
```

## Part 5: Quality Auditing System

The audit system periodically uses Opus to review recent processing:

```python
# quality_audit.py

AUDIT_PROMPT = """You are auditing an automated file processing system.

For each item below, evaluate the AI's output on a 1-5 scale:
- 5: Perfect, couldn't be improved
- 4: Good, minor improvements possible
- 3: Acceptable, but notable issues
- 2: Poor, significant problems
- 1: Failed, completely wrong

For screenshot renames, consider:
- Is the name descriptive and specific?
- Would you know what this file contains from the name alone?
- Is it appropriately concise (2-5 words)?

For audio categorization, consider:
- Is the project categorization correct?
- Does the summary capture the key points?
- Are action items properly extracted?

---

ITEMS TO AUDIT:

{items}

---

Return JSON:
{{
  "audits": [
    {{
      "id": <processing_log_id>,
      "accuracy_score": <1-5>,
      "usefulness_score": <1-5>,
      "notes": "<brief explanation>",
      "would_change": <true/false>,
      "suggested_improvement": "<if would_change, what would be better>"
    }}
  ],
  "overall_observations": "<any patterns you notice across items>"
}}
"""

async def run_quality_audit(
    db: Database,
    router: ModelRouter,
    sample_size: int = 20
) -> dict:
    """
    Pull recent unaudited items, send to Opus for review.
    """
    # Get recent unaudited items
    items = await db.get_unaudited_items(limit=sample_size)
    
    # Format for audit
    formatted = format_items_for_audit(items)
    
    # Call Opus
    response = await router.process(
        prompt=AUDIT_PROMPT.format(items=formatted),
        model=Model.OPUS,
        task_type="quality_audit"
    )
    
    # Parse and store results
    results = json.loads(response.text)
    await db.store_audit_results(results)
    
    return results
```

## Part 6: File Watchers with Batching

```python
# watchers.py

import asyncio
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from collections import defaultdict
from datetime import datetime, timedelta

class BatchingFileHandler(FileSystemEventHandler):
    def __init__(
        self, 
        processor_func,
        batch_size: int = 10,
        batch_wait_seconds: int = 60
    ):
        self.processor = processor_func
        self.batch_size = batch_size
        self.batch_wait = batch_wait_seconds
        self.queue = []
        self.queue_lock = asyncio.Lock()
        self.oldest_item_time = None
        
    def on_created(self, event):
        if event.is_directory:
            return
        
        # Add to queue
        asyncio.create_task(self._add_to_queue(event.src_path))
    
    async def _add_to_queue(self, path: str):
        async with self.queue_lock:
            self.queue.append({
                'path': path,
                'added': datetime.now()
            })
            
            if self.oldest_item_time is None:
                self.oldest_item_time = datetime.now()
                # Start the wait timer
                asyncio.create_task(self._wait_and_process())
            
            # Process immediately if batch is full
            if len(self.queue) >= self.batch_size:
                await self._process_batch()
    
    async def _wait_and_process(self):
        """Wait for batch_wait_seconds, then process whatever we have."""
        await asyncio.sleep(self.batch_wait)
        async with self.queue_lock:
            if self.queue:
                await self._process_batch()
    
    async def _process_batch(self):
        """Process all queued items."""
        items = self.queue.copy()
        self.queue = []
        self.oldest_item_time = None
        
        # Process batch
        await self.processor(items)
```

## Part 7: Obsidian Integration

For the Obsidian view, generate a markdown dashboard that auto-updates:

```markdown
---
cssclass: automation-dashboard
---

# 🤖 Automation Dashboard

> Last updated: {{timestamp}}

## Today's Activity

| Task | Count | Success Rate |
|------|-------|--------------|
| Screenshots | {{screenshot_count}} | {{screenshot_success}}% |
| Audio Memos | {{audio_count}} | {{audio_success}}% |
| Inbox Files | {{inbox_count}} | {{inbox_success}}% |

## Model Performance (7 Days)

| Model | Calls | Avg Score | Avg Latency | Cost |
|-------|-------|-----------|-------------|------|
{{#each models}}
| {{name}} | {{calls}} | {{score}}/5 | {{latency}}s | {{cost}} |
{{/each}}

## Recent Items

{{#each recent}}
- **{{type}}** ({{time_ago}}) → `{{result}}` via {{model}}
{{/each}}

## Quick Actions

- [[Run Quality Audit]]
- [[View Full Logs]]
- [[Open Control Panel]]

---

*Control panel: http://localhost:8420*
```

## Deliverables

1. **Core package structure:**
   ```
   ~/.automation/
   ├── config.yaml           # All configuration
   ├── automation.db         # SQLite database
   ├── logs/                  # Log files
   └── src/
       ├── model_router.py   # Model routing logic
       ├── watchers.py       # File system watchers
       ├── processors.py     # Task-specific processing
       ├── quality_audit.py  # Opus-based auditing
       ├── database.py       # DB operations
       ├── web/              # Control panel
       │   ├── app.py        # FastAPI app
       │   ├── templates/    # HTMX templates
       │   └── static/       # CSS
       └── cli.py            # CLI interface
   ```

2. **CLI commands:**
   ```bash
   automation start          # Start all watchers
   automation stop           # Stop all watchers
   automation status         # Show current status
   automation audit          # Run manual quality audit
   automation logs           # View recent logs
   automation config         # Edit configuration
   automation web            # Start control panel (http://localhost:8420)
   automation stats          # Show usage statistics
   automation test <model>   # Run test batch through specific model
   ```

3. **Launchd plist** for running as daemon on Mac

4. **Setup script** that:
   - Creates directory structure
   - Initializes database
   - Creates default config
   - Installs as launchd service

## Questions to Answer Before Building

1. Where is your Obsidian vault?
2. Where do your Voice Memos save? (Usually `~/Library/Group Containers/group.com.apple.VoiceMemos.shared/Recordings`)
3. Do you want the web control panel, or would you prefer everything through Obsidian + CLI?
4. What port for the control panel? (Suggest 8420)
5. Do you have OpenRouter API key ready, or should we start with Claude Code only?

## Build Order

1. **Phase 1: Core infrastructure**
   - Model router with Claude Code CLI support
   - Database setup
   - Configuration system

2. **Phase 2: Screenshot automation**
   - File watcher with batching
   - Screenshot processor
   - Basic logging

3. **Phase 3: Control panel MVP**
   - Dashboard view
   - Model switcher
   - Log viewer

4. **Phase 4: Quality auditing**
   - Opus audit system
   - Scoring and tracking

5. **Phase 5: Audio + Inbox**
   - Whisper integration
   - Audio categorization
   - Inbox processing

6. **Phase 6: A/B testing**
   - Variant routing
   - Statistical tracking
   - Auto-optimization suggestions

---

*Start with Phase 1-2 to get screenshots working, then iterate.*
