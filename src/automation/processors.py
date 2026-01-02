"""
Processors - Task-specific processing logic for screenshots, audio, and inbox files
"""

import asyncio
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

from .model_router import ModelRouter, ModelResponse
from .database import Database, ProcessingLogEntry
from .watchers import QueuedFile
from .context_search import VaultContextSearch

logger = logging.getLogger(__name__)


# Prompts
SCREENSHOT_RENAME_PROMPT = """Describe this screenshot in 2-5 words suitable for a filename.
Be specific and descriptive. Use lowercase with hyphens between words.

Examples of good filenames:
- days-of-awe-timeline-research
- home-renovation-paint-samples
- obsidian-workflow-diagram
- slack-conversation-bug-report
- code-error-typescript-null

Return ONLY the filename with no extension, no explanation, no quotes."""


SCREENSHOT_BATCH_PROMPT = """For each screenshot below, provide a 2-5 word filename.
Be specific and descriptive. Use lowercase with hyphens between words.

Return a JSON array with filenames in the same order as the images.
Example: ["project-timeline-draft", "error-log-database", "ui-mockup-header"]

Return ONLY the JSON array, nothing else."""


AUDIO_CATEGORIZE_PROMPT = """Analyze this voice memo transcript and categorize it.

Return JSON with this exact structure:
{{
  "title": "brief descriptive title (3-6 words)",
  "project": "one of: days-of-awe, coven, republic, alignment, slow-burn, home, personal, work, other",
  "summary": "1-2 sentence summary of the key points",
  "action_items": ["list", "of", "todos", "if any"],
  "tags": ["relevant", "obsidian", "tags"]
}}

Transcript:
{transcript}

Return ONLY the JSON, nothing else."""


INBOX_CATEGORIZE_PROMPT = """Analyze this file content and suggest how to organize it.

Return JSON:
{{
  "title": "suggested note title",
  "folder": "suggested Obsidian folder path",
  "summary": "brief summary",
  "tags": ["suggested", "tags"]
}}

Content:
{content}

Return ONLY the JSON."""


TRANSCRIPT_REFINEMENT_PROMPT = """You are refining a voice memo transcription for accuracy.

The original transcript was created by Whisper speech-to-text, which may have:
- Misheard proper nouns and character names
- Missed project-specific terminology
- Made errors on uncommon words

PROJECT CONTEXT (use this to correct errors):
{context}

ORIGINAL TRANSCRIPT:
{transcript}

Please output a CORRECTED version of the transcript, fixing:
1. Proper nouns and character names (match them to names in the context above)
2. Project-specific terminology (use exact terms from context)
3. Obvious mishearings that don't match the context

RULES:
- Only fix clear errors, don't rewrite the content
- Preserve the original meaning and flow
- If unsure, keep the original word

Output ONLY the corrected transcript, nothing else."""


class ScreenshotProcessor:
    """Processes screenshots - renames them with descriptive names using Claude Code CLI."""

    # Path to the claude-image-renamer script
    RENAMER_SCRIPT = Path(__file__).parent.parent.parent / "vendor" / "claude-image-renamer" / "claude-image-renamer.sh"

    def __init__(
        self,
        router: ModelRouter,
        db: Database,
        config: Dict[str, Any],
    ):
        self.router = router
        self.db = db
        self.config = config
        self.task_type = "screenshot_rename"

    async def process_batch(self, files: List[QueuedFile]) -> List[Dict[str, Any]]:
        """Process a batch of screenshots."""
        results = []
        for file in files:
            result = await self._process_single(file)
            results.append(result)
            # Small delay between files
            await asyncio.sleep(1)
        return results

    async def _process_single(self, file: QueuedFile) -> Dict[str, Any]:
        """Process a single screenshot using claude-image-renamer script."""
        start_time = datetime.now()

        result = {
            "original_path": file.path,
            "success": False,
        }

        # Check if file still exists
        if not os.path.exists(file.path):
            result["error"] = "File no longer exists"
            self._log_processing(file, None, success=False, error="File not found")
            return result

        # Call the shell script
        try:
            loop = asyncio.get_event_loop()
            proc_result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    [str(self.RENAMER_SCRIPT), file.path],
                    capture_output=True,
                    text=True,
                    timeout=120,  # 2 minute timeout
                ),
            )

            latency_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            result["latency_ms"] = latency_ms

            if proc_result.returncode == 0:
                # Parse output to find new filename
                output = proc_result.stdout + proc_result.stderr
                new_name = self._extract_new_filename(output, file.path)

                if new_name:
                    result["success"] = True
                    result["new_name"] = new_name
                    result["model"] = "claude-code"
                    logger.info(f"Renamed: {Path(file.path).name} -> {new_name}")
                else:
                    result["error"] = "Could not determine new filename"
                    logger.warning(f"Script succeeded but couldn't parse new name from output")
            else:
                result["error"] = proc_result.stderr[:500] if proc_result.stderr else "Script failed"
                logger.error(f"Rename script failed: {result['error']}")

        except subprocess.TimeoutExpired:
            result["error"] = "Script timed out"
            logger.error(f"Rename script timed out for {file.path}")
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Error running rename script: {e}")

        # Log to database
        self._log_processing(
            file,
            result.get("latency_ms"),
            success=result["success"],
            new_name=result.get("new_name"),
            error=result.get("error"),
        )

        return result

    def _extract_new_filename(self, output: str, original_path: str) -> Optional[str]:
        """Extract the new filename from script output."""
        # The script echoes the new filename at the end
        # Also look for mv commands in the output
        lines = output.strip().split('\n')

        # Check last few lines for the filename
        for line in reversed(lines[-10:]):
            line = line.strip()
            # Skip empty lines and common output
            if not line or line.startswith('Uploading') or line.startswith('Warning'):
                continue
            # Look for .png, .jpg etc
            if any(ext in line.lower() for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp']):
                # Extract just the filename if it's a path
                if '/' in line:
                    return Path(line).name
                return line

        # Check if original file was moved by looking for what exists
        original = Path(original_path)
        if not original.exists():
            # File was moved, try to find it in the same directory
            directory = original.parent
            # Look for recently modified files
            for f in directory.glob(f"*.{original.suffix[1:]}"):
                if f.name != original.name:
                    return f.name

        return None

    def _log_processing(
        self,
        file: QueuedFile,
        latency_ms: Optional[int],
        success: bool,
        new_name: Optional[str] = None,
        error: Optional[str] = None,
    ):
        """Log the processing to database."""
        entry = ProcessingLogEntry(
            task_type=self.task_type,
            file_path=file.path,
            file_hash=file.file_hash,
            model="claude-code",
            backend="claude-cli",
            result=new_name,
            success=success,
            error_message=error,
            latency_ms=latency_ms,
            input_tokens=None,
            output_tokens=None,
            cost_usd=0,  # Claude Code is subscription-based
        )
        self.db.log_processing(entry)


class AudioProcessor:
    """Processes audio files - transcribes and categorizes with optional two-pass refinement."""

    def __init__(
        self,
        router: ModelRouter,
        db: Database,
        config: Dict[str, Any],
        obsidian_vault: str,
    ):
        self.router = router
        self.db = db
        self.config = config
        self.obsidian_vault = Path(obsidian_vault)
        self.whisper_model = config.get("models", {}).get("local", {}).get(
            "whisper_model", "base.en"
        )

        # Two-pass transcription settings
        self.two_pass_enabled = config.get("task_defaults", {}).get(
            "audio_transcription", {}
        ).get("two_pass", True)

        # Context search for refinement
        self.context_search = VaultContextSearch(obsidian_vault)

    async def process_batch(self, files: List[QueuedFile]) -> List[Dict[str, Any]]:
        """Process audio files (typically one at a time)."""
        results = []
        for file in files:
            result = await self._process_single(file)
            results.append(result)
        return results

    async def _process_single(self, file: QueuedFile) -> Dict[str, Any]:
        """Process a single audio file with optional two-pass transcription."""
        result = {
            "original_path": file.path,
            "success": False,
        }

        # Step 1: Transcribe with Whisper
        raw_transcript = await self._transcribe(file.path)
        if not raw_transcript:
            result["error"] = "Transcription failed"
            self._log_processing(file, None, None, success=False, error="Transcription failed")
            return result

        result["raw_transcript"] = raw_transcript
        transcript = raw_transcript

        # Step 1.5: Two-pass refinement (if enabled)
        if self.two_pass_enabled:
            project = self.context_search.detect_project(raw_transcript)
            if project:
                context = self.context_search.get_project_context(project)
                if context:
                    refined = await self._refine_transcript(raw_transcript, context)
                    if refined:
                        transcript = refined
                        result["was_refined"] = True
                        result["detected_project"] = project
                        logger.info(f"Refined transcript for project: {project}")

        result["transcript"] = transcript

        # Step 2: Categorize
        model = self.router.get_model_for_task("audio_categorization")
        response = await self.router.process(
            prompt=AUDIO_CATEGORIZE_PROMPT.format(transcript=transcript),
            model_name=model,
            task_type="audio_categorization",
        )

        if not response.success:
            result["error"] = response.error
            self._log_processing(file, response, transcript, success=False)
            return result

        # Parse JSON response
        try:
            import json
            categorization = json.loads(response.text)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            categorization = self._extract_json(response.text)
            if not categorization:
                result["error"] = "Failed to parse categorization"
                self._log_processing(file, response, transcript, success=False)
                return result

        result["categorization"] = categorization
        result["model"] = response.model
        result["latency_ms"] = response.latency_ms

        # Step 3: Create Obsidian note
        note_path = await self._create_obsidian_note(
            file, transcript, categorization
        )
        result["note_path"] = note_path
        result["success"] = True

        # Log
        self._log_processing(file, response, transcript, success=True, note_path=note_path)

        return result

    async def _transcribe(self, audio_path: str) -> Optional[str]:
        """Transcribe audio using Whisper."""
        try:
            # Try faster-whisper first, fall back to whisper
            loop = asyncio.get_event_loop()

            # Check for faster-whisper
            result = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    [
                        "whisper",
                        audio_path,
                        "--model", self.whisper_model,
                        "--output_format", "txt",
                        "--output_dir", "/tmp",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=300,
                ),
            )

            if result.returncode != 0:
                logger.error(f"Whisper failed: {result.stderr}")
                return None

            # Read the output file
            audio_name = Path(audio_path).stem
            txt_path = f"/tmp/{audio_name}.txt"

            if os.path.exists(txt_path):
                with open(txt_path, "r") as f:
                    transcript = f.read().strip()
                os.remove(txt_path)
                return transcript

            return None

        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return None

    async def _refine_transcript(self, transcript: str, context: str) -> Optional[str]:
        """
        Pass 2: Use Claude to refine transcript with project context.

        Args:
            transcript: Raw transcript from Whisper
            context: Project context (character names, terminology, etc.)

        Returns:
            Refined transcript or None if refinement failed
        """
        try:
            model = self.router.get_model_for_task("transcript_refinement")
            if not model:
                model = "haiku"  # Default to fast model for refinement

            response = await self.router.process(
                prompt=TRANSCRIPT_REFINEMENT_PROMPT.format(
                    context=context,
                    transcript=transcript,
                ),
                model_name=model,
                task_type="transcript_refinement",
            )

            if response.success and response.text:
                refined = response.text.strip()
                # Only use if it's substantially similar (avoid hallucinations)
                if len(refined) > len(transcript) * 0.5 and len(refined) < len(transcript) * 2:
                    return refined
                else:
                    logger.warning("Refined transcript length differs too much, using original")
                    return None
            else:
                logger.warning(f"Transcript refinement failed: {response.error}")
                return None

        except Exception as e:
            logger.error(f"Transcript refinement error: {e}")
            return None

    async def _create_obsidian_note(
        self,
        file: QueuedFile,
        transcript: str,
        categorization: Dict[str, Any],
    ) -> str:
        """Create an Obsidian note from the transcription."""
        # Determine folder
        project = categorization.get("project", "other")
        folder_map = {
            "days-of-awe": "Projects/days-of-awe/voice-memos",
            "coven": "Projects/coven/voice-memos",
            "republic": "Projects/republic/voice-memos",
            "alignment": "Projects/alignment/voice-memos",
            "slow-burn": "Projects/slow-burn/voice-memos",
            "home": "Home/voice-memos",
            "personal": "Personal/voice-memos",
            "work": "Work/voice-memos",
            "other": "Voice-Memos",
        }
        folder = folder_map.get(project, "Voice-Memos")

        # Create folder if needed
        folder_path = self.obsidian_vault / folder
        folder_path.mkdir(parents=True, exist_ok=True)

        # Generate filename
        date_str = datetime.now().strftime("%Y-%m-%d")
        title = categorization.get("title", "voice-memo")
        title_slug = re.sub(r'[^a-z0-9\-]', '', title.lower().replace(" ", "-"))
        filename = f"{date_str}-{title_slug}.md"

        note_path = folder_path / filename

        # Handle collision
        counter = 1
        while note_path.exists():
            note_path = folder_path / f"{date_str}-{title_slug}-{counter}.md"
            counter += 1

        # Build note content
        tags = categorization.get("tags", [])
        tags_str = "\n".join(f"  - {tag}" for tag in tags)

        action_items = categorization.get("action_items", [])
        action_items_str = "\n".join(f"- [ ] {item}" for item in action_items) if action_items else "None"

        content = f"""---
date: {datetime.now().isoformat()}
project: {project}
type: voice-memo
audio_file: "{Path(file.path).name}"
tags:
{tags_str}
---

# {categorization.get("title", "Voice Memo")}

## Summary

{categorization.get("summary", "")}

## Action Items

{action_items_str}

## Full Transcript

{transcript}
"""

        # Write note
        with open(note_path, "w") as f:
            f.write(content)

        logger.info(f"Created note: {note_path}")
        return str(note_path)

    def _extract_json(self, text: str) -> Optional[Dict]:
        """Try to extract JSON from model response."""
        import json

        # Try to find JSON in response
        patterns = [
            r'\{[\s\S]*\}',  # Match {...}
            r'```json\s*([\s\S]*?)\s*```',  # Match ```json ... ```
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    json_str = match.group(1) if '```' in pattern else match.group(0)
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue

        return None

    def _log_processing(
        self,
        file: QueuedFile,
        response: Optional[ModelResponse],
        transcript: Optional[str],
        success: bool,
        error: Optional[str] = None,
        note_path: Optional[str] = None,
    ):
        """Log processing to database."""
        entry = ProcessingLogEntry(
            task_type="audio_transcription",
            file_path=file.path,
            file_hash=file.file_hash,
            model=response.model if response else "whisper",
            backend=response.backend if response else "local",
            result=note_path,
            success=success,
            error_message=error or (response.error if response else None),
            latency_ms=response.latency_ms if response else None,
            input_tokens=response.input_tokens if response else None,
            output_tokens=response.output_tokens if response else None,
            cost_usd=response.cost_usd if response else None,
        )
        self.db.log_processing(entry)


class InboxProcessor:
    """Processes misc inbox files - categorizes and files them."""

    def __init__(
        self,
        router: ModelRouter,
        db: Database,
        config: Dict[str, Any],
        obsidian_vault: str,
    ):
        self.router = router
        self.db = db
        self.config = config
        self.obsidian_vault = Path(obsidian_vault)

    async def process_batch(self, files: List[QueuedFile]) -> List[Dict[str, Any]]:
        """Process inbox files."""
        results = []
        for file in files:
            result = await self._process_single(file)
            results.append(result)
        return results

    async def _process_single(self, file: QueuedFile) -> Dict[str, Any]:
        """Process a single inbox file based on its type."""
        path = Path(file.path)
        ext = path.suffix.lower()

        if ext in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
            # Image - treat like screenshot
            return await self._process_image(file)
        elif ext == ".pdf":
            return await self._process_pdf(file)
        elif ext in {".txt", ".md", ".markdown"}:
            return await self._process_text(file)
        else:
            return {
                "original_path": file.path,
                "success": False,
                "error": f"Unsupported file type: {ext}",
            }

    async def _process_image(self, file: QueuedFile) -> Dict[str, Any]:
        """Process an image file (similar to screenshot)."""
        # Delegate to screenshot processor logic
        model = self.router.get_model_for_task("inbox_processing")

        response = await self.router.process(
            prompt=SCREENSHOT_RENAME_PROMPT,
            image_path=file.path,
            model_name=model,
            task_type="inbox_processing",
        )

        result = {
            "original_path": file.path,
            "success": response.success,
            "type": "image",
        }

        if response.success:
            result["suggested_name"] = response.text.strip()

        return result

    async def _process_pdf(self, file: QueuedFile) -> Dict[str, Any]:
        """Process a PDF file."""
        # Extract text from PDF
        try:
            import pypdf

            with open(file.path, "rb") as f:
                reader = pypdf.PdfReader(f)
                text = ""
                for page in reader.pages[:5]:  # First 5 pages
                    text += page.extract_text() or ""
                text = text[:3000]  # Limit context
        except Exception as e:
            return {
                "original_path": file.path,
                "success": False,
                "error": f"PDF extraction failed: {e}",
            }

        # Categorize
        model = self.router.get_model_for_task("inbox_processing")
        response = await self.router.process(
            prompt=INBOX_CATEGORIZE_PROMPT.format(content=text),
            model_name=model,
            task_type="inbox_processing",
        )

        return {
            "original_path": file.path,
            "success": response.success,
            "type": "pdf",
            "categorization": response.text if response.success else None,
            "error": response.error,
        }

    async def _process_text(self, file: QueuedFile) -> Dict[str, Any]:
        """Process a text file."""
        try:
            with open(file.path, "r") as f:
                content = f.read()[:3000]
        except Exception as e:
            return {
                "original_path": file.path,
                "success": False,
                "error": f"Read failed: {e}",
            }

        model = self.router.get_model_for_task("inbox_processing")
        response = await self.router.process(
            prompt=INBOX_CATEGORIZE_PROMPT.format(content=content),
            model_name=model,
            task_type="inbox_processing",
        )

        return {
            "original_path": file.path,
            "success": response.success,
            "type": "text",
            "categorization": response.text if response.success else None,
            "error": response.error,
        }
