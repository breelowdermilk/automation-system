"""
Main Daemon - Orchestrates all automation components
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

from .database import Database
from .model_router import ModelRouter
from .watchers import (
    WatcherManager,
    BatchConfig,
    ScreenshotFileHandler,
    AudioFileHandler,
    QueuedFile,
)
from .processors import ScreenshotProcessor, AudioProcessor, InboxProcessor
from .quality_audit import QualityAuditor, ScheduledAuditor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Path.home() / ".automation" / "automation.log"),
    ],
)
logger = logging.getLogger(__name__)


class AutomationDaemon:
    """
    Main daemon that coordinates all automation components.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.running = False
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        # Initialize components
        db_path = config.get("paths", {}).get("database", "~/.automation/automation.db")
        self.db = Database(str(Path(db_path).expanduser()))

        self.router = ModelRouter(config)

        obsidian_vault = config.get("paths", {}).get("obsidian_vault", "~/Documents/Obsidian")
        obsidian_vault = str(Path(obsidian_vault).expanduser())

        # Processors
        self.screenshot_processor = ScreenshotProcessor(self.router, self.db, config)
        self.audio_processor = AudioProcessor(self.router, self.db, config, obsidian_vault)
        self.inbox_processor = InboxProcessor(self.router, self.db, config, obsidian_vault)

        # Quality auditor
        self.auditor = QualityAuditor(self.router, self.db, config)
        self.scheduled_auditor = ScheduledAuditor(self.auditor, config)

        # Watcher manager
        self.watcher_manager = WatcherManager(config)

    async def run(self):
        """Main run loop."""
        self.running = True
        self.loop = asyncio.get_event_loop()

        # Set up signal handlers
        for sig in (signal.SIGTERM, signal.SIGINT):
            self.loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))

        logger.info("Starting Automation Daemon...")

        # Set up watchers
        self._setup_watchers()

        # Start watchers
        self.watcher_manager.start_all()
        logger.info("File watchers started")

        # Start scheduled auditor
        await self.scheduled_auditor.start()
        logger.info("Scheduled auditor started")

        # Keep running
        try:
            while self.running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass

        logger.info("Automation Daemon stopped")

    async def stop(self):
        """Stop the daemon."""
        logger.info("Stopping Automation Daemon...")
        self.running = False

        # Stop components
        self.watcher_manager.stop_all()
        await self.scheduled_auditor.stop()

    def _setup_watchers(self):
        """Set up file watchers based on config."""
        paths = self.config.get("paths", {})
        batching = self.config.get("batching", {})

        # Screenshot watcher
        screenshot_path = paths.get("watch_screenshots", "~/Desktop")
        screenshot_batch = batching.get("screenshots", {})

        self.watcher_manager.add_watcher(
            path=str(Path(screenshot_path).expanduser()),
            processor=self._process_screenshots,
            batch_config=BatchConfig(
                enabled=screenshot_batch.get("enabled", True),
                max_batch_size=screenshot_batch.get("max_batch_size", 10),
                max_wait_seconds=screenshot_batch.get("max_wait_seconds", 60),
            ),
            file_extensions={".png", ".jpg", ".jpeg", ".gif", ".webp", ".tiff"},
        )
        logger.info(f"Watching screenshots: {screenshot_path}")

        # Audio watcher
        audio_path = paths.get("watch_audio")
        if audio_path:
            self.watcher_manager.add_watcher(
                path=str(Path(audio_path).expanduser()),
                processor=self._process_audio,
                batch_config=BatchConfig(enabled=False),  # Process immediately
                file_extensions={".m4a", ".mp3", ".wav", ".aac"},
            )
            logger.info(f"Watching audio: {audio_path}")

        # Inbox watcher
        inbox_path = paths.get("watch_inbox", "~/Desktop/inbox")
        inbox_batch = batching.get("inbox", {})

        self.watcher_manager.add_watcher(
            path=str(Path(inbox_path).expanduser()),
            processor=self._process_inbox,
            batch_config=BatchConfig(
                enabled=inbox_batch.get("enabled", True),
                max_batch_size=inbox_batch.get("max_batch_size", 5),
                max_wait_seconds=inbox_batch.get("max_wait_seconds", 120),
            ),
            file_extensions={".png", ".jpg", ".pdf", ".txt", ".md"},
        )
        logger.info(f"Watching inbox: {inbox_path}")

        # Set event loop for async processing
        self.watcher_manager.set_event_loop(self.loop)

    async def _process_screenshots(self, files: List[QueuedFile]):
        """Process queued screenshots."""
        try:
            results = await self.screenshot_processor.process_batch(files)

            for result in results:
                if result.get("success"):
                    logger.info(f"Renamed: {result.get('new_name')}")
                else:
                    logger.error(f"Failed: {result.get('error')}")

        except Exception as e:
            logger.error(f"Screenshot processing error: {e}")

    async def _process_audio(self, files: List[QueuedFile]):
        """Process queued audio files."""
        try:
            results = await self.audio_processor.process_batch(files)

            for result in results:
                if result.get("success"):
                    logger.info(f"Created note: {result.get('note_path')}")
                else:
                    logger.error(f"Audio processing failed: {result.get('error')}")

        except Exception as e:
            logger.error(f"Audio processing error: {e}")

    async def _process_inbox(self, files: List[QueuedFile]):
        """Process queued inbox files."""
        try:
            results = await self.inbox_processor.process_batch(files)

            for result in results:
                if result.get("success"):
                    logger.info(f"Processed inbox file: {result.get('original_path')}")
                else:
                    logger.error(f"Inbox processing failed: {result.get('error')}")

        except Exception as e:
            logger.error(f"Inbox processing error: {e}")


def main():
    """Entry point for the daemon."""
    import yaml

    # Load config
    config_path = Path.home() / ".automation" / "config.yaml"

    if not config_path.exists():
        print("Config file not found. Run 'automation setup' first.")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f) or {}

    # Create and run daemon
    daemon = AutomationDaemon(config)

    try:
        asyncio.run(daemon.run())
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
