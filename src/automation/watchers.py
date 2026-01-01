"""
File Watchers - Monitor directories for new files with batching support
"""

import asyncio
import hashlib
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileCreatedEvent
import threading
import logging

logger = logging.getLogger(__name__)


@dataclass
class QueuedFile:
    path: str
    added_at: datetime
    file_hash: Optional[str] = None


@dataclass
class BatchConfig:
    enabled: bool = True
    max_batch_size: int = 10
    max_wait_seconds: int = 60


class BatchingFileHandler(FileSystemEventHandler):
    """
    File system event handler with batching support.
    Queues files and processes them in batches based on size or time.
    """

    def __init__(
        self,
        processor: Callable[[List[QueuedFile]], None],
        batch_config: BatchConfig,
        file_extensions: Optional[Set[str]] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        super().__init__()
        self.processor = processor
        self.batch_config = batch_config
        self.file_extensions = file_extensions or {".png", ".jpg", ".jpeg", ".gif", ".webp"}
        self.loop = loop

        self.queue: List[QueuedFile] = []
        self.queue_lock = threading.Lock()
        self.timer: Optional[threading.Timer] = None
        self.processed_hashes: Set[str] = set()  # Avoid reprocessing

    def on_created(self, event):
        if event.is_directory:
            return

        # Check extension
        ext = Path(event.src_path).suffix.lower()
        if self.file_extensions and ext not in self.file_extensions:
            return

        # Small delay to ensure file is fully written
        time.sleep(0.5)

        # Check if file still exists (might have been a temp file)
        if not os.path.exists(event.src_path):
            return

        # Compute hash to avoid duplicates
        file_hash = self._compute_hash(event.src_path)
        if file_hash in self.processed_hashes:
            logger.debug(f"Skipping duplicate file: {event.src_path}")
            return

        queued = QueuedFile(
            path=event.src_path,
            added_at=datetime.now(),
            file_hash=file_hash,
        )

        with self.queue_lock:
            self.queue.append(queued)
            logger.info(f"Queued file: {event.src_path} (queue size: {len(self.queue)})")

            if not self.batch_config.enabled:
                # Process immediately if batching disabled
                self._process_batch()
            elif len(self.queue) >= self.batch_config.max_batch_size:
                # Process if batch is full
                self._process_batch()
            elif len(self.queue) == 1:
                # Start timer for first item in batch
                self._start_timer()

    def _compute_hash(self, path: str) -> str:
        """Compute MD5 hash of file for deduplication."""
        try:
            with open(path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""

    def _start_timer(self):
        """Start or reset the batch timer."""
        if self.timer:
            self.timer.cancel()

        self.timer = threading.Timer(
            self.batch_config.max_wait_seconds,
            self._timer_callback,
        )
        self.timer.start()

    def _timer_callback(self):
        """Called when batch timer expires."""
        with self.queue_lock:
            if self.queue:
                self._process_batch()

    def _process_batch(self):
        """Process all queued files."""
        if not self.queue:
            return

        # Cancel timer if running
        if self.timer:
            self.timer.cancel()
            self.timer = None

        # Get items and clear queue
        items = self.queue.copy()
        self.queue = []

        # Track hashes to avoid reprocessing
        for item in items:
            if item.file_hash:
                self.processed_hashes.add(item.file_hash)

        # Keep hash set from growing too large
        if len(self.processed_hashes) > 10000:
            self.processed_hashes = set(list(self.processed_hashes)[-5000:])

        logger.info(f"Processing batch of {len(items)} files")

        # Call processor
        try:
            if self.loop:
                asyncio.run_coroutine_threadsafe(
                    self._async_process(items), self.loop
                )
            else:
                self.processor(items)
        except Exception as e:
            logger.error(f"Error processing batch: {e}")

    async def _async_process(self, items: List[QueuedFile]):
        """Async wrapper for processor if it's a coroutine."""
        if asyncio.iscoroutinefunction(self.processor):
            await self.processor(items)
        else:
            self.processor(items)

    def stop(self):
        """Stop the handler and process remaining items."""
        if self.timer:
            self.timer.cancel()
        with self.queue_lock:
            if self.queue:
                self._process_batch()


class DirectoryWatcher:
    """
    Watches a directory for new files.
    """

    def __init__(
        self,
        path: str,
        handler: FileSystemEventHandler,
        recursive: bool = False,
    ):
        self.path = Path(path).expanduser()
        self.handler = handler
        self.recursive = recursive
        self.observer: Optional[Observer] = None

    def start(self):
        """Start watching the directory."""
        if not self.path.exists():
            logger.warning(f"Watch path does not exist: {self.path}")
            self.path.mkdir(parents=True, exist_ok=True)

        self.observer = Observer()
        self.observer.schedule(self.handler, str(self.path), recursive=self.recursive)
        self.observer.start()
        logger.info(f"Started watching: {self.path}")

    def stop(self):
        """Stop watching the directory."""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            logger.info(f"Stopped watching: {self.path}")

        # Also stop the handler to process remaining items
        if hasattr(self.handler, "stop"):
            self.handler.stop()


class WatcherManager:
    """
    Manages multiple directory watchers.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.watchers: List[DirectoryWatcher] = []
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the event loop for async processing."""
        self.loop = loop

    def add_watcher(
        self,
        path: str,
        processor: Callable,
        batch_config: Optional[BatchConfig] = None,
        file_extensions: Optional[Set[str]] = None,
        recursive: bool = False,
    ) -> DirectoryWatcher:
        """Add a new directory watcher."""
        if batch_config is None:
            batch_config = BatchConfig()

        handler = BatchingFileHandler(
            processor=processor,
            batch_config=batch_config,
            file_extensions=file_extensions,
            loop=self.loop,
        )

        watcher = DirectoryWatcher(
            path=path,
            handler=handler,
            recursive=recursive,
        )

        self.watchers.append(watcher)
        return watcher

    def start_all(self):
        """Start all watchers."""
        for watcher in self.watchers:
            watcher.start()

    def stop_all(self):
        """Stop all watchers."""
        for watcher in self.watchers:
            watcher.stop()


# Audio-specific watcher for voice memos
class AudioFileHandler(BatchingFileHandler):
    """
    Specialized handler for audio files.
    Processes immediately by default (no batching).
    """

    def __init__(
        self,
        processor: Callable,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        super().__init__(
            processor=processor,
            batch_config=BatchConfig(enabled=False),  # No batching
            file_extensions={".m4a", ".mp3", ".wav", ".aac", ".ogg"},
            loop=loop,
        )


# Screenshot-specific handler
class ScreenshotFileHandler(BatchingFileHandler):
    """
    Specialized handler for screenshots.
    Batches by default.
    """

    def __init__(
        self,
        processor: Callable,
        batch_config: Optional[BatchConfig] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        if batch_config is None:
            batch_config = BatchConfig(
                enabled=True,
                max_batch_size=10,
                max_wait_seconds=60,
            )

        super().__init__(
            processor=processor,
            batch_config=batch_config,
            file_extensions={".png", ".jpg", ".jpeg", ".gif", ".webp", ".tiff"},
            loop=loop,
        )

    def on_created(self, event):
        # macOS screenshots start with "Screenshot"
        # Also handle generic image files
        filename = Path(event.src_path).name
        if not filename.startswith("."):  # Ignore hidden/temp files
            super().on_created(event)
