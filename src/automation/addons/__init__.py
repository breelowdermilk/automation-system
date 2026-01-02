"""
Addon Processor Loader - Discovers and loads custom processors from ~/.automation/addons/
"""

import importlib.util
import logging
from pathlib import Path
from typing import Dict, Type, Any, List

logger = logging.getLogger(__name__)


class AddonLoader:
    """
    Loads addon processors from the user's addons directory.

    Each addon is a Python file containing a processor class with:
    - PROCESSOR_NAME: str attribute (optional, defaults to filename)
    - process_batch(files: List[QueuedFile]) -> List[Dict]: async method
    """

    ADDONS_DIR = Path.home() / ".automation" / "addons"

    def __init__(self):
        self.processors: Dict[str, Type] = {}
        self.processor_instances: Dict[str, Any] = {}

    def discover_addons(self) -> Dict[str, Type]:
        """
        Scan addons directory and load processor classes.

        Returns:
            Dict mapping processor names to their classes
        """
        # Ensure addons directory exists
        if not self.ADDONS_DIR.exists():
            self.ADDONS_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created addons directory: {self.ADDONS_DIR}")
            return {}

        # Scan for Python files
        for py_file in self.ADDONS_DIR.glob("*.py"):
            if py_file.name.startswith("_"):
                continue

            try:
                self._load_addon(py_file)
            except Exception as e:
                logger.error(f"Failed to load addon {py_file.name}: {e}")

        if self.processors:
            logger.info(f"Loaded {len(self.processors)} addon(s): {list(self.processors.keys())}")

        return self.processors

    def _load_addon(self, path: Path):
        """
        Load a single addon file and extract processor class.

        Looks for classes that:
        1. Have 'Processor' in the name, OR
        2. Have a PROCESSOR_NAME attribute, OR
        3. Have a process_batch method
        """
        # Load module from file
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load spec for {path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find processor class
        for name, obj in vars(module).items():
            if not isinstance(obj, type):
                continue

            # Check if it looks like a processor
            is_processor = (
                'Processor' in name or
                hasattr(obj, 'PROCESSOR_NAME') or
                hasattr(obj, 'process_batch')
            )

            if is_processor and hasattr(obj, 'process_batch'):
                # Get processor name
                processor_name = getattr(obj, 'PROCESSOR_NAME', path.stem)
                self.processors[processor_name] = obj
                logger.info(f"Loaded addon processor: {processor_name} from {path.name}")
                return

        logger.warning(f"No processor class found in {path.name}")

    def create_processor(
        self,
        processor_name: str,
        router: Any,
        db: Any,
        config: Dict[str, Any],
        **kwargs
    ) -> Any:
        """
        Create an instance of an addon processor.

        Args:
            processor_name: Name of the processor to instantiate
            router: ModelRouter instance
            db: Database instance
            config: Configuration dict
            **kwargs: Additional arguments (output_path, etc.)

        Returns:
            Processor instance or None if not found
        """
        if processor_name not in self.processors:
            return None

        processor_class = self.processors[processor_name]

        # Try different constructor signatures
        try:
            # Full signature with output_path
            return processor_class(
                router=router,
                db=db,
                config=config,
                **kwargs
            )
        except TypeError:
            pass

        try:
            # Basic signature
            return processor_class(router, db, config)
        except TypeError:
            pass

        try:
            # Minimal signature
            return processor_class(router, db)
        except TypeError as e:
            logger.error(f"Cannot instantiate {processor_name}: {e}")
            return None

    def get_processor_names(self) -> List[str]:
        """Get list of available addon processor names."""
        return list(self.processors.keys())
