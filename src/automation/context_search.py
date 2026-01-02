"""
Obsidian Vault Context Search - Find project-specific context for transcription refinement

Searches the user's Obsidian vault to find:
- Character names
- Project terminology
- Proper nouns
- Other context useful for refining voice memo transcriptions
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# Project detection patterns - keywords that indicate which project a voice memo is about
PROJECT_PATTERNS = {
    "days-of-awe": [
        r"\bdays\s*of\s*awe\b",
        r"\bjonah\b",
        r"\brobbi\b",
        r"\bcantor\b",
        r"\byom\s*kippur\b",
        r"\bkol\s*nidre\b",
        r"\bsynagogue\b",
        r"\bhigh\s*holidays?\b",
        r"\brosh\s*hashanah\b",
    ],
    "alignment": [
        r"\balignment\b",
        r"\bai\s*ethics\b",
        r"\bvalue\s*alignment\b",
        r"\bartificial\s*intelligence\b",
        r"\bmachine\s*learning\b",
    ],
    "coven": [
        r"\bcoven\b",
        r"\bphoenix\b",
        r"\bwitches?\b",
        r"\bmagic\b",
        r"\bspells?\b",
        r"\bcircle\b",
    ],
    "slow-burn": [
        r"\bslow\s*burn\b",
        r"\bmusical\b",
        r"\bsongs?\b",
        r"\blyrics?\b",
        r"\bmelody\b",
    ],
}


class VaultContextSearch:
    """
    Searches Obsidian vault for project-specific context to improve transcription accuracy.

    Flow:
    1. Detect project from raw transcript using keyword patterns
    2. Find project folder in vault
    3. Extract character names, terminology, proper nouns
    4. Return context string for Claude to use in refinement
    """

    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path).expanduser()
        self.context_cache: Dict[str, str] = {}
        self._project_folders: Dict[str, Path] = {}

    def detect_project(self, transcript: str) -> Optional[str]:
        """
        Detect which project the transcript is about based on keywords.

        Args:
            transcript: Raw transcript text from Whisper

        Returns:
            Project name or None if no match
        """
        transcript_lower = transcript.lower()

        scores: Dict[str, int] = {}
        for project, patterns in PROJECT_PATTERNS.items():
            score = sum(
                1 for pattern in patterns
                if re.search(pattern, transcript_lower, re.IGNORECASE)
            )
            if score > 0:
                scores[project] = score

        if scores:
            best_match = max(scores, key=scores.get)
            logger.debug(f"Detected project: {best_match} (score: {scores[best_match]})")
            return best_match

        return None

    def get_project_context(self, project: str) -> str:
        """
        Get relevant context for a project from the vault.

        Args:
            project: Project name (e.g., "days-of-awe")

        Returns:
            Context string with character names, terminology, etc.
        """
        # Check cache
        if project in self.context_cache:
            return self.context_cache[project]

        context_parts = []

        # Find project folder
        project_dir = self._find_project_folder(project)
        if not project_dir:
            logger.warning(f"No project folder found for: {project}")
            return ""

        logger.debug(f"Searching for context in: {project_dir}")

        # Gather character names
        characters = self._extract_character_names(project_dir)
        if characters:
            context_parts.append(f"CHARACTER NAMES: {', '.join(sorted(characters))}")

        # Gather terminology/glossary
        terminology = self._extract_terminology(project_dir)
        if terminology:
            context_parts.append(f"TERMINOLOGY: {', '.join(sorted(terminology))}")

        # Gather location names
        locations = self._extract_locations(project_dir)
        if locations:
            context_parts.append(f"LOCATIONS: {', '.join(sorted(locations))}")

        context = "\n".join(context_parts)
        self.context_cache[project] = context

        logger.info(f"Found context for {project}: {len(characters)} characters, "
                    f"{len(terminology)} terms, {len(locations)} locations")

        return context

    def _find_project_folder(self, project: str) -> Optional[Path]:
        """Find the project folder in the vault."""
        if project in self._project_folders:
            return self._project_folders[project]

        # Try various common locations
        search_patterns = [
            project,
            f"Projects/{project}",
            f"Writing/{project}",
            f"projects/{project}",
            f"writing/{project}",
        ]

        for pattern in search_patterns:
            candidate = self.vault_path / pattern
            if candidate.exists() and candidate.is_dir():
                self._project_folders[project] = candidate
                return candidate

        # Try fuzzy match on project name
        project_lower = project.lower().replace("-", "").replace("_", "")
        for folder in self.vault_path.iterdir():
            if folder.is_dir():
                folder_lower = folder.name.lower().replace("-", "").replace("_", "")
                if folder_lower == project_lower:
                    self._project_folders[project] = folder
                    return folder

        return None

    def _extract_character_names(self, project_dir: Path) -> Set[str]:
        """Extract character names from project files."""
        names: Set[str] = set()

        # Look for character files
        character_patterns = [
            "characters.md",
            "characters/*.md",
            "cast.md",
            "cast/*.md",
            "people.md",
            "dramatis-personae.md",
        ]

        for pattern in character_patterns:
            for path in project_dir.glob(pattern):
                try:
                    content = path.read_text(errors="ignore")
                    # Extract headers that look like character names
                    # e.g., "# Jonah", "## Rabbi David", "### Sarah Cohen"
                    for match in re.findall(
                        r'^#{1,3}\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
                        content,
                        re.MULTILINE
                    ):
                        names.add(match)
                except Exception as e:
                    logger.debug(f"Error reading {path}: {e}")

        # Also scan scene files for character dialogue markers
        for scene_file in project_dir.glob("**/*.md"):
            try:
                content = scene_file.read_text(errors="ignore")
                # Look for dialogue patterns like "JONAH:" or "**Jonah:**"
                for match in re.findall(
                    r'^(?:\*\*)?([A-Z][A-Z]+(?:\s+[A-Z]+)?)(?:\*\*)?:',
                    content,
                    re.MULTILINE
                ):
                    # Convert from UPPERCASE to Title Case
                    names.add(match.title())
            except Exception:
                pass

        return names

    def _extract_terminology(self, project_dir: Path) -> Set[str]:
        """Extract project-specific terminology."""
        terms: Set[str] = set()

        # Look for glossary/terminology files
        glossary_patterns = [
            "glossary.md",
            "terminology.md",
            "terms.md",
            "vocabulary.md",
            "lexicon.md",
        ]

        for pattern in glossary_patterns:
            for path in project_dir.glob(pattern):
                try:
                    content = path.read_text(errors="ignore")
                    # Look for bold terms like **term** or definition patterns
                    for match in re.findall(r'\*\*([^*]+)\*\*', content):
                        terms.add(match.strip())
                    # Look for term: definition patterns
                    for match in re.findall(r'^([A-Z][a-z]+(?:\s+[a-z]+)*)\s*:', content, re.MULTILINE):
                        terms.add(match.strip())
                except Exception as e:
                    logger.debug(f"Error reading {path}: {e}")

        return terms

    def _extract_locations(self, project_dir: Path) -> Set[str]:
        """Extract location names from project files."""
        locations: Set[str] = set()

        # Look for location/setting files
        location_patterns = [
            "locations.md",
            "settings.md",
            "places.md",
            "world.md",
        ]

        for pattern in location_patterns:
            for path in project_dir.glob(pattern):
                try:
                    content = path.read_text(errors="ignore")
                    # Extract headers as location names
                    for match in re.findall(
                        r'^#{1,3}\s+([A-Z][a-zA-Z\s]+)',
                        content,
                        re.MULTILINE
                    ):
                        locations.add(match.strip())
                except Exception:
                    pass

        return locations

    def clear_cache(self):
        """Clear the context cache."""
        self.context_cache.clear()
        self._project_folders.clear()
