"""
Quality Audit - Periodic review of processing results using Opus
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

from .model_router import ModelRouter
from .database import Database, ProcessingLogEntry, QualityAuditEntry

logger = logging.getLogger(__name__)


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

Return ONLY the JSON, no other text."""


class QualityAuditor:
    """
    Periodically audits recent processing results using a high-quality model.
    """

    def __init__(
        self,
        router: ModelRouter,
        db: Database,
        config: Dict[str, Any],
    ):
        self.router = router
        self.db = db
        self.config = config
        self.audit_model = config.get("task_defaults", {}).get(
            "quality_audit", {}
        ).get("model", "opus")
        self.sample_size = config.get("task_defaults", {}).get(
            "quality_audit", {}
        ).get("sample_size", 20)

    async def run_audit(
        self,
        sample_size: Optional[int] = None,
        task_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run a quality audit on recent unaudited items.

        Args:
            sample_size: Number of items to audit (default from config)
            task_type: Filter by task type (optional)

        Returns:
            Audit results with scores and observations
        """
        sample_size = sample_size or self.sample_size

        # Get unaudited items
        items = self.db.get_unaudited_items(limit=sample_size)

        if not items:
            logger.info("No unaudited items found")
            return {
                "success": True,
                "items_audited": 0,
                "message": "No items to audit",
            }

        # Filter by task type if specified
        if task_type:
            items = [i for i in items if i.task_type == task_type]

        if not items:
            return {
                "success": True,
                "items_audited": 0,
                "message": f"No unaudited items for task type: {task_type}",
            }

        logger.info(f"Auditing {len(items)} items with {self.audit_model}")

        # Format items for audit
        formatted = self._format_items_for_audit(items)

        # Call audit model
        response = await self.router.process(
            prompt=AUDIT_PROMPT.format(items=formatted),
            model_name=self.audit_model,
            task_type="quality_audit",
        )

        if not response.success:
            logger.error(f"Audit failed: {response.error}")
            return {
                "success": False,
                "error": response.error,
            }

        # Parse results
        try:
            results = json.loads(response.text)
        except json.JSONDecodeError:
            # Try to extract JSON
            results = self._extract_json(response.text)
            if not results:
                logger.error("Failed to parse audit response")
                return {
                    "success": False,
                    "error": "Failed to parse audit response",
                    "raw_response": response.text,
                }

        # Store results
        await self._store_audit_results(results, items)

        return {
            "success": True,
            "items_audited": len(items),
            "audits": results.get("audits", []),
            "overall_observations": results.get("overall_observations"),
            "model": response.model,
            "latency_ms": response.latency_ms,
            "cost_usd": response.cost_usd,
        }

    def _format_items_for_audit(self, items: List[ProcessingLogEntry]) -> str:
        """Format processing log items for the audit prompt."""
        formatted_items = []

        for item in items:
            formatted = f"""
Item ID: {item.id}
Task Type: {item.task_type}
File: {item.file_path}
Model Used: {item.model}
Result: {item.result}
"""
            formatted_items.append(formatted.strip())

        return "\n\n---\n\n".join(formatted_items)

    def _extract_json(self, text: str) -> Optional[Dict]:
        """Try to extract JSON from model response."""
        import re

        patterns = [
            r'\{[\s\S]*\}',
            r'```json\s*([\s\S]*?)\s*```',
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

    async def _store_audit_results(
        self,
        results: Dict[str, Any],
        items: List[ProcessingLogEntry],
    ):
        """Store audit results in database."""
        audits = results.get("audits", [])
        item_map = {item.id: item for item in items}

        for audit in audits:
            item_id = audit.get("id")
            if item_id not in item_map:
                continue

            entry = QualityAuditEntry(
                processing_log_id=item_id,
                audit_model=self.audit_model,
                accuracy_score=audit.get("accuracy_score"),
                usefulness_score=audit.get("usefulness_score"),
                suggested_improvement=audit.get("suggested_improvement"),
                would_rename=audit.get("would_change", False),
                better_name=audit.get("suggested_improvement") if audit.get("would_change") else None,
            )

            self.db.log_audit(entry)

        logger.info(f"Stored {len(audits)} audit results")

    async def get_audit_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get a summary of recent audit scores."""
        stats = self.db.get_stats_by_model(days=days)

        summary = {
            "period_days": days,
            "models": [],
        }

        for stat in stats:
            summary["models"].append({
                "model": stat["model"],
                "total_requests": stat["total_requests"],
                "avg_audit_score": round(stat["avg_audit_score"] or 0, 2),
                "success_rate": round(stat["success_rate"] or 0, 1),
            })

        return summary


class ScheduledAuditor:
    """Runs audits on a schedule."""

    def __init__(
        self,
        auditor: QualityAuditor,
        config: Dict[str, Any],
    ):
        self.auditor = auditor
        self.config = config
        self.running = False
        self._task: Optional[asyncio.Task] = None

        # Parse schedule (default every 6 hours)
        schedule = config.get("task_defaults", {}).get(
            "quality_audit", {}
        ).get("schedule", "0 */6 * * *")
        self.interval_hours = self._parse_schedule(schedule)

    def _parse_schedule(self, cron: str) -> int:
        """Parse cron-like schedule to get interval in hours."""
        # Simple parsing - just extract the hour interval
        # Full cron parsing would need a library
        parts = cron.split()
        if len(parts) >= 2 and parts[1].startswith("*/"):
            try:
                return int(parts[1][2:])
            except ValueError:
                pass
        return 6  # Default to every 6 hours

    async def start(self):
        """Start the scheduled auditor."""
        self.running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"Started scheduled auditor (every {self.interval_hours} hours)")

    async def stop(self):
        """Stop the scheduled auditor."""
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run_loop(self):
        """Main audit loop."""
        while self.running:
            try:
                # Wait for interval
                await asyncio.sleep(self.interval_hours * 3600)

                if self.running:
                    logger.info("Running scheduled audit")
                    result = await self.auditor.run_audit()
                    logger.info(f"Audit complete: {result.get('items_audited', 0)} items audited")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Audit error: {e}")
                # Continue running despite errors
                await asyncio.sleep(60)
