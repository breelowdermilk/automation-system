"""
CLI Interface - Command line tool for managing automations
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional
import click
import yaml

from .database import Database
from .model_router import ModelRouter
from .quality_audit import QualityAuditor


def get_config_path() -> Path:
    """Get the configuration file path."""
    return Path.home() / ".automation" / "config.yaml"


def load_config() -> dict:
    """Load configuration from file."""
    config_path = get_config_path()
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def get_db() -> Database:
    """Get database instance."""
    config = load_config()
    db_path = config.get("paths", {}).get("database", "~/.automation/automation.db")
    return Database(str(Path(db_path).expanduser()))


@click.group()
def cli():
    """Automation Control Panel CLI"""
    pass


@cli.command()
def start():
    """Start all automation watchers."""
    click.echo("Starting automation watchers...")

    # Import here to avoid circular imports
    from .main import AutomationDaemon

    config = load_config()
    daemon = AutomationDaemon(config)

    try:
        asyncio.run(daemon.run())
    except KeyboardInterrupt:
        click.echo("\nStopping...")


@cli.command()
def stop():
    """Stop all automation watchers."""
    import subprocess

    # Find and kill the daemon process
    result = subprocess.run(
        ["pkill", "-f", "automation start"],
        capture_output=True,
    )

    if result.returncode == 0:
        click.echo("Stopped automation daemon")
    else:
        click.echo("No running daemon found")


@cli.command()
def status():
    """Show current automation status."""
    config = load_config()
    db = get_db()

    today = db.get_today_summary()

    click.echo("\n📊 Today's Activity")
    click.echo(f"   Screenshots: {today.get('screenshots') or 0}")
    click.echo(f"   Audio Memos: {today.get('audio') or 0}")
    click.echo(f"   Inbox Files: {today.get('inbox') or 0}")
    success_rate = today.get('success_rate') or 0
    click.echo(f"   Success Rate: {success_rate:.1f}%")

    # Model stats
    model_stats = db.get_stats_by_model(days=7)

    click.echo("\n🤖 Model Usage (7 days)")
    for stat in model_stats[:5]:
        cost = f"${stat['total_cost_usd']:.3f}" if stat['total_cost_usd'] else "$0 (sub)"
        click.echo(f"   {stat['model']}: {stat['total_requests']} calls, {cost}")


@cli.command()
@click.option("--sample-size", "-n", default=20, help="Number of items to audit")
@click.option("--task-type", "-t", default=None, help="Filter by task type")
def audit(sample_size: int, task_type: Optional[str]):
    """Run a quality audit."""
    click.echo(f"Running audit on {sample_size} items...")

    config = load_config()
    db = get_db()
    router = ModelRouter(config)
    auditor = QualityAuditor(router, db, config)

    result = asyncio.run(auditor.run_audit(sample_size=sample_size, task_type=task_type))

    if result.get("success"):
        click.echo(f"\n✅ Audited {result.get('items_audited', 0)} items")

        if result.get("overall_observations"):
            click.echo(f"\n📝 Observations: {result['overall_observations']}")

        audits = result.get("audits", [])
        if audits:
            click.echo("\n📊 Results:")
            for audit in audits[:5]:
                score = (audit.get("accuracy_score", 0) + audit.get("usefulness_score", 0)) / 2
                status = "✓" if not audit.get("would_change") else "⚠"
                click.echo(f"   {status} ID {audit['id']}: {score:.1f}/5")
                if audit.get("suggested_improvement"):
                    click.echo(f"      → {audit['suggested_improvement']}")
    else:
        click.echo(f"\n❌ Audit failed: {result.get('error')}")


@cli.command()
@click.option("--limit", "-n", default=20, help="Number of logs to show")
@click.option("--task-type", "-t", default=None, help="Filter by task type")
@click.option("--hours", "-h", default=24, help="Hours to look back")
def logs(limit: int, task_type: Optional[str], hours: int):
    """View recent processing logs."""
    db = get_db()

    entries = db.get_recent_logs(limit=limit, task_type=task_type, hours=hours)

    click.echo(f"\n📜 Recent Logs (last {hours}h)")
    click.echo("-" * 80)

    for entry in entries:
        timestamp = entry.timestamp[:16] if entry.timestamp else "?"
        status = "✓" if entry.success else "✗"
        result = entry.result[:40] if entry.result else (entry.error_message[:40] if entry.error_message else "-")

        click.echo(f"{status} [{timestamp}] {entry.task_type:20} {entry.model:10} → {result}")


@cli.command()
@click.option("--days", "-d", default=7, help="Days to analyze")
def stats(days: int):
    """Show usage statistics."""
    db = get_db()

    model_stats = db.get_stats_by_model(days=days)
    task_stats = db.get_stats_by_task(days=days)

    click.echo(f"\n📊 Statistics (last {days} days)")
    click.echo("\nBy Model:")
    click.echo("-" * 80)
    click.echo(f"{'Model':<20} {'Requests':>10} {'Success':>10} {'Latency':>10} {'Score':>8} {'Cost':>10}")
    click.echo("-" * 80)

    for stat in model_stats:
        latency = f"{stat['avg_latency_ms']/1000:.2f}s" if stat['avg_latency_ms'] else "-"
        score = f"{stat['avg_audit_score']:.1f}" if stat['avg_audit_score'] else "-"
        cost = f"${stat['total_cost_usd']:.4f}" if stat['total_cost_usd'] else "$0"

        click.echo(f"{stat['model']:<20} {stat['total_requests']:>10} {stat['success_rate']:>9.1f}% {latency:>10} {score:>8} {cost:>10}")

    click.echo("\nBy Task Type:")
    click.echo("-" * 60)

    for stat in task_stats:
        click.echo(f"  {stat['task_type']}: {stat['total_requests']} requests, {stat['success_rate']:.1f}% success")


@cli.command()
def config():
    """Edit configuration file."""
    config_path = get_config_path()

    if not config_path.exists():
        click.echo("Config file not found. Run 'automation setup' first.")
        return

    editor = os.environ.get("EDITOR", "nano")
    os.system(f"{editor} {config_path}")


@cli.command()
@click.option("--port", "-p", default=8420, help="Port to run on")
@click.option("--host", "-h", default="127.0.0.1", help="Host to bind to")
@click.option("--native", is_flag=True, help="Open in native window (requires pywebview)")
def web(port: int, host: str, native: bool):
    """Start the web control panel."""
    config = load_config()
    db = get_db()
    router = ModelRouter(config)
    auditor = QualityAuditor(router, db, config)

    from .web.app import init_app, run_server

    init_app(db, router, auditor, config, get_config_path())

    if native:
        # Native window mode using pywebview
        try:
            import webview
            import threading

            # Start server in background thread
            server_thread = threading.Thread(
                target=run_server,
                kwargs={"host": host, "port": port},
                daemon=True,
            )
            server_thread.start()

            # Give server time to start
            import time
            time.sleep(0.5)

            # Create native window
            click.echo("Opening native window...")
            webview.create_window(
                "Automation Control",
                f"http://{host}:{port}",
                width=1000,
                height=700,
            )
            webview.start()

        except ImportError:
            click.echo("pywebview not installed. Run: pip install pywebview")
            click.echo(f"Falling back to browser mode at http://{host}:{port}")
            run_server(host=host, port=port)
    else:
        click.echo(f"Starting web control panel at http://{host}:{port}")
        run_server(host=host, port=port)


@cli.command()
@click.option("--model", "-m", required=True, help="Model to test")
@click.option("--count", "-n", default=5, help="Number of test images")
def test(model: str, count: int):
    """Test a model with sample screenshots."""
    from pathlib import Path

    # Find recent screenshots
    desktop = Path.home() / "Desktop"
    screenshots = list(desktop.glob("Screenshot*.png"))[:count]

    if not screenshots:
        click.echo("No screenshots found on Desktop")
        return

    click.echo(f"Testing {model} with {len(screenshots)} screenshots...")

    config = load_config()
    router = ModelRouter(config)

    async def run_test():
        for img in screenshots:
            result = await router.process(
                prompt="Describe this screenshot in 2-5 words for a filename.",
                image_path=str(img),
                model_name=model,
            )

            status = "✓" if result.success else "✗"
            latency = f"{result.latency_ms/1000:.2f}s"
            output = result.text[:50] if result.success else result.error

            click.echo(f"{status} {img.name[:30]:30} → {output:30} ({latency})")

    asyncio.run(run_test())


@cli.command()
def setup():
    """Initial setup - create directories and config."""
    automation_dir = Path.home() / ".automation"
    automation_dir.mkdir(exist_ok=True)

    # Create config if doesn't exist
    config_path = automation_dir / "config.yaml"
    if not config_path.exists():
        default_config = {
            "task_defaults": {
                "screenshot_rename": {
                    "model": "sonnet",
                    "fallback": "gemini-flash",
                    "batch_size": 10,
                    "batch_wait_seconds": 60,
                },
                "audio_transcription": {
                    "model": "local-whisper",
                },
                "audio_categorization": {
                    "model": "haiku",
                    "fallback": "llama",
                },
                "inbox_processing": {
                    "model": "sonnet",
                    "fallback": "gemini-flash",
                },
                "quality_audit": {
                    "model": "opus",
                    "schedule": "0 */6 * * *",
                    "sample_size": 20,
                },
            },
            "paths": {
                "watch_screenshots": "~/Desktop",
                "watch_audio": "~/Library/Group Containers/group.com.apple.VoiceMemos.shared/Recordings",
                "watch_inbox": "~/Desktop/inbox",
                "obsidian_vault": "~/Documents/Obsidian",  # UPDATE THIS
                "database": "~/.automation/automation.db",
            },
            "batching": {
                "screenshots": {
                    "enabled": True,
                    "max_batch_size": 10,
                    "max_wait_seconds": 60,
                },
                "audio": {
                    "enabled": False,
                },
                "inbox": {
                    "enabled": True,
                    "max_batch_size": 5,
                    "max_wait_seconds": 120,
                },
            },
            "models": {
                "claude_code": {
                    "timeout_seconds": 60,
                },
                "openrouter": {
                    "api_key_env": "OPENROUTER_API_KEY",
                    "timeout_seconds": 30,
                },
                "local": {
                    "ollama_host": "http://localhost:11434",
                    "whisper_model": "base.en",
                },
            },
        }

        with open(config_path, "w") as f:
            yaml.dump(default_config, f, default_flow_style=False)

        click.echo(f"Created config at {config_path}")
    else:
        click.echo(f"Config already exists at {config_path}")

    # Initialize database
    db = get_db()
    click.echo(f"Initialized database")

    # Create inbox directory
    inbox_dir = Path.home() / "Desktop" / "inbox"
    inbox_dir.mkdir(exist_ok=True)
    click.echo(f"Created inbox directory at {inbox_dir}")

    click.echo("\n✅ Setup complete!")
    click.echo("\nNext steps:")
    click.echo("1. Edit config: automation config")
    click.echo("2. Update 'obsidian_vault' path in config")
    click.echo("3. Set OPENROUTER_API_KEY if using OpenRouter")
    click.echo("4. Start daemon: automation start")
    click.echo("5. Open control panel: automation web")


if __name__ == "__main__":
    cli()
