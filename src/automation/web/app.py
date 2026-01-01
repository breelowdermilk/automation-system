"""
Web Control Panel - FastAPI + HTMX based dashboard
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import json
import yaml

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# These will be injected at startup
db = None
router = None
auditor = None
config = None
config_path = None


app = FastAPI(title="Automation Control Panel")

# Templates directory (will be created by setup)
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


def init_app(database, model_router, quality_auditor, app_config, cfg_path):
    """Initialize the app with dependencies."""
    global db, router, auditor, config, config_path
    db = database
    router = model_router
    auditor = quality_auditor
    config = app_config
    config_path = cfg_path


# ============== Dashboard ==============

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard view."""
    # Get today's stats
    today = db.get_today_summary()

    # Get model stats
    model_stats = db.get_stats_by_model(days=7)

    # Get recent logs
    recent_logs = db.get_recent_logs(limit=10)

    # Get recent audits
    recent_audits = db.get_recent_audits(limit=5)

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "today": today,
        "model_stats": model_stats,
        "recent_logs": recent_logs,
        "recent_audits": recent_audits,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })


@app.get("/dashboard-stats", response_class=HTMLResponse)
async def dashboard_stats_partial(request: Request):
    """HTMX partial for refreshing dashboard stats."""
    today = db.get_today_summary()
    model_stats = db.get_stats_by_model(days=7)

    return templates.TemplateResponse("partials/stats.html", {
        "request": request,
        "today": today,
        "model_stats": model_stats,
    })


# ============== Model Configuration ==============

@app.get("/models", response_class=HTMLResponse)
async def model_config_page(request: Request):
    """Model configuration page."""
    return templates.TemplateResponse("models.html", {
        "request": request,
        "config": config,
        "available_models": list(router.config.get("models", {}).keys()) if router else [],
    })


@app.post("/models/update")
async def update_model_config(
    task_type: str = Form(...),
    model: str = Form(...),
    fallback: Optional[str] = Form(None),
):
    """Update model configuration for a task."""
    global config

    if "task_defaults" not in config:
        config["task_defaults"] = {}

    if task_type not in config["task_defaults"]:
        config["task_defaults"][task_type] = {}

    config["task_defaults"][task_type]["model"] = model
    if fallback:
        config["task_defaults"][task_type]["fallback"] = fallback

    # Save config
    _save_config()

    return RedirectResponse(url="/models", status_code=303)


@app.post("/models/ab-test")
async def toggle_ab_test(
    task_type: str = Form(...),
    enabled: bool = Form(False),
):
    """Toggle A/B testing for a task."""
    global config

    if "ab_tests" not in config:
        config["ab_tests"] = {}

    if task_type not in config["ab_tests"]:
        config["ab_tests"][task_type] = {"enabled": False, "variants": []}

    config["ab_tests"][task_type]["enabled"] = enabled

    _save_config()

    return RedirectResponse(url="/models", status_code=303)


# ============== Logs ==============

@app.get("/logs", response_class=HTMLResponse)
async def logs_page(
    request: Request,
    task_type: Optional[str] = None,
    model: Optional[str] = None,
    hours: int = 24,
    page: int = 1,
):
    """Processing logs page."""
    per_page = 50

    logs = db.get_recent_logs(
        limit=per_page * page,
        task_type=task_type,
        model=model,
        hours=hours,
    )

    # Paginate
    start = (page - 1) * per_page
    logs = logs[start:start + per_page]

    return templates.TemplateResponse("logs.html", {
        "request": request,
        "logs": logs,
        "filters": {
            "task_type": task_type,
            "model": model,
            "hours": hours,
        },
        "page": page,
    })


@app.get("/logs/{log_id}", response_class=HTMLResponse)
async def log_detail(request: Request, log_id: int):
    """Log detail view."""
    logs = db.get_recent_logs(limit=1000)
    log = next((l for l in logs if l.id == log_id), None)

    if not log:
        raise HTTPException(status_code=404, detail="Log not found")

    return templates.TemplateResponse("log_detail.html", {
        "request": request,
        "log": log,
    })


# ============== Audits ==============

@app.get("/audits", response_class=HTMLResponse)
async def audits_page(request: Request):
    """Quality audits page."""
    audits = db.get_recent_audits(limit=50)
    summary = await auditor.get_audit_summary(days=7) if auditor else {}

    return templates.TemplateResponse("audits.html", {
        "request": request,
        "audits": audits,
        "summary": summary,
    })


@app.post("/audits/run")
async def run_audit_now():
    """Trigger a manual audit."""
    if not auditor:
        raise HTTPException(status_code=500, detail="Auditor not initialized")

    result = await auditor.run_audit()

    return RedirectResponse(url="/audits", status_code=303)


# ============== Stats ==============

@app.get("/stats", response_class=HTMLResponse)
async def stats_page(request: Request, days: int = 7):
    """Statistics page."""
    model_stats = db.get_stats_by_model(days=days)
    task_stats = db.get_stats_by_task(days=days)
    daily_counts = db.get_daily_counts(days=days)

    return templates.TemplateResponse("stats.html", {
        "request": request,
        "model_stats": model_stats,
        "task_stats": task_stats,
        "daily_counts": daily_counts,
        "days": days,
    })


# ============== API Endpoints ==============

@app.get("/api/stats")
async def api_stats(days: int = 7):
    """API endpoint for stats."""
    return {
        "today": db.get_today_summary(),
        "model_stats": db.get_stats_by_model(days=days),
        "task_stats": db.get_stats_by_task(days=days),
    }


@app.get("/api/logs")
async def api_logs(
    task_type: Optional[str] = None,
    model: Optional[str] = None,
    hours: int = 24,
    limit: int = 50,
):
    """API endpoint for logs."""
    logs = db.get_recent_logs(
        limit=limit,
        task_type=task_type,
        model=model,
        hours=hours,
    )
    return {"logs": [vars(l) for l in logs]}


@app.get("/api/config")
async def api_get_config():
    """Get current configuration."""
    return config


@app.post("/api/config")
async def api_update_config(new_config: dict):
    """Update configuration."""
    global config
    config.update(new_config)
    _save_config()
    return {"status": "ok"}


# ============== Helpers ==============

def _save_config():
    """Save current config to file."""
    if config_path:
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        # Also log to database
        db.save_config(yaml.dump(config), changed_by="web_ui")


def run_server(host: str = "127.0.0.1", port: int = 8420):
    """Run the web server."""
    uvicorn.run(app, host=host, port=port)


# Entry point for standalone running
if __name__ == "__main__":
    run_server()
