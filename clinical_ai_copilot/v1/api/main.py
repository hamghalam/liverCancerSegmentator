from fastapi import FastAPI

from api.routers import (
    health,
    cases,
    workflow,
    reports,
)

app = FastAPI(
    title="Clinical AI Copilot API",
    description="REST API for Clinical AI Copilot",
    version="1.0.0",
)

app.include_router(health.router)
app.include_router(cases.router)
app.include_router(workflow.router)
app.include_router(reports.router)


@app.get("/")
async def root():
    return {
        "service": "Clinical AI Copilot",
        "version": "1.0.0",
        "status": "running",
    }