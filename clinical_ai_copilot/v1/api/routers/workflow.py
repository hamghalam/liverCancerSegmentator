from fastapi import APIRouter

from api.services.copilot import run_workflow

router = APIRouter(
    prefix="/workflow",
    tags=["Workflow"],
)


@router.post("/run")
async def workflow(case_id: str):

    return await run_workflow(case_id)