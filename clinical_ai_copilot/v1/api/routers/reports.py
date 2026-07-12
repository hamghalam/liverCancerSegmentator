from fastapi import APIRouter

router = APIRouter(
    prefix="/reports",
    tags=["Reports"],
)


@router.get("/{case_id}")
async def report(case_id: str):

    return {
        "case_id": case_id,
        "report": None,
    }