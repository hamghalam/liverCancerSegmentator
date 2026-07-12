from uuid import uuid4

from fastapi import APIRouter

from api.schemas.case import CaseRequest

router = APIRouter(
    prefix="/cases",
    tags=["Cases"],
)

_DATABASE = {}


@router.post("")
async def create_case(case: CaseRequest):

    case_id = str(uuid4())

    _DATABASE[case_id] = case

    return {
        "case_id": case_id,
        "status": "created",
    }


@router.get("/{case_id}")
async def get_case(case_id: str):

    return _DATABASE.get(case_id)