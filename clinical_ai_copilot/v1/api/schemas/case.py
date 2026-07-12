from typing import Optional

from pydantic import BaseModel


class PatientProfile(BaseModel):
    age: int
    sex: str
    primary_cancer: str
    performance_status: Optional[str] = None


class CaseRequest(BaseModel):

    ct_path: str

    radiology_report: str

    patient_profile: PatientProfile

    clinical_question: str