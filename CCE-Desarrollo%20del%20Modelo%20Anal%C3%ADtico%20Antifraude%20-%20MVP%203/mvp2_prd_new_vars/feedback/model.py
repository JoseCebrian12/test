from pydantic import BaseModel, Field, constr, conint, field_validator
from typing import Optional, Union, Literal


class AntifraudeRequest(BaseModel):
    debtor_participant_code: str = Field(
        ..., description="Debe tener exactamente 5 dígitos"
    )
    creation_date: str = Field(
        ..., description="Debe tener exactamente 8 dígitos (YYYYMMDD)"
    )
    creation_time: str = Field(
        ..., description="Debe tener exactamente 6 dígitos (HHMMSS)"
    )
    trace: Optional[Union[str, None]] = Field(default=None, max_length=255)
    channel: Literal["Banca Movil", "Homebanking", "ATM", "Sucursal"]
    amount: Union[str, float, int] = Field(..., ge=0)
    currency: Literal["PEN", "USD"]
    response_code: str = Field(..., description="Debe tener exactamente 3 dígitos")
    creditor_cci: str = Field(..., description="Debe tener exactamente 20 dígitos")

    @field_validator("debtor_participant_code")
    def validate_debtor_code(cls, v):
        if not v.isdigit() or len(v) != 5:
            raise ValueError("Debe tener exactamente 5 dígitos")
        return v

    @field_validator("creation_date")
    def validate_creation_date(cls, v):
        if not v.isdigit() or len(v) != 8:
            raise ValueError("Debe tener exactamente 8 dígitos")
        return v

    @field_validator("creation_time")
    def validate_creation_time(cls, v):
        if not v.isdigit() or len(v) != 6:
            raise ValueError("Debe tener exactamente 6 dígitos")
        return v

    @field_validator("response_code")
    def validate_response_code(cls, v):
        if not v.isdigit() or len(v) != 3:
            raise ValueError("Debe tener exactamente 3 dígitos")
        return v

    @field_validator("creditor_cci")
    def validate_creditor_cci(cls, v):
        if not v.isdigit() or len(v) != 20:
            raise ValueError("Debe tener exactamente 20 dígitos")
        return v