from pydantic import BaseModel,ConfigDict

class BaseDataModel(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="ignore",
    )
    