from pydantic import BaseModel, Field

class HTMLSelector(BaseModel):
    title: str = Field(description="Css Selector of the paper title")
    abstract: str = Field(description="Css Selector of the paper abstract")
