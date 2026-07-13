from pydantic import BaseModel, Field, field_validator

class NewsRequest(BaseModel):
    text: str = Field(
        ...,
        description="Texto da notícia a ser classificada.",
        examples=[
            "O Brasil vence a Copa do Mundo e conquista o hexa."
        ]
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str):
        if not value.strip():
            raise ValueError("O texto da notícia não pode ser vazio.")
        return value

class NewsResponse(BaseModel):
    category: str = Field(
        ...,
        description="Categoria prevista pelo modelo.",
        examples=["esporte"]
    )