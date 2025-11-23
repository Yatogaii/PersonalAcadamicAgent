from pydantic import BaseModel, Field, ValidationError
from typing import Any
import json


class HTMLSelector(BaseModel):
    title: str = Field(description="Css Selector of the paper title")
    abstract: str = Field(description="Css Selector of the paper abstract")
    link: str = Field(description="Css Selector of the paper detail page link", default="")
    pdf_link: str = Field(description="Css Selector of the paper pdf link", default="")


def to_html_selector(obj: Any) -> HTMLSelector:
    """Convert a JSON string, dict, or an existing HTMLSelector into an HTMLSelector instance.

    Accepts:
    - HTMLSelector -> returned as-is
    - dict -> validated and converted
    - str -> if JSON string, parsed then converted; otherwise tries pydantic raw parsing

    Raises:
        ValueError on invalid input or validation error.
    """
    if isinstance(obj, HTMLSelector):
        return obj

    # dict-like input
    if isinstance(obj, dict):
        try:
            # pydantic v2 API
            return HTMLSelector.model_validate(obj)
        except AttributeError:
            # pydantic v1
            try:
                return HTMLSelector.parse_obj(obj)
            except ValidationError as e:
                raise ValueError(f"Invalid selector dict: {e}") from e
        except ValidationError as e:
            raise ValueError(f"Invalid selector dict: {e}") from e

    # string input: try JSON parse first
    if isinstance(obj, str):
        s = obj.strip()
        if not s:
            raise ValueError("Empty selector string")
        # try parsing as JSON
        if s.startswith('{') or s.startswith('['):
            try:
                data = json.loads(s)
            except Exception as e:
                raise ValueError(f"Invalid JSON string for selector: {e}") from e
            return to_html_selector(data)

        # fallback: try pydantic parse methods that accept raw json
        try:
            return HTMLSelector.model_validate_json(s)
        except AttributeError:
            try:
                return HTMLSelector.parse_raw(s)
            except Exception as e:
                raise ValueError(f"Cannot parse selector string: {e}") from e
        except ValidationError as e:
            raise ValueError(f"Selector JSON failed validation: {e}") from e

    raise TypeError(f"Unsupported selector input type: {type(obj)}")
