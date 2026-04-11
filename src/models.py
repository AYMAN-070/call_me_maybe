from pydantic import BaseModel
from typing import Dict, Optional


class TypeDef(BaseModel):
    """Present the definition type of argument or return"""
    type: str


class FunctionDef(BaseModel):
    """function from function_definitions.json"""
    name: str
    description: str
    parameters: Dict[str, TypeDef]
    returns: Optional[TypeDef] = None


class PromptTest(BaseModel):
    """allows to transform each line of JSON into a clean Python object."""
    prompt: str
