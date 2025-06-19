import io
import torch
import fastapi
import numpy

from pydantic import BaseModel


# This ist a "data model" for the output of the classifier
class Result(BaseModel):
    category: str
    confidence: float
