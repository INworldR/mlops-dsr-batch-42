import io
import torch
import numpy

from pydantic import BaseModel

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware


# This ist a "data model" for the output of the classifier
class Result(BaseModel):
    category: str
    confidence: float
