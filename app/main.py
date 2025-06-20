import io
import os
import torch
import numpy

from pydantic import BaseModel

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv

load_dotenv()

huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
wand_api_key = os.getenv("WANDB_API_KEY")

#print("HUGGINGFACE_API_KEY:", huggingface_api_key if huggingface_api_key else "Not set")
#print("WANDB_API_KEY:", wand_api_key   if wand_api_key else "Not set")

# This ist a "data model" for the output of the classifier
class Result(BaseModel):
    category: str   # The category of the image
    confidence: float # The confidence of the classification

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Line 1\nLine 2"}
