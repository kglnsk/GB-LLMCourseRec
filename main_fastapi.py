
import os

import pandas as pd
import torch
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
import shutil
import textract
import tempfile
from pydantic import BaseModel, Field
import joblib 
from fastapi.middleware.cors import CORSMiddleware
import zipfile
from typing import List
import requests
import openai 
import json 
import whisper



class InputText(BaseModel):
    text: str

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the schema for the output.
class User(BaseModel):
    course_number: str = Field(description = 'НОМЕР наиболее подходящей ПРОГРАММЫ КУРСА')
    course: str = Field(description="Наиболее подходящая ПРОГРАММА КУРСА")
    tech_skills: str = Field(description="Технические навыки из ПРОГРАММЫ КУРСА, которые совпадают с ТРЕБОВАНИЯМИ")

    



client = openai.OpenAI(
    base_url = "localhost:8000/api/v1",
    api_key = ''
)

asr_model = whisper.load_model("medium")  # You can choose different models like 'tiny', 'base', 'small', 'medium', 'large'

f = open('output_courses.json')
data = json.load(f)
courses_list = '\n\n'.join([f"{i}. {s['name']}\n {s['description']} \n ТЕХНИЧЕСКИЕ НАВЫКИ: {','.join(s['tags'])}" for i,s in enumerate(data) if 'developer' in s['url']])

@app.post("/recommend/")
async def recommend(file: UploadFile = File(...)):
    try:
        # Create a temporary file
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # Extract text from the file
        text = textract.process(tmp_path,encoding="unicode")
        # Process extracted text with the model
        input_data = InputText(text=text)
        prediction_response = await predict_with_additional_data(input_data)

        # Clean up: remove the temporary file
        os.unlink(tmp_path)

        return JSONResponse(status_code=200, content=prediction_response)
 
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})
    

@app.post("/recommend_text/")
async def recommend_text(input_data: InputText):
    try:
        # Assuming `predict_with_additional_data` is a function that takes InputText and returns prediction
        prediction_response = await predict_with_additional_data(input_data)

        return JSONResponse(status_code=200, content=prediction_response)

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})


    

@app.post("/recommend_audio/")
async def recommend_audio(file: UploadFile = File(...)):
    try:
        # Validate file type
        suffix = os.path.splitext(file.filename)[1]
        if suffix not in ['.mp3', '.wav', '.m4a', '.flac','.ogg']:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # Load the Whisper model

        # Perform speech recognition
        result = asr_model.transcribe(tmp_path)
        text = result['text']
        print(text)

        # Process extracted text with the model
        input_data = InputText(text=text)
        prediction_response = await predict_with_additional_data(input_data)

        # Clean up: remove the temporary file
        os.unlink(tmp_path)

        return JSONResponse(status_code=200, content=prediction_response)

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})


    
async def predict_with_additional_data(input_data: InputText):


    promptstring = f"""
    ОПИСАНИЕ ЗАПРОСА:
    {input_data}
    КОНЕЦ ОПИСАНИЯ

    ПРОГРАММЫ КУРСОВ
    {courses_list}
    """
    
    # Generate
    chat_completion = client.chat.completions.create(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    response_format={
        "type": "json_object", 
        "schema": User.model_json_schema()
    },
    messages=[
        {"role": "system", "content": "Ты - помощник по профессиональному развитию и переподготовке. На основе информации, содержащейся в ЗАПРОСА, ответь, какая из ПРОГРАММ КУРСОВ наиболее хорошо подходит для этого ЗАПРОСА, а также, какие навыки из ПРОГРАММЫ КУРСА будут наиболее полезны. Отвечай на русском языке в формате JSON."},
        {"role": "user", "content": promptstring}
    ])

    merged_data = json.loads(chat_completion.choices[0].message.content)
    print(json.dumps(merged_data, indent=2))

    return merged_data
