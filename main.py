
import sys, logging, uvicorn, os
from loguru import logger
from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
from typing import List
from src.pdf_extract.config import global_config as glob
from src.pdf_extract.services import pipelines as trained_pipelines
from importlib import reload


logger.add(sys.stdout, format='{time} | {level: <8} | {name: ^15} | {function: ^15} | '
                              '{line: >3} | {message}', level=logging.DEBUG, serialize=False)
logger.add(sys.stderr, format='{time} | {level: <8} | {name: ^15} | {function: ^15} | '
                              '{line: >3} | {message}', level=logging.ERROR, serialize=False)
logger.add("logs/file_{time}.log")


class my_payload(BaseModel):
    """Define payload schema.
    Args:
        BaseModel (_type_): _description_
    """
    text: List[str]
    fname: List[str] 


app = FastAPI(title="HR API", description="API for HR model", version="0.0.1")

# @app.on_event('startup')
# async def load_model():
#     from src.pdf_extract.services import pipelines as trained_pipelines

@app.get("/")
def health_check():
    status = f"Hi there, your service is up! version = {app.version}"
    logger.info(status)
    return status


@app.post('/predict', tags=["predictions"])
async def get_prediction(payload: my_payload):
    """Prediction endpoint
    Args:
        payload (my_payload): _description_
    Returns:
        _type_: _description_
    """
    df = pd.DataFrame(payload.text, columns=['text'])
    prediction = trained_pipelines.pipe.predict(df['text']).tolist()
    proba = trained_pipelines.pipe.predict_proba(df['text'])[:,1].tolist()
    return {"prediction": prediction, "proba of pos.": proba, 'fname': payload.fname}


if __name__ == "__main__":
    uvicorn.run(app, host=str(glob.UC_APP_CONNECTION), port=int(glob.UC_PORT))

# uvicorn main:app --reload 