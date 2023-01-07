
import sys, logging, uvicorn, os
from loguru import logger
from fastapi import FastAPI
from pydantic import BaseModel
#from pydantic import BaseModel, conlist
#from typing import List
from src.pdf_extract.config import global_config as glob
from src.pdf_extract.services import pipelines as trained_pipelines
from importlib import reload

reload(trained_pipelines)


logger.add(sys.stdout, format='{time} | {level: <8} | {name: ^15} | {function: ^15} | '
                              '{line: >3} | {message}', level=logging.DEBUG, serialize=False)
logger.add(sys.stderr, format='{time} | {level: <8} | {name: ^15} | {function: ^15} | '
                              '{line: >3} | {message}', level=logging.ERROR, serialize=False)
logger.add("logs/file_{time}.log")


class my_payload(BaseModel):
    text: str
    #fname: str 


app = FastAPI(title="HR API", description="API for HR model", version="0.0.1")

@app.get("/")
def health_check():
    status = f"Hi there, your service is up! version = {app.version}"
    logger.info(status)
    return status


@app.on_event('startup')
async def load_model():
    from src.pdf_extract.services import pipelines as trained_pipelines
    pipe = trained_pipelines.pipe


@app.post('/predict', tags=["predictions"])
async def get_prediction(payload: my_payload):
    data = dict(payload)['text']
    prediction = trained_pipelines.pipe.predict(data).tolist()
    proba = trained_pipelines.pipe.predict_proba(data).tolist()
    return {"prediction": prediction,
            "log_proba": proba}


if __name__ == "__main__":
    uvicorn.run(app, host=str(glob.UC_APP_CONNECTION), port=int(glob.UC_PORT))

# uvicorn main:app --reload 