
import sys, logging, uvicorn
from loguru import logger
from fastapi import FastAPI
from src.pdf_extract.config import global_config as glob

logger.add(sys.stdout, format='{time} | {level: <8} | {name: ^15} | {function: ^15} | '
                              '{line: >3} | {message}', level=logging.DEBUG, serialize=False)
logger.add(sys.stderr, format='{time} | {level: <8} | {name: ^15} | {function: ^15} | '
                              '{line: >3} | {message}', level=logging.ERROR, serialize=False)
logger.add("logs/file_{time}.log")

app = FastAPI()

@app.get("/")
def health_check():
    version = "0.0.1"
    status = f"Hi there, your service is up! version = {version}"
    logger.info(status)
    return status

if __name__ == "__main__":
    uvicorn.run(app, host=str(glob.UC_APP_CONNECTION), port=int(glob.UC_PORT))