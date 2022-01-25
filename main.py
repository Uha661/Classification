from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import Classifier
class inp_request(BaseModel):
    test_string:str
    modelName:str
app = FastAPI()

@app.post("/")
async def main(req:inp_request):
    req_dict = req.dict()
    test_string = req_dict["test_string"]
    modelName = req_dict["modelName"]
    return JSONResponse(Classifier.main(test_string,modelName))
