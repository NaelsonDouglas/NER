from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI
from pipeline import model

app = FastAPI()

class TitleDto(BaseModel):
    title:str

@app.post('/extract')
async def root(body:TitleDto):
    doc = pipeline(body.title)
    return {'prediction': True}

if __name__=='__main__':
    uvicorn.run('api:app',host='0.0.0.0', port=configs.API_PORT, reload=True, debug=True, workers=3)