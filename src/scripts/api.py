from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI
import sys
from service import extract_tags

app = FastAPI()

class TitleDto(BaseModel):
    title:str

@app.post('/extract')
async def root(body:TitleDto):
    df = extract_tags(body.title)
    return {'prediction': df.to_dict(orient='records')}

if __name__=='__main__':
    PORT = int(sys.argv[1])
    uvicorn.run('api:app',host='0.0.0.0', port=PORT, reload=True, debug=True, workers=3)