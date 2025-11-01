from fastapi import FastAPI
from lib._ai import get_solutions
from lib._models import CaptchaTask

app = FastAPI(title="Inference Server")

@app.post("/predict")
async def predict(task: CaptchaTask):
    solutions = get_solutions(task.images, task.target)
    return {"solutions": solutions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6000)
