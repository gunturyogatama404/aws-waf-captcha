import asyncio, logging, httpx
from fastapi import FastAPI
from lib._models import TokenTask, CaptchaTask
from lib._aws import AwsTokenProcessor, InvalidCaptchaError, InvalidProxyError

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
app = FastAPI(title="AWS Token Solver API v2")

INFERENCE_URL = "http://127.0.0.1:6000/predict"

@app.post("/solveCaptcha")
async def solve_captcha(task: CaptchaTask):
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(INFERENCE_URL, json=task.dict())
        return resp.json()
    except Exception as e:
        logging.error(e)
        return {"error": str(e)}

@app.post("/getToken")
async def get_token(task: TokenTask):
    try:
        aws = AwsTokenProcessor(task)
        token = aws.get_token()
        return {"token": token, "status": "success"}
    except InvalidCaptchaError:
        return {"status": "failed to solve captcha"}
    except InvalidProxyError:
        return {"status": "invalid proxy"}
    except Exception as ex:
        logging.error(ex)
        return {"exception": str(ex)}

if __name__ == "__main__":
    import uvicorn
    logging.info("🚀 Starting main API...")
    uvicorn.run(app, host="0.0.0.0", port=5000)
