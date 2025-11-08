import uvicorn
import os

if __name__ == "__main__":
    # Render (and other hosts) automatically assign a port via the PORT environment variable
    port = int(os.environ.get("PORT", 8000))

    # This runs your FastAPI app defined inside main.py as `app = FastAPI()`
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
