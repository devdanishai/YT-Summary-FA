from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
import os
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
load_dotenv()

# Configure Google Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Prompt for the Google Gemini API
prompt = """You are a YouTube video summarizer. You will be taking the transcript text
and summarizing the entire video and providing the important summary in points
within 250 words. Please provide the summary of the text given here: """


class VideoRequest(BaseModel):
    youtube_url: str
    language: str = "en"


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/process_video")
async def process_video(video_request: VideoRequest):
    try:
        # Extract video ID
        video_id = video_request.youtube_url.split("=")[1]

        # Get available transcript languages
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        available_languages = [t.language_code for t in transcript_list]

        if video_request.language not in available_languages:
            raise HTTPException(
                status_code=400,
                detail=f"Transcript not available in {video_request.language}. Available languages: {', '.join(available_languages)}"
            )

        # Get the transcript in the specified language
        transcript_text = transcript_list.find_transcript([video_request.language]).fetch()
        transcript = " ".join([i["text"] for i in transcript_text])

        # Generate summary using Gemini
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt + transcript)

        return {"summary": response.text}

    except TranscriptsDisabled:
        raise HTTPException(status_code=400, detail="Transcripts are disabled for this video.")
    except NoTranscriptFound:
        raise HTTPException(status_code=400, detail="No transcript found for this video.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
