# 🎬 Video Transcriber Backend

This is the backend service for the Video Transcription Application. It uses OpenAI's Whisper model via Flask to transcribe audio extracted from uploaded video files.

## Features

-   **Transcription**: Uses Whisper to transcribe audio.
-   **Language Support**: Supports language auto-detection or specifying any language supported by Whisper.
-   **Model Selection**: Automatically selects an appropriate Whisper model size based on available resources (GPU/CPU, Memory) or uses the size specified via environment variables.
-   **GPU Acceleration**: Supports NVIDIA (CUDA) and Apple Silicon (MPS) GPUs for faster transcription.
-   **API**: Provides endpoints for uploading videos, checking transcription status, and downloading subtitles.
-   **Dockerized**: Designed to be run within a Docker container, with dependencies managed via `Dockerfile`.

## Dependencies

-   Python 3.10+
-   Flask, Flask-CORS, psutil, ffmpeg-python, numpy, requests
-   PyTorch >= 2.0.0
-   openai-whisper >= 20231117
-   ffmpeg (installed via `apt-get` in Dockerfile)

## Environment Variables

The backend behavior can be configured via environment variables, typically set by the `run_transcriber.sh` script when running in the combined Docker setup:

-   `WHISPER_MODEL_SIZE`: (Optional) Specifies the Whisper model to use (e.g., `tiny`, `base`, `small`, `medium`, `large`). If not set, the application attempts to select one based on resources.
-   `WHISPER_CACHE_DIR`: (Optional) Path to a directory for caching downloaded Whisper models (e.g., `/root/.cache/whisper`).
-   `GPU_TYPE`: (Informational, set by `run_transcriber.sh`) Detected GPU type (`nvidia`, `apple_silicon`, `none`).
-   `GPU_MEMORY`: (Informational, set by `run_transcriber.sh`) Detected usable GPU memory in GB.
-   `MEMORY_LIMIT`: (Informational, set by `run_transcriber.sh`) Memory limit for the container in bytes.

## API Endpoints

The Flask application runs on port 8765 inside the container.

-   **`POST /api/upload`**
    -   Uploads a video file for transcription.
    -   Requires `multipart/form-data` with:
        -   `file`: The video file.
        -   `language`: (Optional) Language code (e.g., `en`, `sv`) or `auto` for detection. Defaults to `auto`.
        -   `model`: (Optional) Whisper model size to use (e.g., `small`). Defaults to the recommended model based on resources.
    -   Returns `202 Accepted` if transcription starts successfully.
    -   Returns `400 Bad Request` if no file is uploaded.
    -   Returns `500 Internal Server Error` on other errors.

-   **`GET /api/status`**
    -   Checks the current status of the transcription process.
    -   Returns JSON with:
        -   `status`: `idle`, `processing`, `completed`, `error`.
        -   `progress`: Percentage (0-100).
        -   `message`: Current status message.
        -   `subtitles`: Array of subtitle objects (`{start, end, text}`) if completed, else `null`.
        -   `model_info`: Information about the model used and available models.

-   **`POST /api/download`**
    -   Downloads subtitles in SRT format.
    -   Requires JSON body with:
        -   `subtitles`: An array of subtitle objects (`{start, end, text}`), typically the edited ones from the frontend.
    -   Returns the subtitles as a downloadable `.srt` file (`text/plain`).
    -   Returns `400 Bad Request` if no subtitles are provided.
    -   Returns `500 Internal Server Error` on errors.

*(Note: Endpoints are also available without the `/api` prefix, e.g., `/upload`)*

## Running Locally (for Development/Testing)

While the primary deployment method is Docker (see `video-transcriber-service/README.md`), you can run the backend locally for development:

1.  **Prerequisites**:
    -   Python 3.10+ and `pip`.
    -   `ffmpeg` installed and available in your system's PATH.
    -   (Optional but recommended) A Python virtual environment (`venv`).
    -   (Optional) NVIDIA GPU with drivers and CUDA toolkit installed for GPU acceleration.

2.  **Clone the repository**:
    ```bash
    # If you haven't already cloned the parent project
    git clone https://github.com/parland/lecture-transcriber-maukkonen.git
    cd lecture-transcriber-maukkonen/video-transcriber-backend-maukkonen
    ```

3.  **Set up Virtual Environment** (Recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate  # Windows
    ```

4.  **Install Dependencies**:
    ```bash
    pip install -r backend/requirements.txt
    pip install -r backend/requirements-ml.txt
    ```
    *(Note: Installing PyTorch (`torch`) might require specific commands depending on your OS and whether you have a CUDA-enabled GPU. Refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/).)*

5.  **Run the Flask Server**:
    ```bash
    cd backend
    python app.py
    ```
    The server will start on `http://localhost:8765`.

## Docker Build

The `Dockerfile` in this directory defines how to build the backend service image. It installs `ffmpeg`, Python dependencies, and sets up the environment. This Dockerfile is typically used as part of the combined build process defined in `video-transcriber-service/Dockerfile.combined`.
