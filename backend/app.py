from flask import Flask, request, jsonify, make_response
import os
import logging
import threading
import time
from transcriber import transcribe_video
from flask_cors import CORS  # Import CORS

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Global variables to track transcription progress
# Get model info from environment variables
model_size = os.environ.get('WHISPER_MODEL_SIZE', 'small')
gpu_type = os.environ.get('GPU_TYPE', 'none')
gpu_memory = os.environ.get('GPU_MEMORY', '0')

# Log the model info
logger = logging.getLogger(__name__)
logger.info(f"Model size: {model_size}")
logger.info(f"GPU type: {gpu_type}")
logger.info(f"GPU memory: {gpu_memory}")

# Determine available models based on GPU memory
available_models = []
try:
    gpu_mem_float = float(gpu_memory)
    
    # Add models based on memory requirements
    if gpu_mem_float >= 10:
        available_models = ["tiny", "base", "small", "medium", "large"]
    elif gpu_mem_float >= 5:
        available_models = ["tiny", "base", "small", "medium"]
    elif gpu_mem_float >= 2.5:
        available_models = ["tiny", "base", "small"]
    elif gpu_mem_float >= 1.5:
        available_models = ["tiny", "base"]
    else:
        available_models = ["tiny"]
        
    logger.info(f"Available models based on GPU memory: {available_models}")
except ValueError:
    # Default to small model if GPU memory is not a valid number
    available_models = ["tiny", "base", "small"]
    logger.warning(f"Could not parse GPU memory '{gpu_memory}', defaulting to limited models")

transcription_status = {
    "status": "idle",  # idle, processing, completed, error
    "progress": 0,     # 0-100
    "message": "",
    "subtitles": None,
    "model_info": {
        "model_size": model_size,
        "gpu_type": gpu_type,
        "gpu_memory": gpu_memory,
        "available_models": available_models,
        "recommended_model": model_size
    }
}

app = Flask(__name__)

# Ensure the upload folder exists
# Configure CORS to allow requests from any origin with all methods and headers
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": "*"}})

# Add CORS headers to all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def format_time_code(seconds):
    """Format a time in seconds to SRT format: HH:MM:SS,mmm"""
    # Handle string inputs
    if isinstance(seconds, str):
        try:
            seconds = float(seconds)
        except ValueError:
            # If it's already in SRT format, return it as is
            if ":" in seconds and "," in seconds:
                return seconds
            return "00:00:00,000"  # Default if conversion fails
    
    # Calculate hours, minutes, seconds, and milliseconds
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_part = seconds % 60
    seconds_int = int(seconds_part)
    milliseconds = int((seconds_part - seconds_int) * 1000)
    
    # Format as HH:MM:SS,mmm
    return f"{hours:02d}:{minutes:02d}:{seconds_int:02d},{milliseconds:03d}"

def progress_callback(message, progress):
    """Callback function to update transcription progress."""
    global transcription_status
    transcription_status["message"] = message
    transcription_status["progress"] = progress
    logger.info(f"Progress update: {progress}% - {message}")

def transcribe_in_background(file_path, language, model_size="small"):
    """Run transcription in a background thread and update status."""
    global transcription_status
    
    # Update model info in status
    if "model_info" in transcription_status:
        transcription_status["model_info"]["model_size"] = model_size
        logger.info(f"Using model size: {model_size}")
    
    try:
        # Update status to processing
        transcription_status["status"] = "processing"
        transcription_status["progress"] = 0
        transcription_status["message"] = "Starting transcription..."
        
        # Set the progress callback
        from transcriber import set_progress_callback
        set_progress_callback(progress_callback)
        
        # Perform actual transcription
        subtitles = transcribe_video(file_path, language, model_size)
        
        # Update status to completed
        transcription_status["status"] = "completed"
        transcription_status["progress"] = 100
        transcription_status["message"] = "Transcription completed"
        transcription_status["subtitles"] = subtitles
        
        logger.info(f"Transcription completed with {len(subtitles)} subtitles")
        
    except Exception as e:
        # Update status to error
        transcription_status["status"] = "error"
        transcription_status["message"] = f"Error: {str(e)}"
        transcription_status["progress"] = 100  # Set progress to 100% to indicate completion
        transcription_status["subtitles"] = []  # Return empty subtitles
        logger.error(f"Error during transcription: {str(e)}")

@app.route("/status", methods=["GET", "OPTIONS"])
@app.route("/api/status", methods=["GET", "OPTIONS"])
def get_status():
    """Get the current transcription status."""
    logger.info("Status endpoint called")
    
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return make_response('', 200)
    
    return jsonify(transcription_status)

@app.route("/upload", methods=["POST", "OPTIONS"])
@app.route("/api/upload", methods=["POST", "OPTIONS"])
def upload_file():
    """Handle the file upload and transcription request."""
    global transcription_status
    logger.info("Upload endpoint called")
    
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return make_response('', 200)
    
    try:
        # Reset status but preserve model_info
        model_info = transcription_status.get("model_info", {})
        transcription_status = {
            "status": "idle",
            "progress": 0,
            "message": "",
            "subtitles": None,
            "model_info": model_info
        }
        
        # Log request details
        logger.info(f"Request method: {request.method}")
        logger.info(f"Request headers: {request.headers}")
        logger.info(f"Request files: {request.files}")
        logger.info(f"Request form: {request.form}")
        
        # Get the file and language from the request
        file = request.files.get("file")
        language = request.form.get("language", "auto")
        selected_model = request.form.get("model", model_size)  # Use the default model if not specified
        
        # Validate the selected model is available
        if selected_model not in transcription_status["model_info"]["available_models"]:
            selected_model = transcription_status["model_info"]["recommended_model"]
            logger.warning(f"Selected model {selected_model} not available, using {transcription_status['model_info']['recommended_model']} instead")
        
        logger.info(f"File received: {file.filename if file else None}")
        logger.info(f"Language: {language}")
        logger.info(f"Selected model: {selected_model}")
        
        if file:
            # Create uploads directory if it doesn't exist
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            
            # Get file details
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            logger.info(f"Saving file to {file_path}")
            
            # Save the file
            file.save(file_path)
            
            # Log file details
            logger.info(f"File saved to {file_path}")
            logger.info(f"File exists: {os.path.exists(file_path)}")
            logger.info(f"File size: {os.path.getsize(file_path) if os.path.exists(file_path) else 'N/A'} bytes")
            logger.info(f"File permissions: {oct(os.stat(file_path).st_mode)[-3:] if os.path.exists(file_path) else 'N/A'}")
            
            # Ensure the file is readable
            try:
                with open(file_path, 'rb') as f:
                    # Read a small chunk to verify file is accessible
                    f.read(1024)
                logger.info("File is readable")
            except Exception as e:
                logger.error(f"Error reading file: {str(e)}")

            # Start transcription in a background thread
            transcription_status["status"] = "processing"
            transcription_status["progress"] = 0
            transcription_status["message"] = "Starting transcription..."
            
            thread = threading.Thread(
                target=transcribe_in_background,
                args=(file_path, language, selected_model)
            )
            thread.daemon = True
            thread.start()
            
            return jsonify({"message": "Transcription started"}), 202

        logger.error("No file uploaded")
        return jsonify({"error": "No file uploaded"}), 400

    except Exception as e:
        logger.error(f"Error during upload: {str(e)}")
        transcription_status["status"] = "error"
        transcription_status["message"] = str(e)
        return jsonify({"error": str(e)}), 500


@app.route("/download", methods=["POST", "OPTIONS"])
@app.route("/api/download", methods=["POST", "OPTIONS"])
def download_subtitles():
    """Generate and download the edited subtitles as an SRT file."""
    logger.info("Download endpoint called")
    
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        return make_response('', 200)
    
    try:
        # Log request details
        logger.info(f"Request method: {request.method}")
        logger.info(f"Request headers: {request.headers}")
        
        data = request.get_json()
        logger.info(f"Request data: {data}")
        
        subtitles = data.get("subtitles", [])
        logger.info(f"Subtitles count: {len(subtitles)}")
        
        if not subtitles:
            logger.error("No subtitles to download")
            return jsonify({"error": "No subtitles to download"}), 400

        # Generate SRT file content
        srt_content = ""
        for idx, subtitle in enumerate(subtitles, start=1):
            # Format time codes as HH:MM:SS,mmm
            start_time = format_time_code(subtitle['start'])
            end_time = format_time_code(subtitle['end'])
            
            srt_content += f"{idx}\n"
            srt_content += f"{start_time} --> {end_time}\n"
            srt_content += f"{subtitle['text']}\n\n"
        
        # Create a response with the SRT content
        response = make_response(srt_content)
        response.headers["Content-Type"] = "text/plain"
        response.headers["Content-Disposition"] = "attachment; filename=edited_subtitles.srt"
        
        logger.info("Subtitles file created successfully")
        return response

    except Exception as e:
        logger.error(f"Error during download: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Listen on all interfaces (0.0.0.0) and use port 8765
    app.run(host='0.0.0.0', port=8765, debug=True)
