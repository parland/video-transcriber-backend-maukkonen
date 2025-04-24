from flask import Flask, request, jsonify
import os
from transcriber import transcribe_video
from flask_cors import CORS  # Import CORS

app = Flask(__name__)

# Ensure the upload folder exists
CORS(app)  # Enable CORS for all routes
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/upload", methods=["POST"])
def upload_file():
    """Handle the file upload and transcription request."""
    try:
        # Get the file and language from the request
        file = request.files.get("file")
        language = request.form.get("language", "auto")
        
        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            # Transcribe the video
            subtitles = transcribe_video(file_path, language)
            return jsonify({"subtitles": subtitles})

        return jsonify({"error": "No file uploaded"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/download", methods=["POST"])
def download_subtitles():
    """Generate and download the edited subtitles as an SRT file."""
    try:
        data = request.get_json()
        subtitles = data.get("subtitles", [])
        
        if not subtitles:
            return jsonify({"error": "No subtitles to download"}), 400

        # Generate SRT file content
        srt_content = ""
        for idx, subtitle in enumerate(subtitles, start=1):
            srt_content += f"{idx}\n"
            srt_content += f"{subtitle['start']} --> {subtitle['end']}\n"
            srt_content += f"{subtitle['text']}\n\n"
        
        # Create a temporary file to download the subtitles
        with open("subtitles.srt", "w") as f:
            f.write(srt_content)

        return jsonify({"message": "Subtitles created successfully"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
