import whisper
import os
import subprocess

# Set the full path to your ffmpeg binary here (adjust this if needed)
os.environ["FFMPEG_BINARY"] = r"C:\ffmpeg\bin\ffmpeg.exe"  # Update to your actual ffmpeg path

def extract_audio(video_path, audio_path):
    """Extract audio from the video file."""
    try:
        # Use subprocess to call ffmpeg to extract audio from video
        subprocess.run([
            "ffmpeg", 
            "-i", video_path,  # Input video file
            "-vn",              # No video
            "-acodec", "pcm_s16le",  # Audio codec (WAV format)
            "-ar", "16000",         # Audio sampling rate (16kHz)
            "-ac", "1",             # Mono audio
            audio_path              # Output audio file
        ], check=True)
        print(f"Audio extracted to {audio_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during audio extraction: {e}")
        raise

def transcribe_video(video_path, language="en"):
    """Transcribe the audio from the video file using Whisper."""
    audio_path = "extracted_audio.wav"  # Temporary audio file name
    
    # Extract audio from video
    extract_audio(video_path, audio_path)
    
    # Load Whisper model (you can choose 'base', 'small', 'medium', 'large', etc.)
    model = whisper.load_model("base")
    
    try:
        # Transcribe the audio
        print(f"Transcribing audio from {audio_path}...")
        result = model.transcribe(audio_path, language=language)
        
        if 'text' not in result:
            print("No transcription text found.")
        else:
            print("Transcription completed successfully.")
            print(result['text'])
        
        # Split the transcription text into subtitles
        subtitles = []
        segments = result['segments']
        for segment in segments:
            subtitle = {
                "start": segment['start'],
                "end": segment['end'],
                "text": segment['text']
            }
            subtitles.append(subtitle)
        
        # Clean up the temporary audio file
        os.remove(audio_path)
        
        return subtitles  # Return the formatted subtitles as a list of dictionaries
    except Exception as e:
        print(f"Error during transcription: {e}")
        raise
