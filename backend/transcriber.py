import whisper
import os
import subprocess
import torch
import psutil
import time

# Remove hardcoded Windows ffmpeg path
# We'll use the system ffmpeg from PATH in Docker

# Set up cache directory for Whisper models
cache_dir = os.environ.get('WHISPER_CACHE_DIR', None)
if cache_dir:
    print(f"Using Whisper cache directory: {cache_dir}")
    os.makedirs(cache_dir, exist_ok=True)
else:
    print("No custom Whisper cache directory specified")

def extract_audio(video_path, audio_path):
    """Extract audio from the video file."""
    try:
        # Print detailed information about the video file
        print(f"Attempting to extract audio from: {video_path}")
        print(f"File exists: {os.path.exists(video_path)}")
        print(f"File size: {os.path.getsize(video_path) if os.path.exists(video_path) else 'N/A'} bytes")
        print(f"File permissions: {oct(os.stat(video_path).st_mode)[-3:] if os.path.exists(video_path) else 'N/A'}")
        
        # Use subprocess to call ffmpeg to extract audio from video with detailed output
        cmd = [
            "ffmpeg",
            "-i", video_path,  # Input video file
            "-vn",              # No video
            "-acodec", "pcm_s16le",  # Audio codec (WAV format)
            "-ar", "16000",         # Audio sampling rate (16kHz)
            "-ac", "1",             # Mono audio
            audio_path              # Output audio file
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        # Capture stdout and stderr
        result = subprocess.run(
            cmd,
            check=False,  # Don't raise exception yet
            capture_output=True,
            text=True
        )
        
        # Print detailed output
        print(f"ffmpeg exit code: {result.returncode}")
        print(f"ffmpeg stdout: {result.stdout}")
        print(f"ffmpeg stderr: {result.stderr}")
        
        # Now raise exception if command failed
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode,
                cmd,
                output=result.stdout,
                stderr=result.stderr
            )
            
        print(f"Audio extracted to {audio_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during audio extraction: {e}")
        print(f"Command output: {e.output}")
        print(f"Command stderr: {e.stderr}")
        raise
    except Exception as e:
        print(f"Unexpected error during audio extraction: {str(e)}")
        raise

def get_available_memory_gb():
    """Get available system memory in GB"""
    # Get memory from environment variable if set (for container memory limits)
    if 'MEMORY_LIMIT' in os.environ:
        try:
            # Convert from bytes to GB
            return float(os.environ['MEMORY_LIMIT']) / (1024 * 1024 * 1024)
        except (ValueError, TypeError):
            pass
    
    # Fallback to psutil
    mem = psutil.virtual_memory()
    print(f"Total memory: {mem.total / (1024 * 1024 * 1024):.2f} GB")
    print(f"Available memory: {mem.available / (1024 * 1024 * 1024):.2f} GB")
    
    # Use a higher value for Apple Silicon to allow larger models
    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
        print("Apple Silicon detected, using higher memory estimate")
        return max(12.0, mem.available / (1024 * 1024 * 1024))
    
    return mem.available / (1024 * 1024 * 1024)

def select_model_size():
    """Select appropriate model size based on available memory and GPU"""
    available_memory = get_available_memory_gb()
    
    # Get model size from environment variable if set
    env_model_size = os.environ.get('WHISPER_MODEL_SIZE')
    if env_model_size:
        print(f"Using model size from environment variable: {env_model_size}")
        print(f"GPU type: {os.environ.get('GPU_TYPE', 'none')}")
        print(f"GPU memory: {os.environ.get('GPU_MEMORY', '0')}GB")
        
        # Validate model size
        valid_sizes = ["tiny", "base", "small", "medium", "large"]
        if env_model_size in valid_sizes:
            return env_model_size
        else:
            print(f"Warning: Invalid model size '{env_model_size}'. Using 'small' instead.")
            return "small"
    
    # Print memory information
    print(f"Available memory reported: {available_memory:.2f} GB")
    
    # Check for different GPU types with detailed logging
    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch, 'mps') and torch.backends.mps.is_available()
    
    # Print detailed GPU information
    print(f"CUDA available: {has_cuda}")
    if has_cuda:
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    print(f"MPS available attributes: {hasattr(torch, 'mps')}")
    if hasattr(torch, 'mps'):
        print(f"MPS backend available: {torch.backends.mps.is_available()}")
        print(f"MPS backend built: {torch.backends.mps.is_built()}")
    
    # Try to force MPS usage on Apple Silicon
    try:
        if hasattr(torch, 'mps') and torch.backends.mps.is_available():
            # Create a small tensor on MPS to test if it works
            test_tensor = torch.ones(1).to('mps')
            print(f"Successfully created tensor on MPS: {test_tensor.device}")
            has_mps = True
        else:
            print("MPS not available or not working properly")
    except Exception as e:
        print(f"Error testing MPS: {e}")
        has_mps = False
    
    # Check if we're running in a Docker container
    in_docker = os.path.exists('/.dockerenv')
    print(f"Running in Docker container: {in_docker}")
    
    # Select model based on available resources
    if has_cuda:
        if available_memory >= 24:
            model_size = "large"
        elif available_memory >= 16:
            model_size = "medium"
        elif available_memory >= 8:
            model_size = "small"
        else:
            model_size = "tiny"
        print(f"NVIDIA GPU detected, using {model_size} model with CUDA.")
        return model_size
    
    # Check for Apple Silicon
    elif has_mps:
        if available_memory >= 32:
            model_size = "large"
        elif available_memory >= 16:
            model_size = "medium"
        elif available_memory >= 8:
            model_size = "small"
        else:
            model_size = "tiny"
        print(f"Apple Silicon GPU detected, using {model_size} model with MPS.")
        return model_size
    
    # If no GPU is detected, use CPU with appropriate model size
    else:
        if available_memory >= 16:
            model_size = "medium"
        elif available_memory >= 8:
            model_size = "small"
        else:
            model_size = "tiny"
        print(f"No GPU detected, using {model_size} model with CPU.")
        return model_size

# Global variable for progress updates
progress_callback = None

def set_progress_callback(callback):
    """Set a callback function to receive progress updates."""
    global progress_callback
    progress_callback = callback

def update_progress(message, progress, subtitle_count=None):
    """Update progress through the callback if available."""
    global progress_callback
    
    # Add subtitle count to the message if provided
    if subtitle_count is not None:
        message = f"{message} [Subtitles: {subtitle_count}]"
    
    print(f"Progress: {progress}%, {message}")
    if progress_callback:
        progress_callback(message, progress)

def transcribe_video(video_path, language="en", model_size=None):
    """Transcribe the audio from the video file using Whisper."""
    audio_path = "extracted_audio.wav"  # Temporary audio file name
    
    try:
        # Extract audio from video
        update_progress("Extracting audio from video...", 5)
        extract_audio(video_path, audio_path)
        update_progress("Audio extraction complete", 15)
        
        # Use provided model size or select based on available resources
        update_progress("Selecting model size...", 20)
        if model_size is None:
            model_size = select_model_size()
        print(f"Using {model_size} model for transcription")
        update_progress(f"Selected {model_size} model", 25)
        
        # Load Whisper model with the selected size
        update_progress("Loading model...", 30)
        
        # Check if model already exists in cache
        if cache_dir and os.path.exists(os.path.join(cache_dir, f"{model_size}.pt")):
            update_progress(f"Using cached {model_size} model", 32)
        else:
            update_progress(f"Downloading {model_size} model (this may take a while)...", 32)
        
        # Load the model with cache directory if specified
        model = whisper.load_model(model_size, download_root=cache_dir)
        update_progress("Model loaded successfully", 40)
        
        # Check for GPU availability with detailed logging and error handling
        device = "cpu"
        
        update_progress("Detecting available devices...", 45)
        if torch.cuda.is_available():
            print("Using CUDA device for model")
            device = "cuda"
            model.to(device)
            update_progress("Using CUDA GPU for transcription", 50)
        elif hasattr(torch, 'mps') and torch.backends.mps.is_available():
            print("Attempting to use MPS device for model")
            try:
                # Test MPS availability with a small operation
                test_tensor = torch.ones(1).to('mps')
                print(f"MPS test successful: {test_tensor.device}")
                device = "mps"
                model.to(device)
                print("Successfully moved model to MPS device")
                update_progress("Using Apple Silicon GPU for transcription", 50)
            except Exception as e:
                print(f"Error using MPS device: {e}")
                print("Falling back to CPU")
                device = "cpu"
                update_progress("Falling back to CPU for transcription", 50)
        else:
            print("No GPU detected, using CPU for model")
            update_progress("Using CPU for transcription", 50)
        
        print(f"Final device selection: {device}")
        
        # Handle 'auto' language option
        whisper_language = None  # None means auto-detect in Whisper
        if language != "auto":
            whisper_language = language
        
        print(f"Language setting: {language}, Whisper language parameter: {whisper_language}")
        update_progress("Starting transcription...", 55)
        
        # Transcribe the audio
        print(f"Transcribing audio from {audio_path}...")
        update_progress("Transcribing audio (this may take a while)...", 60)
        
        # Simple progress updates
        last_update = time.time()
        
        # Add intermediate progress updates
        update_progress("Analyzing audio...", 62)
        time.sleep(1)  # Small delay to show progress
        
        # First, run a simple test to ensure the model is working
        try:
            print("Running a simple test on the model...")
            update_progress("Testing model with a counting sample...", 65)
            
            # Create a test audio file with counting
            test_audio_path = "test_audio.wav"
            
            # Generate a text file with the counting script
            with open("counting_script.txt", "w") as f:
                f.write("one two three four five")
            
            # Use text2wave or espeak to generate audio from text
            try:
                # Try using espeak (more commonly available)
                subprocess.run([
                    "espeak",
                    "-f", "counting_script.txt",
                    "-w", test_audio_path,
                    "-s", "150",  # Speed
                    "-a", "200",  # Amplitude
                ], check=True)
            except (subprocess.SubprocessError, FileNotFoundError):
                # If espeak fails, create a simple beep tone
                print("espeak not available, creating a simple tone instead")
                subprocess.run([
                    "ffmpeg",
                    "-f", "lavfi",
                    "-i", "sine=frequency=1000:duration=1",
                    "-acodec", "pcm_s16le",
                    "-ar", "16000",
                    "-ac", "1",
                    test_audio_path
                ], check=True)
            
            # Try to transcribe the test audio
            print("Transcribing test audio...")
            test_result = model.transcribe(audio=test_audio_path, language=whisper_language)
            test_text = test_result.get('text', 'No text')
            print(f"Test transcription result: {test_text}")
            
            # Clean up test files
            os.remove(test_audio_path)
            if os.path.exists("counting_script.txt"):
                os.remove("counting_script.txt")
                
            print("Test completed successfully")
            update_progress(f"Model test result: '{test_text}'", 68)
        except Exception as e:
            print(f"Test failed: {e}")
            update_progress(f"Model test failed: {str(e)}", 68)
            # Continue anyway, as the test might fail due to silence
        
        update_progress("Detecting speech segments...", 70)
        time.sleep(1)  # Small delay to show progress
        update_progress("Starting transcription process...", 72)
        time.sleep(1)  # Small delay to show progress
        
        # Set up progress monitoring for the actual transcription
        start_time = time.time()
        last_progress_update = start_time
        
        def progress_monitor():
            """Monitor progress during transcription"""
            nonlocal last_progress_update
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Update progress every 5 seconds
            if current_time - last_progress_update >= 5:
                last_progress_update = current_time
                # Calculate progress based on elapsed time (rough estimate)
                # Assume transcription takes about 2 minutes for a 10-minute video
                progress = min(85, 75 + (elapsed / 120) * 10)
                update_progress(f"Transcribing... (elapsed: {int(elapsed)}s)", int(progress))
                print(f"Transcription in progress... Elapsed: {int(elapsed)} seconds")
                
                # Add a timeout after 5 minutes (300 seconds)
                if elapsed > 300:
                    print("Transcription taking too long, stopping...")
                    update_progress("Transcription timeout after 5 minutes", 100)
                    # Set a flag to indicate timeout
                    nonlocal stop_monitoring
                    stop_monitoring = True
                    raise Exception("Transcription timeout after 5 minutes")
        
        # Start a monitoring thread
        import threading
        stop_monitoring = False
        
        def monitoring_thread():
            while not stop_monitoring:
                progress_monitor()
                time.sleep(1)
        
        monitor = threading.Thread(target=monitoring_thread)
        monitor.daemon = True
        monitor.start()
        
        update_progress("Processing audio data...", 75)
        
        # Perform the actual transcription with detailed logging
        try:
            print(f"Starting transcription of {audio_path} at {time.strftime('%H:%M:%S')}")
            update_progress(f"Transcribing audio file ({os.path.getsize(audio_path) / (1024*1024):.1f} MB)...", 75)
            
            # Set a timeout for logging
            transcription_start = time.time()
            
            # Use a simpler approach without signals or multiprocessing
            # Just perform the transcription directly
            try:
                # Set a start time to track duration
                transcription_start_time = time.time()
                
                # Perform the transcription
                result = model.transcribe(audio=audio_path, language=whisper_language)
                
                # Calculate duration
                transcription_duration = time.time() - transcription_start_time
                print(f"Transcription completed in {transcription_duration:.1f} seconds")
                
            except Exception as e:
                update_progress(f"Error during transcription: {str(e)}", 100)
                raise
            
            # Stop the monitoring thread
            stop_monitoring = True
            if monitor.is_alive():
                monitor.join(timeout=1)
            
            transcription_time = time.time() - transcription_start
            print(f"Transcription completed in {transcription_time:.1f} seconds")
            
            # Count the number of segments/subtitles
            segments = result.get('segments', [])
            subtitle_count = len(segments)
            
            update_progress(f"Transcription completed in {transcription_time:.1f} seconds", 90, subtitle_count)
        except Exception as e:
            update_progress(f"Error during transcription: {str(e)}", 100)
            raise
        
        if 'text' not in result:
            print("No transcription text found.")
            update_progress("No transcription text found", 95)
            return []
        else:
            print("Transcription completed successfully.")
            print(result['text'])
            update_progress("Processing transcription results...", 95, len(segments))
        
        # Split the transcription text into subtitles
        subtitles = []
        segments = result.get('segments', [])
        for segment in segments:
            subtitle = {
                "start": segment['start'],
                "end": segment['end'],
                "text": segment['text']
            }
            subtitles.append(subtitle)
        
        # Clean up the temporary audio file
        try:
            os.remove(audio_path)
        except Exception as e:
            print(f"Warning: Could not remove temporary audio file: {e}")
        
        update_progress("Transcription complete", 100, len(subtitles))
        return subtitles  # Return the formatted subtitles as a list of dictionaries
    except Exception as e:
        print(f"Error during transcription: {e}")
        update_progress(f"Error: {str(e)}", 100)
        # Try to clean up the temporary audio file
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except:
            pass
        raise
