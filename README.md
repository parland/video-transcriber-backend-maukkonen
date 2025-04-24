# 🎬 Video Transcriber — Subtitle Generator with Whisper AI

A simple web app that allows you to upload a video, transcribe the audio into subtitles using OpenAI's Whisper model (supporting **English** and **Swedish**), edit the subtitles in-browser, and download them as a `.srt` file ready for YouTube or other platforms.

Built with:
- 🧠 Whisper AI (via Python + Flask)
- 🎛 Next.js frontend
- 🎧 ffmpeg for audio extraction

---

## 📦 Features

✅ Upload video (MP4, MOV, etc.)  
✅ Choose spoken language (English or Swedish)  
✅ Auto-transcribe using Whisper  
✅ Preview and edit subtitles  
✅ Download subtitles in `.srt` format

---

## 🛠 Local Setup Guide

### 1. Requirements

- Python 3.8+
- Node.js + npm/yarn
- ffmpeg installed & accessible in PATH
- (Optional but recommended) virtualenv

---

### 2. Clone & Set Up Project

```bash
git clone https://github.com/yourusername/video-transcriber.git
cd video-transcriber

# Create folders if not already done
mkdir backend frontend
