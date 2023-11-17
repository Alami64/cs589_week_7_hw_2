# Whisper-based Conversational AI

This project integrates various technologies to create a conversational AI that listens to audio input, processes it, and provides an appropriate response based on a loaded PDF document. It uses OpenAI's Whisper for transcription, Speech Recognition for audio processing, and GPT models for generating responses.

## Features

- **Audio Transcription:** Utilizes OpenAI's Whisper model for accurate audio transcription.
- **Speech Recognition:** Processes live audio input to detect and transcribe speech.
- **Dynamic Response Generation:** Leverages GPT models to generate responses based on a loaded PDF document.
- **Customizable Settings:** Offers options for model size, energy levels for microphone sensitivity, pause duration, and wake word detection.
- **Audio Playback:** Converts text responses into speech using Google's Text-to-Speech (gTTS) and plays them back.

## Installation

Ensure you have Python installed and then install the required dependencies:

```bash
pip install openai-whisper click torch speech_recognition gtts pydub python-dotenv langchain docarray pyaudio docarray
```

## Usage

- Set your OpenAI API key in an .env file.
- Load the desired PDF document by modifying the load_db function call in the script.
- Run the script using the command line. You can customize its behavior using various options like --model, --english, and others.

```bash
python your_script_name.py --model base --english --energy 300 --pause 0.8
```
