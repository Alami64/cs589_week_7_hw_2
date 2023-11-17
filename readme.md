# Voice-Activated Conversational AI

This application is a voice-activated conversational AI that leverages OpenAI's Whisper model for speech recognition and GPT models for generating intelligent responses. It is designed to listen to spoken questions or commands and provide relevant responses based on a loaded PDF document.

## Features

- **Whisper Model Integration:** Utilizes OpenAI's Whisper for state-of-the-art speech recognition.
- **Real-time Voice Processing:** Employs Speech Recognition to capture and process voice inputs.
- **Dynamic Response Generation:** Uses GPT models through the LangChain library to generate contextually relevant responses.
- **Customizable Listening Parameters:** Offers settings for model type, energy threshold, pause duration, and dynamic energy adjustment.
- **Wake Word Detection:** Listens for a specific wake word before processing speech.
- **Voice Feedback:** Converts response text to speech and plays it back using gTTS and PyDub.

## Installation

To run this project, ensure you have Python installed. Then, install the necessary libraries:

```bash
pip install whisper speech_recognition gtts pydub python-dotenv langchain numpy re torch click
```

## Configuration

- Set your OpenAI API key in the `.env` file.
- Modify the `load_db` function in the script to load your desired PDF document for context-based responses.

## Usage

Execute the script via the command line with customizable options:

```bash
python script_name.py --model base --english --energy 300 --pause 0.8 --wake_word "jarvis"
```

## Main Functions

1. **Audio Capture and Processing:**

   ```python
   def record_audio(audio_queue, energy, pause, dynamic_energy):
       # Logic to capture and process audio
   ```

2. **Speech Transcription:**

   ```python
   def transcribe_forever(audio_queue, result_queue, audio_model, english, wake_word, verbose):
       # Continuous transcription logic
   ```

3. **Response Generation and Playback:**

   ```python
   def reply(result_queue, conversational_chain):
       # Generating and playing back responses
   ```
