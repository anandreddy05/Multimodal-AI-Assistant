# ✈️ Multimodal AI Assistant – AirAI

**AirAI** is an intelligent, voice-enabled virtual assistant built for an airline company. It uses OpenAI’s cutting-edge models to create a rich, multimodal user experience by combining chat, voice, and image generation. Whether you're checking flight prices or getting a visual sneak peek of your destination, AirAI is your smart travel companion.

---

## Features

- **Chat-based AI Assistant** powered by `GPT-4o-mini`
- **Ticket Price Retrieval** using function calling
- **DALL·E 3 Image Generation** for city-specific travel posters
- **Text-to-Speech (TTS)** audio responses using `tts-1` voice model
- **Gradio-powered UI** for real-time user interaction
- Built with Python, OpenAI API, Gradio, and Pydub

---

## Tech Stack

| Component         | Description                                      |
|------------------|--------------------------------------------------|
| `GPT-4o-mini`     | Chat-based conversation + function calling       |
| `whisper-1`       | Converts microphone audio to text |
| `DALL·E 3`        | Destination-themed poster image generation       |
| `TTS-1`           | Converts assistant responses to natural speech   |
| `Pydub`           | Audio playback of AI-generated speech            |
| `Gradio`          | Interactive web interface                        |
| `dotenv`          | Secure environment variable management           |

---

## How it works

1. **User** types or speaks a request.  
2. If spoken, Whisper transcribes the audio and drops the text into the chat box.  
3. The assistant processes the text; when price info is needed it invokes the `get_ticket_price` tool.  
4. A **concise one-sentence answer** is returned.  
5. The text is synthesized to speech and played, and—if a city price was requested—a DALL·E poster is shown.

---

## 📸 Example Output

> **User:** What is the price to Delhi?  
> **Assistant:** The ticket price to Delhi is $10.  
> _![Chat UI Preview](Img_Output\Gradio_Output.png))
> _🔊 Spoken response using Alloy voice_

---

## Local Setup

### Prerequisites

- Python 3.8+
- OpenAI API Key with access to:
  - `gpt-4o-mini`
  - `dall-e-3`
  - `tts-1`
  - `whisper-1`
- `ffmpeg` installed (required by `pydub` for audio)

### Installation

```bash
[(https://github.com/anandreddy05/Multimodal-AI-Assistant.git)]
cd airai-assistant
pip install openai python-dotenv gradio
```

## .env file

Create a .env file in the root directory

- OPENAI_API=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

## Run the app

```bash
python app.py
```
