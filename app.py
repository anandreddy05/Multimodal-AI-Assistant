import os
from dotenv import load_dotenv
from openai import OpenAI
import json
import gradio as gr
import base64
from io import BytesIO
from PIL import Image


load_dotenv()
api_key = os.getenv("OPENAI_API")
if api_key and api_key[:2] == "sk":
    print("API key is set")
else:
    print("Something went wrong")
    
MODEL = "gpt-4o-mini"
openai = OpenAI(api_key=api_key)

system_prompt = \
"""
You are a helpful ai assistant for an Airline called AirAI
Give short, relevant and concise answers, no more than 1 sentence
If you do not know about the answer just say I don't know.
"""

ticket_prices = {"hyderabad":"$15","vijayawada":"$25","delhi":"$10","bengaluru":"$15"}
def get_ticket_price(destination_city):
    city = destination_city.lower()
    return ticket_prices.get(city,"No Flights Available") 

price_function = {
    "name": "get_ticket_price",
    "description": "Get the price of the ticket to the destination city. Call this whenever you need to know the ticket price.",
    "parameters": {
        "type": "object",  # JSON object
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that customer wants to travel to."
            }
        },
        "required": ["destination_city"],
        "additionalProperties": False  
    }
}

tools = [{"type":"function","function":price_function}]

def handle_tool_call(tool_call_msg):
    tool_call = tool_call_msg.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    city = args.get("destination_city")
    price = get_ticket_price(city)
    response = {
        "role":"tool",
        "tool_call_id": tool_call.id,
        "content":json.dumps({"destination_city":city,"price":price})
    }
    return response,city

def artist(city):
    prompt = (
        f"Ultra-vibrant travel-poster illustration of {city} on a perfect holiday. "
        "Foreground: a cheerful visitor taking a selfie. "
        "Mid-ground: the city's signature landmarks and skyline rendered in bold, saturated colors. "
        "Background: clear blue sky with stylized sun rays and playful clouds. "
        "Include small icons of local food, cultural symbols, and nature unique to the region, "
        "arranged around the composition like sticker art. "
        "Style: modern flat graphic poster, sharp lines, subtle paper-texture, cinematic lighting. "
        "4K resolution."
    )
    image_response = openai.images.generate(
        model='dall-e-3', 
        prompt=prompt,
        size = "1024x1024",
        n=1,
        response_format="b64_json"
    )
    image_base64 = image_response.data[0].b64_json
    image_data = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_data))

def text_to_speech(message):
    response = openai.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=message
    )
    # Try returning raw bytes first - if this doesn't work, we'll need temp files
    return response.content

def speech_to_text(audio_file):
    """Convert audio file to text using OpenAI Whisper API"""
    if audio_file is None:
        return ""
    
    try:
        # Open the file directly - Gradio provides the file path
        with open(audio_file, "rb") as audio:
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio,
                response_format="text"
            )
        return transcript.strip()
    except Exception as e:
        print(f"Error in speech-to-text conversion: {e}")
        return ""

def chat(message,history):
    image = None
    messages = [
        {"role":"system","content":system_prompt},
    ]
    for user_msg, assistant_msg in history:
        messages.append({"role":"user","content":user_msg})
        messages.append({"role":"assistant","content":assistant_msg})
    messages.append({"role":"user","content":message})
    
    response = openai.chat.completions.create(
        model = MODEL,
        messages=messages,
        tools=tools
    )
    choice = response.choices[0]
    if choice.finish_reason == "tool_calls":
        tool_call_msg = choice.message
        tool_response_msg,city = handle_tool_call(tool_call_msg)
        messages.append(tool_call_msg)
        messages.append(tool_response_msg)
        image = artist(city)
        response = openai.chat.completions.create(
            model=MODEL,
            messages= messages,
            )
    reply = response.choices[0].message.content
    audio_file = text_to_speech(reply)
    return reply, image, audio_file

with gr.Blocks(title="AirAI – Multimodal Assistant") as ui:
    with gr.Row():
        chatbot = gr.Chatbot(height=500, type="messages")
        image_output = gr.Image(height=500)
    with gr.Row():
        audio_output = gr.Audio(label="Assistant Voice", autoplay=True)

    with gr.Row():
        msg = gr.Textbox(label="Type your message", scale=4)
        audio_input = gr.Audio(label="🎤 Voice Input", sources=["microphone"], type="filepath", scale=1)
    with gr.Row():
        send_btn = gr.Button("Send", variant="primary")
        clear = gr.Button("Clear")
    
    def process_audio_input(audio_file):
        """Process audio input and convert to text"""
        if audio_file is None:
            return ""
        
        transcribed_text = speech_to_text(audio_file)
        return transcribed_text
    
    def user(user_msg, history):
        return "", history + [{"role": "user", "content": user_msg}]
    
    def bot(history):
        user_msg = history[-1]["content"]
        # Convert messages format to tuples for the chat function
        history_tuples = []
        for i in range(0, len(history)-1, 2):
            if i+1 < len(history):
                history_tuples.append((history[i]["content"], history[i+1]["content"]))
        
        bot_msg, image, audio_file = chat(user_msg, history_tuples)
        history.append({"role": "assistant", "content": bot_msg})
        return history, image, audio_file
    
    def handle_audio_input(audio_file, history):
        """Handle audio input by converting to text and processing"""
        if audio_file is None:
            return history, None, None, ""
        
        # Convert audio to text
        transcribed_text = process_audio_input(audio_file)
        if not transcribed_text:
            return history, None, None, ""
        
        # Add user message to history
        updated_history = history + [{"role": "user", "content": transcribed_text}]
        
        # Convert messages format to tuples for the chat function
        history_tuples = []
        for i in range(0, len(updated_history)-1, 2):
            if i+1 < len(updated_history):
                history_tuples.append((updated_history[i]["content"], updated_history[i+1]["content"]))
        
        # Get bot response
        bot_msg, image, audio_file = chat(transcribed_text, history_tuples)
        final_history = updated_history + [{"role": "assistant", "content": bot_msg}]
        
        return final_history, image, audio_file, transcribed_text

    # Text input handling
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, [chatbot, image_output, audio_output]
    )
    
    send_btn.click(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, [chatbot, image_output, audio_output]
    )
    
    # Audio input handling
    audio_input.change(
        handle_audio_input, 
        [audio_input, chatbot], 
        [chatbot, image_output, audio_output, msg]
    )

    clear.click(lambda:[],None,chatbot,queue=False)
ui.launch(inbrowser=True)