import os
import io
import time
import queue
import webrtcvad
import threading
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from dotenv import load_dotenv

from openai import OpenAI
from elevenlabs import play, VoiceSettings
from elevenlabs.client import ElevenLabs
from langgraph.graph import StateGraph, MessagesState, END, START
from langgraph.pregel.remote import RemoteGraph
from langchain_core.messages import HumanMessage, convert_to_messages

# Load environment variables from .env file
load_dotenv()

# Check required environment variables
required_env_vars = ["OPENAI_API_KEY", "ELEVENLABS_API_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

# Local deployment (via LangGraph Studio)
local_deployment_url = "http://localhost:2024"
graph_name = "task_maistro"

# Connect to the deployment
try:
    remote_graph = RemoteGraph(graph_name, url=local_deployment_url)
except Exception as e:
    print(f"Error connecting to deployment: {e}")
    print("Make sure your LangGraph deployment is running at", local_deployment_url)
    exit(1)

class VoiceActivityDetector:
    def __init__(self, sample_rate=16000, frame_duration=30):
        self.vad = webrtcvad.Vad(3)  # Aggressiveness mode 3 (highest)
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_size = int(sample_rate * frame_duration / 1000)
        self.audio_queue = queue.Queue()
        self.is_speaking = False
        self.silence_duration = 0
        self.max_silence = 1.0  # Maximum silence duration in seconds
        self.min_speech_frames = 3  # Minimum number of speech frames to confirm speech
        self.speech_frames_count = 0
        self.recording = True
        self.audio_data = []

    def process_audio(self, indata, frames, time, status):
        """Callback for sounddevice's InputStream"""
        self.audio_queue.put(indata.copy())

    def is_speech(self, frame):
        """Check if a frame contains speech"""
        try:
            return self.vad.is_speech(frame.tobytes(), self.sample_rate)
        except:
            return False

    def reset(self):
        """Reset the detector for next recording"""
        self.audio_data = []
        self.is_speaking = False
        self.silence_duration = 0
        self.speech_frames_count = 0
        self.audio_queue = queue.Queue()

    def record_audio(self):
        """Record and process audio with VAD"""
        self.reset()  # Reset state for new recording
        
        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='int16',
                          callback=self.process_audio, blocksize=self.frame_size):
            print("\nListening... (Speak naturally, silence will end the recording)")
            
            while self.recording:
                try:
                    audio_chunk = self.audio_queue.get(timeout=1)
                    self.audio_data.extend(audio_chunk)

                    # Check for speech in the current frame
                    is_current_speech = self.is_speech(audio_chunk)
                    
                    if is_current_speech:
                        self.speech_frames_count += 1
                        self.silence_duration = 0
                        if self.speech_frames_count >= self.min_speech_frames:
                            self.is_speaking = True
                    else:
                        if self.is_speaking:
                            self.silence_duration += self.frame_duration / 1000
                            if self.silence_duration >= self.max_silence:
                                if len(self.audio_data) > self.sample_rate:  # At least 1 second of audio
                                    break
                        self.speech_frames_count = 0

                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    self.recording = False
                    break

        return np.concatenate(self.audio_data) if self.audio_data else None

def record_audio_until_stop(state: MessagesState):
    """Records audio using voice activity detection"""
    detector = VoiceActivityDetector()
    audio_data = detector.record_audio()
    
    if audio_data is None or len(audio_data) < detector.sample_rate:
        return None  # Return None if no valid audio was recorded
    
    # Convert to WAV format in-memory
    audio_bytes = io.BytesIO()
    write(audio_bytes, detector.sample_rate, audio_data)
    audio_bytes.seek(0)
    audio_bytes.name = "audio.wav"

    # Transcribe via Whisper
    transcription = openai_client.audio.transcriptions.create(
       model="whisper-1", 
       file=audio_bytes,
    )

    print("You said:", transcription.text)
    return {"messages": [HumanMessage(content=transcription.text)]}

def play_audio(state: MessagesState):
    
    """Plays the audio response from the remote graph with ElevenLabs."""

    # Response from the agent 
    response = state['messages'][-1]

    # Prepare text by replacing ** with empty strings
    # These can cause unexpected behavior in ElevenLabs
    cleaned_text = response.content.replace("**", "")
    
    # Call text_to_speech API with turbo model for low latency
    response = elevenlabs_client.text_to_speech.convert(
        voice_id="pNInz6obpgDQGcFmaJgB", # Adam pre-made voice
        output_format="mp3_22050_32",
        text=cleaned_text,
        model_id="eleven_turbo_v2_5", 
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
        ),
    )
    
    # Play the audio back
    play(response)

# Define parent graph
builder = StateGraph(MessagesState)

# Add remote graph directly as a node
builder.add_node("audio_input", record_audio_until_stop)
builder.add_node("todo_app", remote_graph)
builder.add_node("audio_output", play_audio)
builder.add_edge(START, "audio_input")
builder.add_edge("audio_input", "todo_app")
builder.add_edge("todo_app","audio_output")
builder.add_edge("audio_output",END)
graph = builder.compile()

if __name__ == "__main__":
    import uuid
    thread_id = str(uuid.uuid4())
    print("Starting voice interaction. Press Ctrl+C to exit.")
    print("\nTips:")
    print("- Speak naturally, pauses will be detected automatically")
    print("- A longer pause (1 second) will end the current recording")
    print("- The assistant will respond after each pause")
    print("- Press Ctrl+C to exit the program")
    
    # Set user ID for storing memories
    config = {"configurable": {"user_id": "Test-Audio-UX", "thread_id": thread_id}}
    
    try:
        while True:  # Main conversation loop
            # Record and process audio
            state = record_audio_until_stop({"messages": []})
            if state is None:  # No valid audio recorded
                continue
                
            # Process through the graph
            for chunk in graph.stream(state, stream_mode="values", config=config):
                if "messages" in chunk:
                    print("\nAssistant:", chunk["messages"][-1].content)
            
            # Small delay before next recording
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nExiting voice interaction...")
    except Exception as e:
        print(f"\nAn error occurred: {e}")