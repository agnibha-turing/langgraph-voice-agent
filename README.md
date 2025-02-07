# Task mAIstro

Managing tasks effectively is a universal challenge. Task mAIstro is an AI-powered task management agent that combines natural language processing with long-term memory to create a more intuitive and adaptive experience. This repo can be used to deploy Task mAIstro and interact with it through text or voice!

Key features:
* Natural conversation through text and / or voice to update or add tasks
* Adaptive learning of your management style and preferences
* Persistent memory of tasks, context, and preferences
* Flexible deployment options - local or hosted

![audio_task_maistro](https://github.com/user-attachments/assets/170e1088-499a-4373-b724-da51e9778296)

## Quickstart

1. Populate the `.env` file: 
```
$ cp .env.example .env
```

2. Download the LangGraph Studio desktop app for Mac [here](https://github.com/langchain-ai/langgraph-studio?tab=readme-ov-file#download).

3. Load this repository as a project in LangGraph Studio.

4. Start chatting with the task mAIstro through the text interface in LangGraph Studio.

5. Experiment with voice UX in `ntbk/audio_ux.ipynb`.

## Task mAIstro Application

### Architecture

Task mAIstro leverages [LangGraph](https://langchain-ai.github.io/langgraph/) to maintain three memory types:

1. **ToDo List Memory**
   - Task descriptions and deadlines
   - Time estimates and status tracking
   - Actionable next steps

2. **User Profile Memory**
   - Personal preferences and context
   - Work/life patterns
   - Historical interactions

3. **Interaction Memory**
   - Task management style
   - Communication preferences
   - Organizational patterns

The schema for each memory type as well as the graph flow is defined in `task_maistro.py`. 
The graph flow is orchestrated by a central `task_maistro` node that:
- Chooses to update one of the three memories based on the user's input
- Uses tool calling with the [Trustcall library](https://github.com/hinthornw/trustcall) to update the chosen memory type

### 📚 Learning Resources

> **Want to learn more?** Check out [Module 5 of our LangGraph Course](https://academy.langchain.com/courses/intro-to-langgraph): 
> - All notebooks are available [here](https://github.com/langchain-ai/langchain-academy/tree/main/module-5)
> - All notebooks have accompanying videos

### Deployment Options

Task mAIstro offers three flexible deployment paths:

#### 1. LangGraph Studio (Recommended for quickstart)
The fastest way to get started is with [LangGraph Studio](https://github.com/langchain-ai/langgraph-studio):
- Download the [desktop app](https://github.com/langchain-ai/langgraph-studio?tab=readme-ov-file#download)
- Ensure [Docker Desktop](https://docs.docker.com/engine/install/) is running
- Load this repository as a project

![Screenshot 2024-11-20 at 7 42 35 PM](https://github.com/user-attachments/assets/05c0b98f-2b8c-4098-a7fc-67044fd0db21)

#### 2. LangGraph CLI
For developers who prefer the command line:
- Use the [LangGraph CLI](https://langchain-ai.github.io/langgraph/concepts/langgraph_cli/#up)
- Deploy locally with a single command
- Full control over deployment configuration

#### 3. LangGraph Cloud
For production deployments:
- Deploy to [LangGraph Cloud](https://langchain-ai.github.io/langgraph/concepts/langgraph_cloud/)
- Manage through your LangSmith account
- Access via LangGraph Studio web UI

### 🚀 Learning Resources
> **Want to learn more about deployments?** Check out [Module 6 of our LangGraph Course](https://academy.langchain.com/courses/intro-to-langgraph) 
> - All notebooks are available [here](https://github.com/langchain-ai/langchain-academy/tree/main/module-6)
> - All notebooks have accompanying videos

## Voice Interface

Task mAIstro supports voice interactions using:
- [OpenAI's Whisper](https://platform.openai.com/docs/guides/speech-to-text) for speech-to-text
- [ElevenLabs](https://github.com/elevenlabs/elevenlabs-python) for text-to-speech

### Setup

1. Install FFmpeg (required for ElevenLabs):
```bash
# macOS
brew install ffmpeg

# Windows (using chocolatey)
choco install ffmpeg

# Linux
sudo apt-get install ffmpeg
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables in `.env`:
```bash
cp .env.example .env
```
Then edit `.env` to include your API keys:
- `OPENAI_API_KEY` - Get from [OpenAI](https://platform.openai.com/api-keys)
- `ELEVENLABS_API_KEY` - Get from [ElevenLabs](https://elevenlabs.io/api)
- `LANGCHAIN_API_KEY` - Get from [LangSmith](https://smith.langchain.com/)

### Running the Voice Interface

You can interact with Task mAIstro through voice in two ways:

#### 1. Using the Jupyter Notebook (Interactive)
Open and run `ntbk/audio_ux.ipynb` in Jupyter:
```bash
jupyter notebook ntbk/audio_ux.ipynb
```

#### 2. Using the Standalone Script
Run the standalone Python script:
```bash
python app.py
```

The script will:
1. Record your voice when you speak (press Enter to stop recording)
2. Convert your speech to text using Whisper
3. Process your request using Task mAIstro
4. Respond with voice using ElevenLabs

To exit the program, press Ctrl+C.

Note: Make sure your LangGraph deployment is running before starting the voice interface.
