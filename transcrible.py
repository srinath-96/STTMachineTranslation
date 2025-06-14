import os
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from RealtimeSTT import AudioToTextRecorder
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
# --- Configuration ---
# You can change the model size here. Options: tiny, base, small, medium, large-v2
# Smaller models are faster but less accurate.
MODEL_SIZE = "medium.en"
LANGUAGE = "en" # Language for transcription

import os
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download, HfFolder

# Set up Hugging Face authentication
HF_TOKEN = os.environ.get("HF_TOKEN")
HfFolder.save_token(HF_TOKEN)
os.environ["HF_HUB_DISABLE_OFFLINE"] = "0"
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.expanduser("~/.cache/huggingface")

# Initialize the model with proper download settings
try:
    print("Downloading model...")
    # First download the model using huggingface_hub
    model_path = snapshot_download(
        "Systran/faster-whisper-tiny.en",
        token=HF_TOKEN,
        local_files_only=False
    )
    
    # Then initialize WhisperModel with the downloaded path
    model = WhisperModel(
        model_path,
        device="cpu",
        compute_type="int8"
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# --- Main Application Logic ---
if __name__ == '__main__':
    console = Console()
    console.print("[green]Initializing Realtime Speech-to-Text...")

    full_sentences = []
    displayed_text = ""

    # This function is called every time a new chunk of text is transcribed
    def text_detected(text):
        """Displays the real-time transcription in a formatted panel."""
        global displayed_text
        
        # Combine old sentences with the new real-time text
        live_text = Text()
        for i, sentence in enumerate(full_sentences):
            style = "yellow" if i % 2 == 0 else "cyan"
            live_text.append(sentence + " ", style=style)
        
        live_text.append(text, style="bold bright_white")

        # Update the live display only if the text has changed
        if live_text.plain != displayed_text:
            displayed_text = live_text.plain
            panel = Panel(live_text, title="[bold green]Live Transcription[/bold green]", border_style="green")
            live.update(panel)

    # This function is called when a full sentence is detected
    def process_text(text):
        """Adds the complete sentence to our list."""
        full_sentences.append(text)
        text_detected("") # Update display to clear the real-time part

    # --- Initialize and Run Recorder ---
    try:
        # Initialize the recorder with your chosen settings
        recorder = AudioToTextRecorder(
            model=MODEL_SIZE,
            language=LANGUAGE,
            spinner=False # Disable the loading spinner
        )

        # The Live object from 'rich' provides a smooth, updating display
        with Live(console=console, refresh_per_second=10, screen=False) as live:
            initial_panel = Panel(Text("Starting up...", style="cyan"), title="[bold yellow]Status[/bold yellow]")
            live.update(initial_panel)

            console.print("[green]Ready! Start speaking...")
            
            # The main loop that listens for your voice and transcribes it
            while True:
                recorder.text(process_text)

    except KeyboardInterrupt:
        console.print("\n[bold red]Transcription stopped by user.[/bold red]")
    except Exception as e:
        console.print(f"\n[bold red]An error occurred: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())