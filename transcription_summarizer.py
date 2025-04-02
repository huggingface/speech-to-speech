import os
import time
import random
import argparse
import datetime
import torch
from pathlib import Path
import json
import re
import numpy as np
import sounddevice as sd
import librosa
from mlx_lm import load, generate
from melo.api import TTS
import requests  # Add for API requests

class TranscriptionSummarizer:
    def __init__(self, args):
        # Initialize parameters
        self.transcripts_dir = Path(args.transcripts_dir)
        self.summaries_dir = Path(args.summaries_dir)
        self.min_interval = args.min_interval * 60  # Convert to seconds
        self.max_interval = args.max_interval * 60  # Convert to seconds
        self.speak_summary = args.speak_summary
        self.tts_device = args.tts_device
        self.use_api = args.use_api
        self.api_key = args.api_key
        self.api_url = args.api_url
        self.time_window_minutes = args.time_window  # New parameter for time window
        
        # Create summaries directory if it doesn't exist
        os.makedirs(self.summaries_dir, exist_ok=True)
        
        # Initialize the MLX LM model only if not using API
        if not self.use_api:
            print(f"Loading language model: {args.model_name}")
            self.model, self.tokenizer = load(args.model_name)
            print("Language model loaded!")
        else:
            print(f"Using API for language model: {self.api_url}")
        
        # Initialize TTS if enabled
        if self.speak_summary:
            print("Initializing MeloTTS...")
            self.tts_model = TTS(language="FR", device=self.tts_device)
            self.speaker_id = self.tts_model.hps.data.spk2id["FR"]
            # Warm up TTS model
            _ = self.tts_model.tts_to_file("Warming up", self.speaker_id, quiet=True)
            print("MeloTTS initialized!")
        
        # Cache for storing transcript contents
        self.transcript_cache = {}
        
    def read_transcript_file(self, file_path):
        """Read a transcript file and extract content without timestamps"""
        if file_path in self.transcript_cache:
            return self.transcript_cache[file_path]
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Get current time for determining age of transcriptions
            current_time = datetime.datetime.now()
            cutoff_time = current_time - datetime.timedelta(minutes=self.time_window_minutes)
            
            # Extract content by filtering by timestamp and removing timestamp markers
            transcript_content = []
            for line in lines:
                # Extract timestamp using regex
                timestamp_match = re.match(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]', line)
                if timestamp_match:
                    timestamp_str = timestamp_match.group(1)
                    timestamp = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                    
                    # Only include lines from within the time window
                    if timestamp >= cutoff_time:
                        # Remove timestamp pattern and get content
                        content = re.sub(r'\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]\s*', '', line).strip()
                        if content:
                            transcript_content.append(content)
            
            result = "\n".join(transcript_content)
            self.transcript_cache[file_path] = result
            return result
            
        except Exception as e:
            print(f"Error reading transcript file {file_path}: {e}")
            return ""
    
    def create_prompt(self, transcript_text):
        """Create a prompt for the language model based on the transcript"""
        prompt = "Please make a short summary of the main points from the following transcript. Use the same language as the transcript:\n"       
        prompt += transcript_text
        prompt += "\nSummary:"
        return prompt
    
    def generate_summary(self, transcript_text):
        """Generate a summary of the transcript using MLX LM or API"""
        try:
            prompt = self.create_prompt(transcript_text)
            
            if self.use_api:
                # Use the external API (Gemini) for summarization
                summary = self.generate_summary_api(prompt)
            else:
                # Generate the summary using local model
                output = generate(
                    self.model,
                    self.tokenizer,
                    prompt=prompt,
                    max_tokens=512,
                    verbose=False
                )
                # Clean up the output
                summary = output.strip()
                
                # Clear MPS cache if using MPS
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                    
            return summary, prompt
            
        except Exception as e:
            print(f"Error generating summary: {e}")
            return "Error generating summary.", "Error generating prompt."
    
    def generate_summary_api(self, prompt):
        """Generate a summary using the Gemini API"""
        try:
            headers = {
                "Content-Type": "application/json"
            }
            
            # Prepare the request payload
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }]
            }
            
            # Construct the URL with the API key
            url = f"{self.api_url}?key={self.api_key}"
            
            # Make the API request
            response = requests.post(url, headers=headers, json=payload)
            
            # Check for successful response
            if response.status_code == 200:
                result = response.json()
                # Extract the generated text from the response
                if 'candidates' in result and len(result['candidates']) > 0:
                    content = result['candidates'][0]['content']
                    if 'parts' in content and len(content['parts']) > 0:
                        return content['parts'][0]['text']
                
                # Fallback if the expected structure is not found
                return str(result)
            else:
                print(f"API request failed with status code {response.status_code}")
                print(f"Response: {response.text}")
                return f"Error: API request failed with status code {response.status_code}"
                
        except Exception as e:
            print(f"Error calling API: {e}")
            return f"Error calling API: {str(e)}"
    
    def speak_text(self, text):
        """Use MeloTTS to speak out the text"""
        if not self.speak_summary:
            return
        
        try:
            # Prepare intro text
            intro_text = "The summary of transcriptions from the last 15 minutes is: "
            full_text = intro_text + text
            
            print(f"Speaking summary...")
            
            # Generate audio with MeloTTS
            audio_data = self.tts_model.tts_to_file(full_text, self.speaker_id, quiet=True)
            
            # MeloTTS returns 44.1kHz audio, we need to resample for playback
            audio_resampled = librosa.resample(audio_data, orig_sr=44100, target_sr=22050)
            
            # Convert to float32 for sounddevice
            audio_float32 = audio_resampled.astype(np.float32)
            
            # Play the audio
            sd.play(audio_float32, 22050)
            sd.wait()
            
            # Clear MPS cache if needed
            if self.tts_device == "mps" and torch.backends.mps.is_available():
                torch.mps.empty_cache()
                
            print("Finished speaking summary")
            
        except Exception as e:
            print(f"Error speaking summary: {e}")
    
    def save_summary(self, summary_data, files):
        """Save the generated summary to a file"""
        summary, prompt = summary_data
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.summaries_dir / f"summary_{timestamp}.json"
        
        # Create a metadata dictionary
        metadata = {
            "timestamp": timestamp,
            "source_files": [str(f) for f in files],
            "prompt": prompt,
            "summary": summary
        }
        
        # Write the metadata and summary to a JSON file
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"Summary saved to {summary_file}")
        
        # Also print the summary
        print("\n=== SUMMARY ===")
        print(summary)
        print("==============\n")
        
        # Speak the summary if enabled
        if self.speak_summary:
            self.speak_text(summary)
    
    def process_all_transcripts(self):
        """Process all transcription files from the specified time window"""
        # Get all transcript files
        transcript_files = list(self.transcripts_dir.glob("transcription_*.txt"))
        
        if not transcript_files:
            print("No transcript files found.")
            return
            
        # Combine all transcripts from within the time window
        all_text = ""
        processed_files = []
        
        for file_path in transcript_files:
            transcript_text = self.read_transcript_file(file_path)
            if transcript_text.strip():
                all_text += transcript_text + " "
                processed_files.append(file_path)
        
        # Generate summary if there's text to summarize
        if all_text.strip():
            print(f"Processing {len(processed_files)} transcript file(s) from the last {self.time_window_minutes} minutes")
            summary, prompt = self.generate_summary(all_text)
            self.save_summary((summary, prompt), processed_files)
        else:
            print(f"No content found within the {self.time_window_minutes} minute window.")
    
    def run(self):
        """Run the summarization process"""
        print(f"Monitoring transcript directory: {self.transcripts_dir}")
        print(f"Summaries will be saved to: {self.summaries_dir}")
        print(f"Summary interval: {self.min_interval/60}-{self.max_interval/60} minutes")
        print(f"Time window: {self.time_window_minutes} minutes")
        print(f"Speaking summaries: {self.speak_summary}")
        print(f"Using API: {self.use_api}")
        
        try:
            while True:
                # Process all transcripts from the time window
                self.process_all_transcripts()
                
                # Clear the transcript cache to ensure fresh content next time
                self.transcript_cache = {}
                
                # Sleep to avoid high CPU usage
                random_interval = random.uniform(self.min_interval, self.max_interval)
                interval_mins = random_interval / 60
                # Report next interval
                print(f"Next summary in {interval_mins:.1f} minutes")

                time.sleep(random_interval)
                
        except KeyboardInterrupt:
            print("Stopping...")
        
        print("Stopped.")


def main():
    parser = argparse.ArgumentParser(description="Transcription summarizer using MLX LM")
    
    # Directory parameters
    parser.add_argument("--transcripts-dir", type=str, default="transcripts", 
                        help="Directory containing transcription files")
    parser.add_argument("--summaries-dir", type=str, default="summaries", 
                        help="Directory to save summary files")
    
    # Timing parameters
    parser.add_argument("--min-interval", type=float, default=5, 
                        help="Minimum interval between summaries (minutes)")
    parser.add_argument("--max-interval", type=float, default=12, 
                        help="Maximum interval between summaries (minutes)")
    parser.add_argument("--time-window", type=int, default=15,
                        help="Time window for processing transcripts (minutes)")
    
    # Model parameters
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct-1M", 
                        help="MLX language model name")
    
    # API parameters
    parser.add_argument("--use-api", action="store_true", 
                        help="Use external API instead of local model")
    parser.add_argument("--api-key", type=str, default="", 
                        help="API key for the language model service")
    parser.add_argument("--api-url", type=str, 
                        default="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent", 
                        help="URL for the language model API")
    
    # TTS parameters
    parser.add_argument("--speak-summary", action="store_true", default=True,
                        help="Speak out the summary using MeloTTS")
    parser.add_argument("--tts-device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu",
                        help="Device to use for TTS model (mps or cpu)")
    
    args = parser.parse_args()
    
    # Validate API settings
    if args.use_api and not args.api_key:
        parser.error("--api-key is required when using --use-api")
    
    summarizer = TranscriptionSummarizer(args)
    summarizer.run()


if __name__ == "__main__":
    main() 