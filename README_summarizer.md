# Transcription Summarizer

A tool that periodically monitors transcription files, processes them at configurable intervals, and generates concise summaries of transcriptions from the last 15 minutes using MLX with the Qwen2.5-7B-Instruct-1M language model. The summaries are both saved to disk and spoken aloud using MeloTTS.

## Features

- Monitors a directory for new transcription files
- Periodically generates summaries of accumulated transcripts
- **Only summarizes transcriptions from the last 15 minutes based on their timestamps**
- Removes timestamps from transcripts before summarization
- Includes the full prompt sent to the language model in the output
- Saves summaries with metadata in JSON format
- Speaks the summaries aloud using MeloTTS
- Optimized for Mac with Metal Performance Shaders (MPS)

## Requirements

- Python 3.8+
- macOS (optimized for Apple Silicon with MPS)
- All dependencies from requirements_simple.txt
- Working audio output device

## Usage

Run alongside the speech transcription pipeline:

```bash
# First, start the speech transcription in one terminal:
python mic_transcriber.py

# Then, in another terminal, start the summarizer:
python transcription_summarizer.py
```

### Command Line Options

- `--transcripts-dir`: Directory containing transcription files (default: "transcripts")
- `--summaries-dir`: Directory to save summary files (default: "summaries")
- `--min-interval`: Minimum interval between summaries in minutes (default: 5)
- `--max-interval`: Maximum interval between summaries in minutes (default: 12)
- `--model-name`: MLX language model to use (default: "Qwen/Qwen2.5-7B-Instruct-1M")
- `--speak-summary`: Enable/disable speaking the summary (default: enabled)
- `--tts-device`: Device to use for TTS model (default: "mps" on Apple Silicon, "cpu" otherwise)

Example with custom settings:

```bash
python transcription_summarizer.py --min-interval 3 --max-interval 10 --speak-summary
```

To disable the speech output:

```bash
python transcription_summarizer.py --no-speak-summary
```

## Output

### Text Output
Summaries are saved to JSON files in the specified summaries directory with metadata including:
- Timestamp of when the summary was created
- List of source transcript files that were summarized
- The full prompt sent to the language model
- The generated summary text

Example summary JSON:
```json
{
  "timestamp": "20240331_121545",
  "source_files": [
    "transcripts/transcription_20240331_120012.txt",
    "transcripts/transcription_20240331_120345.txt"
  ],
  "prompt": "Please summarize the following conversation transcript concisely. Use the same language as the transcript:\nHello, I'm calling about the project deadline. Can we extend it by a week? Yes, that should be fine. I'll update the calendar. Thank you, I appreciate it.\nSummary:",
  "summary": "The caller requested a one-week extension for the project deadline, which was approved, and the calendar will be updated accordingly."
}
```

### Speech Output
The summary is spoken aloud with an introduction like "The summary of transcriptions from the last 15 minutes is: [summary text]". This uses MeloTTS, an efficient text-to-speech engine optimized for Mac.

## How It Works

1. The summarizer runs on a timer to process transcription files
2. At intervals between min_interval and max_interval, it processes all pending transcript files
3. For each batch of transcripts:
   - It filters transcriptions by timestamp, keeping only those from the last 15 minutes
   - It removes timestamps from filtered transcripts
   - Creates an appropriate prompt
   - Generates a summary using the MLX language model
   - Saves both the prompt and summary with metadata
   - Speaks the summary using MeloTTS
4. After processing, it waits for the next interval

## Troubleshooting

- If you encounter memory issues, try running with a lower `--max-interval` to process smaller batches of transcripts
- For more detailed logging, modify the script to add more verbose print statements
- If the model download is slow or fails, ensure you have a good internet connection for the first run
- If there are issues with the speech output, check your audio settings and try disabling MPS with `--tts-device cpu` 