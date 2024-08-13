# Speech To Speech: an effort for an open-sourced and modular GPT4-o

## Approach

### structure
This repo implements a speech to speech cascaded pipeline, with consecutive parts: 
1. Voice Activity Detection (VAD): [silero VAD v5](https://github.com/snakers4/silero-vad)
2. Speech to Text (STT): Whisper models (including distilled ones)
3. Language Model (LM): whatever instruct model that is on the hub! 🤗
4. Text to Speech (TTS): [Parler-TTS](https://github.com/huggingface/parler-tts)

### modularity
It aims to come with a fully open and modulable approach, leveraging through the Transformers library models already avaible on the Hugging Face hub. 
The level of modularity intented for each part is the following:
- VAD: we should stick with the proposed implementation mainly taken from [Silero's repo](https://github.com/snakers4/silero-vad)
- STT: we should stick with Whisper. Nevertheless, every Whisper checkpoint can be used, enabling the usage of distil-whisper and french-distil-whisper for example
- LM: this part is fully modulable and can be changed by simply modifying the Hugging Face hub model id. Nevertheless, one need to pick and instruct model since usage here is chatting with the model.
- TTs: we should stick with Parler-TTS mini architecture. Nevertheless, different checkpoints of the model can be used, notably enabling usage of fine-tuned multilingual checkpoints

Moreover, the pipeline is designed and code written with the intent of making it as easy as possible to be modified. Each part is implemented as a class and can be easily re implemened to match your needs.

## Setup

Clone the repo:
```bash
git clone https://github.com/eustlb/speech-to-speech.git
cd speech-to-speech
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

This pipeline is intented to be run in two ways:
- on a server/ client manner, where models run a server and audio input/ output are streamed from a client
- locally, where we use the same approach as the client/server by simply using the loopback address

### server/ client approach

To run the pipeline, first run on the server:
```bash
python s2s_pipeline.py --recv_host "0.0.0.0" --send_host "0.0.0.0"
```

Then run the client locally that will handle sending microphone input and receiving generated audio:
```bash
python listen_and_play.py --host *ip adress of your server*
```

### local approach 
Simply use the loopback address.
Run the pipeline:
```bash
python s2s_pipeline.py --recv_host localhost --send_host localhost
```

Run the client:
```bash
python listen_and_play.py --host localhost
```

## Command-line usage

### model parameters

`model_name`, `torch_dtype` and `device` are exposed for each parts leveraging the Transformers' implementations: Speech To Text, Language Model, Text to Speech.
Simply specify the targeted pipeline part with the corresponding prefix:
- `stt` (Speech To Text)
- `lm` (Language Model)
- `tts` (Text To Speech)

For example, `--lm_model_name google/gemma-2b-it`.

### generation parameters

Likewise, other generation parameter of the model's generate method can be set using the part's prefix + `_gen_`, e.g. `--stt_gen_max_new_tokens 128`.
If not already exposed, they can be easily added to the pipeline part's arguments class (see for example the `LanguageModelHandlerArguments` class)

### notable parameters


#### VAD parameters

- `--thresh`: threshold value to trigger voice activity detection.
- `--min_speech_ms`: minimum duration of an detected voice activity to be considered as speech.
- `--min_silence_ms`: minimum length of silence intervals to be used for segmenting speech. Needs to be high enough to cut a sentence but low enough to reduce latency. 


#### Language Model 

- `--init_chat_role`: Defaults to `None`. Initial role in the chat template if there's any. Refer to the model's card to set this value (e.g. for 
[Phi-3-mini-4k-instruct]((https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)) you have to set `--init_chat_role system`)
- `--init_chat_prompt` Needs to be set when setting `--init_chat_role`. Defaults to `"You are a helpful AI assistant."`



#### Speech To Text
- `--description`: sets the description of Parler-TTS generated voice (see [here] for more details), defaults to: `"A female speaker with a slightly low-pitched voice delivers her words quite expressively, in a very confined sounding environment with clear audio quality. She speaks very fast."`

- `--play_steps_s`: duration of first chunk to be sent during streaming the output of Parler-TTS. A lower value mean the first chunk is ready faster, but will require more codec decoding steps overall. This value should be tuned to your device and latency requirements.



## Citations

### Distil-Whisper

```
@misc{gandhi2023distilwhisper,
      title={Distil-Whisper: Robust Knowledge Distillation via Large-Scale Pseudo Labelling}, 
      author={Sanchit Gandhi and Patrick von Platen and Alexander M. Rush},
      year={2023},
      eprint={2311.00430},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

### Silero VAD
```
@misc{Silero VAD,
  author = {Silero Team},
  title = {Silero VAD: pre-trained enterprise-grade Voice Activity Detector (VAD), Number Detector and Language Classifier},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/snakers4/silero-vad}},
  commit = {insert_some_commit_here},
  email = {hello@silero.ai}
}
```
