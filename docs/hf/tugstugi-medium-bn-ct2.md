---
license: apache-2.0
language:
  - bn
library_name: ctranslate2
base_model: bengaliAI/tugstugi_bengaliai-asr_whisper-medium
pipeline_tag: automatic-speech-recognition
tags:
  - whisper
  - faster-whisper
  - ctranslate2
  - bangla
  - bengali
---

# tugstugi-medium-bn (CTranslate2, float16)

A [CTranslate2](https://github.com/OpenNMT/CTranslate2) conversion of
[`bengaliAI/tugstugi_bengaliai-asr_whisper-medium`](https://huggingface.co/bengaliAI/tugstugi_bengaliai-asr_whisper-medium),
a Whisper medium model fine-tuned for Bangla (Bengali) speech recognition by
bengaliAI. All credit for the model goes to its original authors; this repo
only changes its format so that [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
can load it without PyTorch.

It is the Bangla model that [bytx](https://github.com/hades874/bytx), a free
local tool for transcribing Bangla and English YouTube and Facebook videos,
downloads during `bytx setup`.

## Changes from the original

The weights were converted to CTranslate2 format with float16 quantization.
Nothing was retrained or fine-tuned. The conversion is equivalent to:

```
ct2-transformers-converter --model bengaliAI/tugstugi_bengaliai-asr_whisper-medium \
    --output_dir tugstugi-medium-bn --quantization float16 \
    --copy_files tokenizer.json preprocessor_config.json
```

(bytx's `scripts/convert_model.py` runs the same conversion.)

## Use with faster-whisper

```python
from faster_whisper import WhisperModel

model = WhisperModel("bytx/tugstugi-medium-bn-ct2", device="cuda", compute_type="float16")
segments, _ = model.transcribe("audio.mp3", language="bn", without_timestamps=True)
print(" ".join(s.text for s in segments))
```

On a CPU use `device="cpu", compute_type="int8"`.

## Notes from using it in bytx

- The model does not reliably predict Whisper's timestamp tokens, so bytx
  decodes with `without_timestamps=True` and takes timestamps from its own
  voice-activity clips.
- Bangla needs about 0.7 tokens per character, so a 30-second Whisper window
  can overflow the 224-token limit and lose text. bytx cuts speech into clips
  of at most 12 seconds and decodes each one separately.
- Priming the decoder with an `initial_prompt` can make the model repeat the
  prompt back.

## License

Apache-2.0, the same as the original model. The original model card and
license are at
[`bengaliAI/tugstugi_bengaliai-asr_whisper-medium`](https://huggingface.co/bengaliAI/tugstugi_bengaliai-asr_whisper-medium).
