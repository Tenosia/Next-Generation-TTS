# DetailTTS: Learning Residual Detail Information for Zero-shot Text-to-speech

## Overview

DetailTTS is a zero-shot text-to-speech synthesis system that learns residual detail information to improve audio reconstruction quality. Unlike traditional VQ (Vector Quantization) based methods that struggle with audio reconstruction, DetailTTS models residual details through learnable embeddings and duration modeling, achieving superior results.

## Features

- **Zero-shot TTS**: Generate speech in any voice using just a reference audio
- **Multi-language Support**: Chinese (ZH), Japanese (JP), English (EN), and Korean (KR)
- **High Quality**: Superior audio reconstruction compared to traditional VQ-based methods
- **Easy to Use**: Simple Python API for inference
- **Extensible**: Easy to add support for new languages

## Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/adelacvg/ttts.git
cd ttts
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
pip install -e .
```

3. **Download pre-trained models:**
   - Visit [Pre-trained Models](https://huggingface.co/adelacvg/TTTS_v4/tree/main)
   - Download the VQVAE model checkpoint

### Basic Usage

```python
from ttts.api import DetailTTS

# Initialize the model
tts = DetailTTS(
    model_path='path/to/vqvae_model.pt',
    config_path='ttts/vqvae/config_v3.json',
    device='cuda:0'  # or 'cpu'
)

# Synthesize speech
audio = tts.synthesize(
    text="Hello, this is a test.",
    reference_audio='path/to/reference.wav',
    lang='EN',
    output_path='output.wav'
)
```

### Multi-language Examples

```python
# Chinese
audio = tts.synthesize(
    text="大家好，今天来点大家想看的东西。",
    reference_audio='reference.wav',
    lang='ZH',
    output_path='gen_zh.wav'
)

# Japanese
audio = tts.synthesize(
    text='皆さん、こんにちは！',
    reference_audio='reference.wav',
    lang='JP',
    output_path='gen_jp.wav'
)

```

## 📚 Documentation

### API Reference

#### `DetailTTS` Class

Main class for text-to-speech synthesis.

**Parameters:**
- `model_path` (str): Path to the VQVAE model checkpoint
- `config_path` (str, optional): Path to model configuration JSON. Default: `'ttts/vqvae/config_v3.json'`
- `device` (str, optional): Device to run inference on ('cuda:0', 'cpu', etc.). Auto-detects if None.
- `tokenizer_dir` (str, optional): Directory containing tokenizer files. Default: `'ttts/tokenizers'`

**Methods:**

##### `synthesize(text, reference_audio, lang='ZH', output_path=None, sample_rate=32000)`

Synthesize speech from text using a reference audio.

**Parameters:**
- `text` (str): Input text to synthesize
- `reference_audio` (str or torch.Tensor): Path to reference audio file or audio tensor
- `lang` (str): Language code ('ZH', 'JP', 'EN', 'KR')
- `output_path` (str, optional): Path to save output audio
- `sample_rate` (int): Output sample rate. Default: 32000

**Returns:**
- `torch.Tensor`: Generated audio waveform

## Project Structure

```
ttts/
├── api.py                 # Main inference API
├── vqvae/                 # VQ-VAE model implementation
│   ├── train_v3.py        # Training script
│   ├── config_v3.json     # Model configuration
│   └── ...
├── gpt/                   # GPT model for text processing
│   ├── model.py
│   ├── train.py
│   └── voice_tokenizer.py
├── diffusion/             # Diffusion model components
├── prepare/               # Data preprocessing scripts
│   ├── 1_vad_asr_save_to_jsonl.py
│   ├── 2_romanize_text.py
│   └── ...
├── tokenizers/            # Language tokenizers
│   ├── zh_tokenizer.json
│   ├── en_tokenizer.json
│   ├── jp_tokenizer.json
│   └── kr_tokenizer.json
└── utils/                 # Utility functions
```

## Training

### 1. Tokenizer Training

First, merge all text data:
```bash
python ttts/prepare/bpe_all_text_to_one_file.py
```

Then train the tokenizer. Check `ttts/gpt/voice_tokenizer` for more information.

### 2. Data Preprocessing

Preprocess your dataset:
```bash
# Step 1: VAD and ASR
python ttts/prepare/1_vad_asr_save_to_jsonl.py

# Step 2: Romanize text
python ttts/prepare/2_romanize_text.py
```

### 3. VQVAE Training

Train the VQVAE model:
```bash
accelerate launch ttts/vqvae/train_v3.py
```

### 4. Fine-tuning

To fine-tune on your own dataset:
1. Update the model path in `train_v3.py` to point to a pre-trained model
2. Prepare your dataset following the preprocessing steps
3. Run the training script

## Multi-language Support

DetailTTS currently supports:
- **Chinese (ZH)**: Uses Pinyin for pronunciation
- **Japanese (JP)**: Uses Romaji for pronunciation
- **English (EN)**: Direct text input
- **Korean (KR)**: Uses romanization

### Adding a New Language

To add support for a new language:

1. **Collect text data** in the target language
2. **Train a tokenizer** using `ttts/gpt/voice_tokenizer`
3. **Update the language mapping** in `ttts/api.py`:
   - Add language code to `LANG_CODES`
   - Add text conversion logic in `convert_to_latin()`
   - Add tokenizer initialization

For languages that require pronunciation information (like Chinese and Japanese), ensure your text preprocessing includes phonetic information.

## Demo

Try the interactive demo:
- [👉 **Visit the Demo Page**](https://detailtts.github.io/)
- [Open in Colab](https://colab.research.google.com/github/adelacvg/ttts/blob/v4/demo.ipynb)

## 🔧 Configuration

### Model Configuration

Edit `ttts/vqvae/config_v3.json` to adjust model parameters:

```json
{
  "train": {
    "batch_size": 8,
    "learning_rate": 1e-4,
    "epochs": 100,
    ...
  },
  "vqvae": {
    "inter_channels": 128,
    "hidden_channels": 128,
    ...
  }
}
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**
   - Reduce batch size in training configuration
   - Use gradient accumulation
   - Use a smaller model

2. **Model file not found**
   - Ensure model path is correct
   - Download pre-trained models from HuggingFace

3. **Tokenizer errors**
   - Verify tokenizer files exist in `ttts/tokenizers/`
   - Check that tokenizer paths are correct

4. **Audio quality issues**
   - Ensure reference audio is clear and high quality
   - Check that sample rate matches (32kHz)
   - Verify text preprocessing is correct for the language

## Citation

If you use DetailTTS in your research, please cite:

```bibtex
@misc{detailtts,
  title={DetailTTS: Learning Residual Detail Information for Zero-shot Text-to-speech},
  author={DetailTTS Team},
  year={2024},
  url={https://github.com/adelacvg/ttts}
}
```

## Acknowledgements

- [Tortoise](https://github.com/neonbjb/tortoise-tts) - Initial inspiration and codebase
- [VITS](https://github.com/jaywalnut310/vits) - Core architecture foundation
- [Bert-VITS2](https://github.com/fishaudio/Bert-VITS2) - Optimized code contributions (e.g., MAS implementation)

## License

See [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions and issues, please open an issue on GitHub.

---

**Note**: This project is actively maintained. For the latest updates, please check the repository.
