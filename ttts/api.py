"""
DetailTTS Inference API

A user-friendly API for zero-shot text-to-speech synthesis using DetailTTS.
Supports multiple languages: Chinese (ZH), Japanese (JP), English (EN), and Korean (KR).
"""

import os
import json
from pathlib import Path
from typing import Optional, Union
import torch
import torchaudio
import torchaudio.functional as audio_F
from pypinyin import lazy_pinyin, Style
import cutlet
from hangul_romanize import Transliter
from hangul_romanize.rule import academic

from ttts.gpt.voice_tokenizer import VoiceBpeTokenizer
from ttts.utils.infer_utils import load_model
from ttts.utils.data_utils import spectrogram_torch, HParams


class DetailTTS:
    """Main API class for DetailTTS inference."""
    
    # Language codes mapping
    LANG_CODES = {
        'ZH': 0,  # Chinese
        'JP': 1,  # Japanese
        'EN': 2,  # English
        'KR': 3,  # Korean
    }
    
    def __init__(
        self,
        model_path: str,
        config_path: str = 'ttts/vqvae/config_v3.json',
        device: Optional[str] = None,
        tokenizer_dir: str = 'ttts/tokenizers'
    ):
        """
        Initialize DetailTTS model.
        
        Args:
            model_path: Path to the VQVAE model checkpoint
            config_path: Path to the model configuration JSON file
            device: Device to run inference on ('cuda:0', 'cpu', etc.). 
                   If None, automatically selects CUDA if available.
            tokenizer_dir: Directory containing tokenizer files
        """
        # Auto-detect device if not specified
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        self.config_path = config_path
        
        # Load configuration
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.hps = HParams(**config)
        
        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading model from {model_path}...")
        self.model = load_model('vqvae', model_path, config_path, device)
        self.model.eval()
        print("Model loaded successfully!")
        
        # Initialize tokenizers
        self.tokenizer_dir = tokenizer_dir
        self._init_tokenizers()
        
        # Initialize text converters
        self.katsu = cutlet.Cutlet()
        self.kr_transliter = Transliter(academic)
    
    def _init_tokenizers(self):
        """Initialize tokenizers for all supported languages."""
        self.tokenizers = {}
        for lang in ['zh', 'en', 'jp', 'kr']:
            tokenizer_path = os.path.join(self.tokenizer_dir, f'{lang}_tokenizer.json')
            if os.path.exists(tokenizer_path):
                self.tokenizers[lang.upper()] = VoiceBpeTokenizer(tokenizer_path)
            else:
                print(f"Warning: Tokenizer not found for {lang}: {tokenizer_path}")
    
    def convert_to_latin(self, text: str, lang: str) -> Optional[str]:
        """
        Convert text to Latin script for tokenization.
        
        Args:
            text: Input text
            lang: Language code ('ZH', 'JP', 'EN', 'KR')
            
        Returns:
            Converted text in Latin script, or None if language not supported
        """
        lang = lang.upper()
        
        if lang == "ZH":
            text = ' '.join(lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True))
            text = ' ' + text + ' '
        elif lang == "JP":
            text = self.katsu.romaji(text)
            text = ' ' + text + ' '
        elif lang == "EN":
            text = ' ' + text + ' '
        elif lang == "KR":
            text = self.kr_transliter.translit(text)
            text = ' ' + text + ' '
        else:
            return None
        return text
    
    def synthesize(
        self,
        text: str,
        reference_audio: Union[str, torch.Tensor],
        lang: str = 'ZH',
        output_path: Optional[str] = None,
        sample_rate: int = 32000
    ) -> torch.Tensor:
        """
        Synthesize speech from text using a reference audio.
        
        Args:
            text: Input text to synthesize
            reference_audio: Path to reference audio file or audio tensor
            lang: Language code ('ZH', 'JP', 'EN', 'KR')
            output_path: Optional path to save output audio. If None, audio is not saved.
            sample_rate: Output sample rate (default: 32000)
            
        Returns:
            Generated audio waveform as torch.Tensor
        """
        lang = lang.upper()
        
        if lang not in self.LANG_CODES:
            raise ValueError(f"Unsupported language: {lang}. Supported: {list(self.LANG_CODES.keys())}")
        
        if lang not in self.tokenizers:
            raise ValueError(f"Tokenizer not available for language: {lang}")
        
        # Convert text to Latin and tokenize
        latin_text = self.convert_to_latin(text, lang)
        if latin_text is None:
            raise ValueError(f"Failed to convert text to Latin for language: {lang}")
        
        tokenizer = self.tokenizers[lang]
        text_tokens = tokenizer.encode(latin_text.lower())
        text_tokens = torch.LongTensor(text_tokens).unsqueeze(0)
        
        # Add language offset
        lang_id = self.LANG_CODES[lang]
        text_tokens = text_tokens + 256 * lang_id
        text_tokens = text_tokens.to(self.device)
        text_length = torch.LongTensor([text_tokens.shape[1]]).to(self.device)
        
        # Load and process reference audio
        if isinstance(reference_audio, str):
            wav, sr = torchaudio.load(reference_audio)
        else:
            wav = reference_audio
            sr = sample_rate
        
        # Convert to mono if stereo
        if wav.shape[0] > 1:
            wav = wav[0].unsqueeze(0)
        
        wav = wav.to(self.device)
        
        # Resample to 32kHz
        wav32k = audio_F.resample(wav, sr, 32000)
        wav32k = wav32k[:, :int(self.hps.data.hop_length * (wav32k.shape[-1] // self.hps.data.hop_length))]
        wav = torch.clamp(wav32k, min=-1.0, max=1.0)
        
        # Compute spectrogram
        spec = spectrogram_torch(
            wav,
            self.hps.data.filter_length,
            self.hps.data.hop_length,
            self.hps.data.win_length,
            center=False
        ).squeeze(0)
        
        wav_length = torch.LongTensor([wav.shape[1]])
        spec_length = torch.LongTensor([x // self.hps.data.hop_length for x in wav_length]).to(self.device)
        
        # Generate audio
        with torch.no_grad():
            wav_out = self.model.infer(
                text_tokens,
                text_length,
                spec.unsqueeze(0),
                spec_length
            )
        
        # Save if output path is provided
        if output_path:
            torchaudio.save(output_path, wav_out.squeeze(0).cpu(), sample_rate)
            print(f"Audio saved to {output_path}")
        
        return wav_out.squeeze(0).cpu()


# Example usage
if __name__ == "__main__":
    # Configuration
    MODEL_PATH = os.getenv('TTTS_MODEL_PATH', '/path/to/model.pt')
    CONFIG_PATH = 'ttts/vqvae/config_v3.json'
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # Example 1: Chinese
    if os.path.exists(MODEL_PATH):
        tts = DetailTTS(
            model_path=MODEL_PATH,
            config_path=CONFIG_PATH,
            device=DEVICE
        )
        
        # Synthesize Chinese text
        audio = tts.synthesize(
            text="大家好，今天来点大家想看的东西。",
            reference_audio='ttts/6.wav',
            lang='ZH',
            output_path='gen_zh.wav'
        )
        
        # Example 2: Japanese
        # audio = tts.synthesize(
        #     text='皆さん、こんにちは！今日は皆さんに読んでいただきたいことがあります。',
        #     reference_audio='ttts/jp.wav',
        #     lang='JP',
        #     output_path='gen_jp.wav'
        # )
        
        # Example 3: English
        # audio = tts.synthesize(
        #     text="Hello everyone, here's something you'll want to read today.",
        #     reference_audio='ttts/en.wav',
        #     lang='EN',
        #     output_path='gen_en.wav'
        # )
        
        # Example 4: Korean
        # audio = tts.synthesize(
        #     text="안녕하세요, 여러분, 오늘 읽어보시면 좋을 내용이 있습니다.",
        #     reference_audio='ttts/kr.wav',
        #     lang='KR',
        #     output_path='gen_kr.wav'
        # )
    else:
        print(f"Please set MODEL_PATH environment variable or update the path in the script.")
        print(f"Model path: {MODEL_PATH}")
