"""
Abstractive Summarization Module
Implements transformer-based abstractive summarization using BART, T5, and Pegasus.
"""

from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class AbstractiveSummarizer:
    """Abstractive text summarization using transformer models."""

    def __init__(self, model_name: str = 'facebook/bart-large-cnn', device: str = 'auto'):
        """
        Initialize the abstractive summarizer.

        Args:
            model_name: Hugging Face model name
                       - 'facebook/bart-large-cnn' (default, good balance)
                       - 'google/pegasus-xsum' (shorter summaries)
                       - 't5-base' (versatile)
                       - 'google/pegasus-cnn_dailymail' (news focused)
            device: 'cpu', 'cuda', or 'auto'
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self._model_loaded = False

    def _load_model(self):
        """Lazy load the model to save memory."""
        if self._model_loaded:
            return

        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
            import torch

            # Determine device
            if self.device == 'auto':
                device = 0 if torch.cuda.is_available() else -1
            elif self.device == 'cuda':
                device = 0
            else:
                device = -1

            print(f"Loading model: {self.model_name} on device: {'GPU' if device >= 0 else 'CPU'}")

            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

            # Create pipeline
            self.pipeline = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device
            )

            self._model_loaded = True
            print("Model loaded successfully!")

        except ImportError:
            raise ImportError(
                "Transformers library not installed. "
                "Install with: pip install transformers torch"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def summarize(self, text: str, max_length: int = 130,
                 min_length: int = 30, do_sample: bool = False,
                 num_beams: int = 4, length_penalty: float = 2.0,
                 early_stopping: bool = True) -> Dict:
        """
        Generate abstractive summary.

        Args:
            text: Input text to summarize
            max_length: Maximum length of summary
            min_length: Minimum length of summary
            do_sample: Whether to use sampling
            num_beams: Number of beams for beam search
            length_penalty: Length penalty for beam search
            early_stopping: Whether to stop early in beam search

        Returns:
            Dictionary with summary and metadata
        """
        self._load_model()

        try:
            # Check text length
            token_count = len(self.tokenizer.encode(text))

            # Adjust max_length if text is short
            if token_count < max_length:
                max_length = max(min_length, int(token_count * 0.5))

            # Generate summary
            result = self.pipeline(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                num_beams=num_beams,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
                truncation=True
            )

            summary_text = result[0]['summary_text']

            return {
                'summary': summary_text,
                'method': 'abstractive',
                'model': self.model_name,
                'original_tokens': token_count,
                'summary_tokens': len(self.tokenizer.encode(summary_text)),
                'compression_ratio': len(summary_text) / len(text) if text else 0,
                'parameters': {
                    'max_length': max_length,
                    'min_length': min_length,
                    'num_beams': num_beams,
                    'length_penalty': length_penalty
                }
            }

        except Exception as e:
            return {
                'summary': '',
                'method': 'abstractive',
                'model': self.model_name,
                'error': str(e),
                'success': False
            }

    def summarize_long_text(self, text: str, chunk_size: int = 1024,
                           max_length: int = 130, min_length: int = 30) -> Dict:
        """
        Summarize long text by chunking and combining summaries.

        Args:
            text: Input text to summarize
            chunk_size: Size of text chunks in tokens
            max_length: Maximum length of each chunk summary
            min_length: Minimum length of each chunk summary

        Returns:
            Dictionary with summary and metadata
        """
        self._load_model()

        try:
            # Tokenize text
            tokens = self.tokenizer.encode(text, truncation=False)

            # Split into chunks
            chunks = []
            for i in range(0, len(tokens), chunk_size):
                chunk_tokens = tokens[i:i + chunk_size]
                chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                chunks.append(chunk_text)

            # Summarize each chunk
            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                result = self.summarize(
                    chunk,
                    max_length=max_length,
                    min_length=min_length
                )
                if result.get('summary'):
                    chunk_summaries.append(result['summary'])

            # Combine chunk summaries
            combined_summary = ' '.join(chunk_summaries)

            # If combined summary is still too long, summarize again
            combined_tokens = len(self.tokenizer.encode(combined_summary))
            if combined_tokens > chunk_size:
                final_result = self.summarize(
                    combined_summary,
                    max_length=max_length * 2,
                    min_length=min_length
                )
                final_summary = final_result['summary']
            else:
                final_summary = combined_summary

            return {
                'summary': final_summary,
                'method': 'abstractive_chunked',
                'model': self.model_name,
                'original_tokens': len(tokens),
                'num_chunks': len(chunks),
                'chunk_summaries': chunk_summaries,
                'final_summary': final_summary
            }

        except Exception as e:
            return {
                'summary': '',
                'method': 'abstractive_chunked',
                'model': self.model_name,
                'error': str(e),
                'success': False
            }

    def batch_summarize(self, texts: List[str], **kwargs) -> List[Dict]:
        """
        Summarize multiple texts in batch.

        Args:
            texts: List of texts to summarize
            **kwargs: Arguments to pass to summarize()

        Returns:
            List of summary dictionaries
        """
        self._load_model()
        return [self.summarize(text, **kwargs) for text in texts]

    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'loaded': self._model_loaded,
            'supported_models': [
                'facebook/bart-large-cnn',
                'google/pegasus-xsum',
                'google/pegasus-cnn_dailymail',
                't5-base',
                't5-small',
                'facebook/bart-base'
            ]
        }

    def unload_model(self):
        """Unload model from memory."""
        if self._model_loaded:
            del self.model
            del self.tokenizer
            del self.pipeline
            self._model_loaded = False

            # Clear CUDA cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
