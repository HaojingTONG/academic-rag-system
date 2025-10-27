"""
Image Caption Extractor Module
Provides OCR and image captioning capabilities for figure extraction
Supports PaddleOCR/Tesseract for OCR and BLIP/CLIP for image captioning
"""

import logging
from typing import Dict, Optional, List
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)

class ImageCaptionExtractor(ABC):
    """Abstract base class for image caption extraction"""

    @abstractmethod
    def extract_text(self, image_path: str) -> str:
        """Extract text from image using OCR"""
        pass

    @abstractmethod
    def generate_caption(self, image_path: str) -> str:
        """Generate descriptive caption for image"""
        pass

    def extract_all(self, image_path: str) -> Dict[str, str]:
        """Extract both OCR text and caption"""
        result = {
            'ocr_text': '',
            'image_caption': '',
            'success': False,
            'error': None
        }

        try:
            if Path(image_path).exists():
                result['ocr_text'] = self.extract_text(image_path)
                result['image_caption'] = self.generate_caption(image_path)
                result['success'] = True
            else:
                result['error'] = f"Image file not found: {image_path}"
                logger.warning(result['error'])
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error extracting from {image_path}: {e}")

        return result


class StubImageCaptionExtractor(ImageCaptionExtractor):
    """Stub implementation that logs warnings when dependencies are missing"""

    def __init__(self):
        self.ocr_available = False
        self.caption_available = False

        # Check for OCR libraries
        try:
            import paddleocr
            self.ocr_available = True
            self.ocr_backend = 'paddleocr'
        except ImportError:
            try:
                import pytesseract
                self.ocr_available = True
                self.ocr_backend = 'tesseract'
            except ImportError:
                logger.warning(
                    "No OCR library found. Install paddleocr or pytesseract for OCR support. "
                    "Falling back to stub implementation."
                )
                self.ocr_backend = None

        # Check for captioning libraries
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            self.caption_available = True
            self.caption_backend = 'blip'
        except ImportError:
            try:
                import clip
                self.caption_available = True
                self.caption_backend = 'clip'
            except ImportError:
                logger.warning(
                    "No image captioning library found. Install transformers (BLIP) or clip for captioning. "
                    "Falling back to stub implementation."
                )
                self.caption_backend = None

    def extract_text(self, image_path: str) -> str:
        """Extract text from image using OCR (stub implementation)"""
        if not self.ocr_available:
            logger.debug(f"OCR not available for {image_path}, skipping text extraction")
            return ""

        logger.info(f"OCR extraction using {self.ocr_backend} for {image_path}")
        # Stub: return empty string, actual implementation would call OCR
        return ""

    def generate_caption(self, image_path: str) -> str:
        """Generate caption for image (stub implementation)"""
        if not self.caption_available:
            logger.debug(f"Captioning not available for {image_path}, skipping caption generation")
            return ""

        logger.info(f"Caption generation using {self.caption_backend} for {image_path}")
        # Stub: return empty string, actual implementation would call captioning model
        return ""


class PaddleOCRExtractor(ImageCaptionExtractor):
    """OCR extractor using PaddleOCR"""

    def __init__(self, use_gpu: bool = False):
        try:
            from paddleocr import PaddleOCR
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=use_gpu, show_log=False)
            logger.info("PaddleOCR initialized successfully")
        except ImportError:
            raise ImportError("PaddleOCR not installed. Install with: pip install paddleocr")

    def extract_text(self, image_path: str) -> str:
        """Extract text using PaddleOCR"""
        try:
            result = self.ocr.ocr(image_path, cls=True)
            if result and result[0]:
                # Extract text from OCR results
                texts = [line[1][0] for line in result[0]]
                return ' '.join(texts)
            return ""
        except Exception as e:
            logger.error(f"PaddleOCR extraction failed for {image_path}: {e}")
            return ""

    def generate_caption(self, image_path: str) -> str:
        """PaddleOCR doesn't generate captions, return empty string"""
        return ""


class BLIPCaptionExtractor(ImageCaptionExtractor):
    """Image captioning using BLIP model"""

    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            from PIL import Image
            import torch

            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForConditionalGeneration.from_pretrained(model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.Image = Image
            logger.info(f"BLIP model initialized on {self.device}")
        except ImportError:
            raise ImportError("Transformers or PIL not installed. Install with: pip install transformers pillow")

    def extract_text(self, image_path: str) -> str:
        """BLIP doesn't do OCR, return empty string"""
        return ""

    def generate_caption(self, image_path: str) -> str:
        """Generate caption using BLIP"""
        try:
            image = self.Image.open(image_path).convert('RGB')
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs, max_length=50)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            logger.error(f"BLIP caption generation failed for {image_path}: {e}")
            return ""


class HybridImageCaptionExtractor(ImageCaptionExtractor):
    """Hybrid extractor combining OCR and captioning"""

    def __init__(self, ocr_extractor: Optional[ImageCaptionExtractor] = None,
                 caption_extractor: Optional[ImageCaptionExtractor] = None):
        """
        Initialize hybrid extractor with optional OCR and caption extractors

        Args:
            ocr_extractor: Extractor for OCR (default: PaddleOCR if available)
            caption_extractor: Extractor for captioning (default: BLIP if available)
        """
        self.ocr_extractor = ocr_extractor
        self.caption_extractor = caption_extractor

        # Try to initialize default extractors if not provided
        if self.ocr_extractor is None:
            try:
                self.ocr_extractor = PaddleOCRExtractor()
            except ImportError:
                logger.debug("PaddleOCR not available, OCR disabled")

        if self.caption_extractor is None:
            try:
                self.caption_extractor = BLIPCaptionExtractor()
            except ImportError:
                logger.debug("BLIP not available, captioning disabled")

    def extract_text(self, image_path: str) -> str:
        """Extract text using OCR extractor"""
        if self.ocr_extractor:
            return self.ocr_extractor.extract_text(image_path)
        return ""

    def generate_caption(self, image_path: str) -> str:
        """Generate caption using caption extractor"""
        if self.caption_extractor:
            return self.caption_extractor.generate_caption(image_path)
        return ""


def create_extractor(ocr_backend: str = 'auto',
                    caption_backend: str = 'auto') -> ImageCaptionExtractor:
    """
    Factory function to create appropriate image caption extractor

    Args:
        ocr_backend: 'auto', 'paddleocr', 'tesseract', or 'none'
        caption_backend: 'auto', 'blip', 'clip', or 'none'

    Returns:
        ImageCaptionExtractor instance
    """
    ocr_extractor = None
    caption_extractor = None

    # Initialize OCR extractor
    if ocr_backend == 'auto':
        try:
            ocr_extractor = PaddleOCRExtractor()
        except ImportError:
            logger.info("PaddleOCR not available, OCR disabled")
    elif ocr_backend == 'paddleocr':
        ocr_extractor = PaddleOCRExtractor()
    elif ocr_backend == 'tesseract':
        logger.warning("Tesseract backend not yet implemented")
    elif ocr_backend != 'none':
        raise ValueError(f"Unknown OCR backend: {ocr_backend}")

    # Initialize caption extractor
    if caption_backend == 'auto':
        try:
            caption_extractor = BLIPCaptionExtractor()
        except ImportError:
            logger.info("BLIP not available, captioning disabled")
    elif caption_backend == 'blip':
        caption_extractor = BLIPCaptionExtractor()
    elif caption_backend == 'clip':
        logger.warning("CLIP backend not yet implemented")
    elif caption_backend != 'none':
        raise ValueError(f"Unknown caption backend: {caption_backend}")

    # Return hybrid or stub extractor
    if ocr_extractor or caption_extractor:
        return HybridImageCaptionExtractor(ocr_extractor, caption_extractor)
    else:
        logger.warning("No extractors available, using stub implementation")
        return StubImageCaptionExtractor()


def append_image_descriptions_to_chunk(chunk, image_path: str,
                                       extractor: Optional[ImageCaptionExtractor] = None) -> None:
    """
    Append image descriptions to a chunk's text and metadata

    Args:
        chunk: Chunk object to update
        image_path: Path to image file
        extractor: ImageCaptionExtractor instance (default: auto-create)
    """
    if extractor is None:
        extractor = create_extractor()

    try:
        result = extractor.extract_all(image_path)

        if result['success']:
            descriptions = []

            if result['ocr_text']:
                descriptions.append(f"[OCR Text: {result['ocr_text']}]")
                chunk.metadata['ocr_text'] = result['ocr_text']

            if result['image_caption']:
                descriptions.append(f"[Image Description: {result['image_caption']}]")
                chunk.metadata['image_description'] = result['image_caption']

            if descriptions:
                # Append to chunk text for embedding
                chunk.text = chunk.text + "\n\n" + " ".join(descriptions)

                # Update counts
                chunk.char_count = len(chunk.text)
                chunk.word_count = len(chunk.text.split())

                logger.info(f"Added image descriptions to chunk {chunk.chunk_id}")
        else:
            logger.warning(f"Failed to extract from {image_path}: {result.get('error')}")

    except Exception as e:
        logger.error(f"Error appending image descriptions: {e}")
        # Don't fail ingestion, just log error


# Extension point for custom extractors
def register_custom_extractor(name: str, extractor_class):
    """
    Register a custom image caption extractor

    Args:
        name: Name for the extractor
        extractor_class: Class implementing ImageCaptionExtractor
    """
    # This is an extension point for future custom extractors
    logger.info(f"Registered custom extractor: {name}")
    pass
