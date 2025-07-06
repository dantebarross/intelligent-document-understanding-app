"""
Optimized OCR Service for historical and low quality documents.
Enhanced version with adaptive preprocessing and quality analysis.
"""

import os
import cv2
import numpy as np
import pytesseract
import easyocr
from PIL import Image
import PyPDF2
from io import BytesIO
import logging

logger = logging.getLogger(__name__)


class OCRServiceOptimized:
    """Optimized OCR service for historical and low quality documents."""
    
    def __init__(self):
        """Initialize optimized OCR service."""
        tesseract_cmd = os.getenv('TESSERACT_CMD', 'tesseract')
        if os.path.exists(tesseract_cmd):
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        try:
            self.easyocr_reader = easyocr.Reader(['en', 'pt'], gpu=False)
            self.easyocr_available = True
        except Exception as e:
            logger.warning(f"EasyOCR initialization failed: {e}")
            self.easyocr_available = False
    
    def extract_text(self, file_path):
        """
        Extract text from documents using optimized techniques.
        
        Args:
            file_path (str): Path to file
            
        Returns:
            str: Extracted text
        """
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.pdf':
                return self._extract_text_from_pdf(file_path)
            elif file_extension in ['.png', '.jpg', '.jpeg']:
                return self._extract_text_from_image_optimized(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            raise
    
    def _extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF."""
        extracted_text = ""
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        extracted_text += text + "\n"
                
                if not extracted_text.strip():
                    logger.warning(f"No text found in PDF {pdf_path}, treating as scanned document")
                    return ""
                    
        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {e}")
            raise
        
        return extracted_text.strip()
    
    def _extract_text_from_image_optimized(self, image_path):
        """Extract text from image using optimized techniques."""
        try:
            image_info = self._analyze_image_quality(image_path)
            processed_image = self._preprocess_image_adaptive(image_path, image_info)
            best_text = self._ocr_with_multiple_configs(processed_image)
            
            if best_text and len(best_text.strip()) > 0:
                return best_text
            
            if self.easyocr_available:
                try:
                    text_easyocr = self._ocr_with_easyocr(image_path)
                    if text_easyocr.strip():
                        return text_easyocr
                except Exception as e:
                    logger.warning(f"EasyOCR failed: {e}")
            
            return ""
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            raise
    
    def _analyze_image_quality(self, image_path):
        """Analyze image quality to choose preprocessing technique."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            mean_brightness = np.mean(gray)
            contrast = np.std(gray)
            
            return {
                'blur_score': blur_score,
                'brightness': mean_brightness,
                'contrast': contrast,
                'dimensions': gray.shape
            }
            
        except Exception as e:
            logger.error(f"Error analyzing image quality: {e}")
            return None
    
    def _preprocess_image_adaptive(self, image_path, image_info):
        """Adaptive preprocessing based on image quality analysis."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            if image_info is None:
                return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            blur_score = image_info['blur_score']
            brightness = image_info['brightness']
            contrast = image_info['contrast']
            
            if brightness > 220:  # Very bright image (old documents)
                processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            elif blur_score < 100:  # Blurred image
                processed = cv2.bilateralFilter(gray, 9, 75, 75)
            elif brightness < 120:  # Dark image
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                processed = clahe.apply(gray)
            elif contrast < 40:  # Low contrast
                processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            else:
                processed = cv2.fastNlMeansDenoising(gray)
            
            return processed
            
        except Exception as e:
            logger.error(f"Error in adaptive preprocessing: {e}")
            return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    def _ocr_with_multiple_configs(self, image):
        """OCR with multiple optimized configurations."""
        try:
            configs = [
                '--oem 3 --psm 6',  # Structured documents
                '--oem 3 --psm 4',  # Documents with columns
                '--oem 3 --psm 3',  # Continuous text
                '--oem 3 --psm 7',  # Single line
                '--oem 3 --psm 8',  # Single word
                '--oem 3 --psm 1',  # Automatic with orientation detection
            ]
            
            best_text = ""
            best_score = 0
            
            pil_image = Image.fromarray(image)
            
            for config in configs:
                try:
                    text = pytesseract.image_to_string(pil_image, config=config)
                    text = text.strip()
                    
                    if not text:
                        continue
                    
                    score = self._calculate_text_quality_score(text)
                    
                    if score > best_score:
                        best_score = score
                        best_text = text
                        
                except Exception as e:
                    continue
            
            return best_text
            
        except Exception as e:
            logger.error(f"Error in multiple config OCR: {e}")
            raise
    
    def _calculate_text_quality_score(self, text):
        """Calculate quality score of extracted text."""
        if not text:
            return 0
        
        words = text.split()
        if not words:
            return 0
        
        avg_word_length = sum(len(word) for word in words) / len(words)
        spaces_ratio = text.count(' ') / len(text)
        long_sequences = sum(1 for word in words if len(word) > 20)
        
        alpha_chars = sum(1 for c in text if c.isalpha())
        total_chars = len(text)
        alpha_ratio = alpha_chars / total_chars if total_chars > 0 else 0
        
        score = (
            len(text) * 0.3 +
            min(avg_word_length, 8) * 0.2 +
            spaces_ratio * 100 * 0.2 +
            alpha_ratio * 100 * 0.2 +
            max(0, 10 - long_sequences) * 0.1
        )
        
        return score
    
    def _ocr_with_easyocr(self, image_path):
        """OCR with EasyOCR as fallback."""
        try:
            if not self.easyocr_available:
                raise RuntimeError("EasyOCR not available")
            
            results = self.easyocr_reader.readtext(image_path)
            results.sort(key=lambda x: x[0][0][1])
            
            text_list = []
            for (bbox, text, confidence) in results:
                if confidence > 0.1:
                    text_list.append(text)
            
            return ' '.join(text_list)
            
        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
            raise
    
    def extract_text_from_bytes(self, file_bytes, file_extension):
        """Extract text from file bytes."""
        try:
            if file_extension.lower() == '.pdf':
                return self._extract_text_from_pdf_bytes(file_bytes)
            elif file_extension.lower() in ['.png', '.jpg', '.jpeg']:
                return self._extract_text_from_image_bytes(file_bytes)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            logger.error(f"Error extracting text from bytes: {e}")
            raise
    
    def _extract_text_from_pdf_bytes(self, pdf_bytes):
        """Extract text from PDF bytes."""
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
            extracted_text = ""
            
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text.strip():
                    extracted_text += text + "\n"
            
            return extracted_text.strip()
            
        except Exception as e:
            logger.error(f"Error reading PDF bytes: {e}")
            raise
    
    def _extract_text_from_image_bytes(self, image_bytes):
        """Extract text from image bytes."""
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Could not decode image from bytes")
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            
            if brightness > 220:
                processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            else:
                processed = cv2.fastNlMeansDenoising(gray)
            
            text = self._ocr_with_multiple_configs(processed)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error processing image bytes: {e}")
            raise
