"""
OCR Service for extracting text from documents.
Supports PDF and image files using Tesseract and EasyOCR.
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


class OCRService:
    """Service for Optical Character Recognition from documents."""
    
    def __init__(self):
        """Initialize OCR service with Tesseract and EasyOCR."""
        # Configure Tesseract path (Windows)
        tesseract_cmd = os.getenv('TESSERACT_CMD', 'tesseract')
        if os.path.exists(tesseract_cmd):
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        # Initialize EasyOCR reader
        try:
            self.easyocr_reader = easyocr.Reader(['en', 'pt'], gpu=False)
            self.easyocr_available = True
        except Exception as e:
            logger.warning(f"EasyOCR initialization failed: {e}")
            self.easyocr_available = False
    
    def extract_text(self, file_path):
        """
        Extract text from document file.
        
        Args:
            file_path (str): Path to the document file
            
        Returns:
            str: Extracted text
        """
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.pdf':
                return self._extract_text_from_pdf(file_path)
            elif file_extension in ['.png', '.jpg', '.jpeg']:
                return self._extract_text_from_image(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            raise
    
    def _extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file."""
        extracted_text = ""
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Try to extract text directly from PDF
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        extracted_text += text + "\n"
                
                # If no text found, it might be a scanned PDF
                if not extracted_text.strip():
                    logger.warning(f"No text found in PDF {pdf_path}, treating as scanned document")
                    # Convert PDF pages to images and OCR them
                    # This would require additional libraries like pdf2image
                    # For now, return empty string
                    return ""
                    
        except Exception as e:
            logger.error(f"Error reading PDF {pdf_path}: {e}")
            raise
        
        return extracted_text.strip()
    
    def _extract_text_from_image(self, image_path):
        """Extract text from image file using OCR."""
        try:
            # Preprocess image for better OCR
            processed_image = self._preprocess_image(image_path)
            
            # Try Tesseract first
            try:
                text_tesseract = self._ocr_with_tesseract(processed_image)
                if text_tesseract.strip():
                    return text_tesseract
            except Exception as e:
                logger.warning(f"Tesseract OCR failed: {e}")
            
            # Fallback to EasyOCR if available
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
    
    def _preprocess_image(self, image_path):
        """Preprocess image for better OCR results."""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (1, 1), 0)
            
            # Apply threshold to get black and white image
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Apply morphological operations to clean up
            kernel = np.ones((1, 1), np.uint8)
            processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            return processed
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            # Return original image if preprocessing fails
            return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    def _ocr_with_tesseract(self, image):
        """Extract text using Tesseract OCR."""
        try:
            # Configure Tesseract
            config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,!?@#$%^&*()_+-=[]{}|;:,.<>?/'
            
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image)
            
            # Extract text
            text = pytesseract.image_to_string(pil_image, config=config)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Tesseract OCR error: {e}")
            raise
    
    def _ocr_with_easyocr(self, image_path):
        """Extract text using EasyOCR."""
        try:
            if not self.easyocr_available:
                raise RuntimeError("EasyOCR not available")
            
            # Extract text
            results = self.easyocr_reader.readtext(image_path)
            
            # Sort results by y-coordinate to maintain reading order
            results.sort(key=lambda x: x[0][0][1])  # Sort by top-left y coordinate
            
            # Combine all detected text with even lower confidence threshold
            text_list = []
            for (bbox, text, confidence) in results:
                # Very low confidence threshold to capture all possible text
                if confidence > 0.1:  
                    text_list.append(text)
                    print(f"EasyOCR detected: '{text}' (confidence: {confidence:.3f})")
            
            combined_text = ' '.join(text_list)
            logger.info(f"EasyOCR extracted: {len(text_list)} text segments: {combined_text}")
            
            return combined_text
            
        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
            raise
    
    def extract_text_from_bytes(self, file_bytes, file_extension):
        """
        Extract text from file bytes.
        
        Args:
            file_bytes (bytes): File content as bytes
            file_extension (str): File extension (.pdf, .png, .jpg, .jpeg)
            
        Returns:
            str: Extracted text
        """
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
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Could not decode image from bytes")
            
            # Preprocess image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Extract text with Tesseract
            pil_image = Image.fromarray(thresh)
            text = pytesseract.image_to_string(pil_image)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error processing image bytes: {e}")
            raise
