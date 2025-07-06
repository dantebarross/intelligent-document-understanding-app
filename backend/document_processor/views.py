"""
Views for document processing API.
"""

import os
import time
import logging
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status

from .services.ocr_service_optimized import OCRServiceOptimized
from .services.vector_db_service import VectorDBService
from .services.llm_service import LLMService

logger = logging.getLogger(__name__)


@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def extract_entities(request):
    """
    Extract entities from uploaded document.
    
    Accepts: PDF, PNG, JPG, JPEG files
    Returns: JSON with document type, confidence, and extracted entities
    """
    start_time = time.time()
    
    try:
        # Validate request
        if 'file' not in request.FILES:
            return Response({
                'success': False,
                'error': 'No file provided',
                'code': 'NO_FILE'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        uploaded_file = request.FILES['file']
        
        # Validate file
        validation_error = _validate_file(uploaded_file)
        if validation_error:
            return Response({
                'success': False,
                'error': validation_error['error'],
                'code': validation_error['code']
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Save file temporarily
        file_path = _save_temporary_file(uploaded_file)
        
        try:
            # Initialize services
            ocr_service = OCRServiceOptimized()
            vector_service = VectorDBService()
            llm_service = LLMService()
            
            # Step 1: Extract text using OCR
            logger.info("Starting OCR extraction...")
            extracted_text = ocr_service.extract_text(file_path)
            
            if not extracted_text.strip():
                return Response({
                    'success': False,
                    'error': 'No text could be extracted from the document',
                    'code': 'OCR_FAILED'
                }, status=status.HTTP_422_UNPROCESSABLE_ENTITY)
            
            # Step 2: Classify document type using vector search
            logger.info("Classifying document type...")
            document_type, classification_confidence = vector_service.classify_document(extracted_text)
            
            # Step 3: Extract entities using LLM
            logger.info(f"Extracting entities for document type: {document_type}")
            entities = {}
            
            if llm_service.is_available():
                entities = llm_service.extract_entities(extracted_text, document_type)
                if not entities:
                    logger.warning("LLM entity extraction failed")
                    entities = {}
            else:
                logger.warning("LLM service not available")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Prepare response
            response_data = {
                'success': True,
                'document_type': document_type,
                'confidence': round(classification_confidence, 3),
                'entities': entities,
                'extracted_text': extracted_text,
                'processing_time': f"{processing_time:.2f}s"
            }
            
            logger.info(f"Document processed successfully in {processing_time:.2f}s")
            return Response(response_data, status=status.HTTP_200_OK)
            
        finally:
            # Clean up temporary file
            _cleanup_file(file_path)
            
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        return Response({
            'success': False,
            'error': 'Internal server error during document processing',
            'code': 'PROCESSING_ERROR'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def health_check(request):
    """Health check endpoint."""
    try:
        # Check services
        services_status = {}
        
        # Check OCR service
        try:
            ocr_service = OCRServiceOptimized()
            services_status['ocr'] = 'available'
        except Exception as e:
            services_status['ocr'] = f'error: {str(e)}'
        
        # Check Vector DB service
        try:
            vector_service = VectorDBService()
            stats = vector_service.get_statistics()
            services_status['vector_db'] = {
                'status': 'available',
                'documents': stats.get('total_documents', 0)
            }
        except Exception as e:
            services_status['vector_db'] = f'error: {str(e)}'
        
        # Check LLM service
        try:
            llm_service = LLMService()
            services_status['llm'] = 'available' if llm_service.is_available() else 'not configured'
        except Exception as e:
            services_status['llm'] = f'error: {str(e)}'
        
        return Response({
            'status': 'healthy',
            'services': services_status,
            'supported_formats': ['PDF', 'PNG', 'JPG', 'JPEG'],
            'max_file_size': '10MB'
        })
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return Response({
            'status': 'unhealthy',
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def get_supported_document_types(request):
    """Get list of supported document types."""
    try:
        llm_service = LLMService()
        document_types = llm_service.get_supported_document_types()
        
        # Add field information for each type
        types_info = {}
        for doc_type in document_types:
            types_info[doc_type] = {
                'fields': llm_service.get_fields_for_document_type(doc_type)
            }
        
        return Response({
            'supported_types': document_types,
            'types_info': types_info
        })
        
    except Exception as e:
        logger.error(f"Error getting document types: {e}")
        return Response({
            'error': 'Could not retrieve document types'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


def _validate_file(uploaded_file):
    """Validate uploaded file."""
    # Check file size (10MB limit)
    max_size = 10 * 1024 * 1024  # 10MB
    if uploaded_file.size > max_size:
        return {
            'error': 'File size exceeds maximum limit of 10MB',
            'code': 'FILE_TOO_LARGE'
        }
    
    # Check file format
    allowed_extensions = ['.pdf', '.png', '.jpg', '.jpeg']
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    
    if file_extension not in allowed_extensions:
        return {
            'error': 'Invalid file format. Only PDF, PNG, JPG, and JPEG files are supported',
            'code': 'INVALID_FORMAT'
        }
    
    return None


def _save_temporary_file(uploaded_file):
    """Save uploaded file temporarily."""
    try:
        # Create temporary filename
        file_extension = os.path.splitext(uploaded_file.name)[1]
        temp_filename = f"temp_{int(time.time())}_{uploaded_file.name}"
        
        # Save file
        file_path = default_storage.save(temp_filename, ContentFile(uploaded_file.read()))
        
        # Return full path
        return default_storage.path(file_path)
        
    except Exception as e:
        logger.error(f"Error saving temporary file: {e}")
        raise


def _cleanup_file(file_path):
    """Clean up temporary file."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.warning(f"Could not clean up file {file_path}: {e}")
