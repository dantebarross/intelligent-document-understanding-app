#!/usr/bin/env python
"""
Initialize vector database with sample documents for classification.
Uses a subset of training data for quick initialization.
"""

import os
import sys
import django
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'intelligent_document_api.settings')
django.setup()

from document_processor.services.vector_db_service import VectorDBService
from document_processor.services.ocr_service_optimized import OCRServiceOptimized
from training_data_mapping import get_target_category
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Initialize vector database with sample documents."""
    
    vector_service = VectorDBService()
    ocr_service = OCRServiceOptimized()
    
    training_data_path = Path("training_data")
    
    if not training_data_path.exists():
        logger.error(f"Training data directory {training_data_path} does not exist.")
        logger.info("Download dataset from: https://www.kaggle.com/datasets/shaz13/real-world-documents-collections")
        return
    
    logger.info("Starting vector database initialization with sample documents...")
    
    processed_count = 0
    category_counts = {}
    max_per_category = 5  # Sample size for quick initialization
    
    for category_dir in training_data_path.iterdir():
        if not category_dir.is_dir():
            continue
            
        source_category = category_dir.name
        target_category = get_target_category(source_category)
        
        if not target_category:
            continue
        
        if target_category not in category_counts:
            category_counts[target_category] = 0
        
        logger.info(f"Processing {source_category} -> {target_category}")
        
        # Get image files
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(category_dir.glob(ext))
        
        if not image_files:
            continue
        
        # Process only first few files for quick initialization
        files_to_process = image_files[:max_per_category]
        
        for i, file_path in enumerate(files_to_process):
            try:
                logger.info(f"Processing {i+1}/{len(files_to_process)}: {file_path.name}")
                
                extracted_text = ocr_service.extract_text(str(file_path))
                
                if not extracted_text.strip():
                    logger.warning(f"No text extracted from {file_path.name}")
                    continue
                
                vector_service.add_document(
                    text=extracted_text,
                    document_type=target_category,
                    file_path=str(file_path)
                )
                
                processed_count += 1
                category_counts[target_category] += 1
                
                logger.info(f"Successfully processed {file_path.name}")
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
    
    try:
        vector_service.save_index()
        logger.info("Vector database saved successfully!")
    except Exception as e:
        logger.error(f"Error saving vector database: {e}")
        return
    
    logger.info(f"Vector database initialization complete! Total documents processed: {processed_count}")
    
    for doc_type, count in sorted(category_counts.items()):
        logger.info(f"  {doc_type}: {count} documents")
    
    if processed_count == 0:
        logger.warning("No documents were processed. Ensure training_data/ contains the Kaggle dataset.")


if __name__ == "__main__":
    main()
