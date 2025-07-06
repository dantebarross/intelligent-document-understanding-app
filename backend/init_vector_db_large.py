"""
Initialize vector database with large training dataset.
Processes documents from training_data/ using category mapping.
"""

import os
import sys
import logging
from pathlib import Path
from training_data_mapping import get_target_category, get_all_source_categories

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'intelligent_document_api.settings')
import django
django.setup()

from document_processor.services.vector_db_service import VectorDBService
from document_processor.services.ocr_service_optimized import OCRServiceOptimized

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def count_documents_by_category(training_data_path):
    """Count documents in each mapped category."""
    counts = {}
    total_files = 0
    mapped_files = 0
    
    logger.info("Scanning training data categories...")
    
    for category_dir in Path(training_data_path).iterdir():
        if not category_dir.is_dir():
            continue
            
        category_name = category_dir.name
        target_category = get_target_category(category_name)
        
        files = list(category_dir.glob('*.jpg')) + list(category_dir.glob('*.jpeg')) + list(category_dir.glob('*.png'))
        file_count = len(files)
        total_files += file_count
        
        if target_category:
            mapped_files += file_count
            if target_category not in counts:
                counts[target_category] = 0
            counts[target_category] += file_count
            logger.info(f"{category_name} -> {target_category}: {file_count} files")
        else:
            logger.info(f"{category_name} -> IGNORED: {file_count} files")
    
    logger.info(f"Total files: {total_files}, Mapped: {mapped_files}, Ignored: {total_files - mapped_files}")
    for category, count in sorted(counts.items()):
        logger.info(f"  {category}: {count} documents")
    
    return counts

def process_training_data(vector_service, ocr_service, training_data_path, max_per_category=None):
    """Process documents from training_data directory."""
    processed_count = 0
    category_counts = {}
    
    logger.info("Starting large dataset processing...")
    
    for category_dir in Path(training_data_path).iterdir():
        if not category_dir.is_dir():
            continue
            
        source_category = category_dir.name
        target_category = get_target_category(source_category)
        
        if not target_category:
            logger.info(f"Skipping ignored category: {source_category}")
            continue
        
        if target_category not in category_counts:
            category_counts[target_category] = 0
        
        logger.info(f"Processing {source_category} -> {target_category}")
        
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(category_dir.glob(ext))
        
        logger.info(f"Found {len(image_files)} files in {source_category}")
        
        files_to_process = image_files
        if max_per_category:
            files_to_process = image_files[:max_per_category]
            if len(image_files) > max_per_category:
                logger.info(f"Limiting to {max_per_category} files per category")
        
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
                
                if processed_count % 10 == 0:
                    logger.info(f"Progress: {processed_count} documents processed")
                    
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                continue
    
    return processed_count, category_counts

def main():
    """Main function to initialize vector database with large dataset."""
    
    training_data_path = "training_data"
    max_per_category = 20
    
    if not os.path.exists(training_data_path):
        logger.error(f"Training data directory not found: {training_data_path}")
        return
    
    logger.info("PHASE 1: Scanning dataset")
    category_counts = count_documents_by_category(training_data_path)
    
    if not category_counts:
        logger.error("No mapped documents found!")
        return
    
    total_docs = sum(category_counts.values())
    logger.info(f"Ready to process {total_docs} documents.")
    
    if total_docs > 100:
        logger.info("This will take significant time! Consider setting max_per_category for testing first.")
    
    logger.info("PHASE 2: Initializing services")
    vector_service = VectorDBService()
    ocr_service = OCRServiceOptimized()
    
    logger.info("Clearing existing vector database...")
    vector_service.clear_index()
    
    logger.info("PHASE 3: Processing documents")
    processed_count, final_counts = process_training_data(
        vector_service, 
        ocr_service, 
        training_data_path,
        max_per_category
    )
    
    logger.info("PHASE 4: Saving vector database")
    try:
        vector_service.save_index()
        logger.info("Vector database saved successfully!")
    except Exception as e:
        logger.error(f"Error saving vector database: {e}")
        return
    
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Total documents processed: {processed_count}")
    
    for category, count in sorted(final_counts.items()):
        logger.info(f"  {category}: {count} documents")
    
    logger.info("Vector database ready for use!")

if __name__ == "__main__":
    main()
