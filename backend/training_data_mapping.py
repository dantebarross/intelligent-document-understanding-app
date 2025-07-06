"""
Category mapping from training_data to system categories.
"""

CATEGORY_MAPPING = {
    # Invoices and commercial documents
    'invoice': ['invoice'],
    
    # Receipts and forms
    'receipt': ['form', 'questionnaire'],
    
    # Contracts and legal documents
    'contract': ['letter', 'memo'],
    
    # Financial statements and documents
    'bank_statement': ['budget'],
    
    # Identification documents
    'id_document': ['resume'],
    
    # Medical/scientific documents
    'medical_record': ['scientific_publication', 'scientific_report'],
    
    # Unmapped categories (ignore for now)
    'ignored': [
        'advertisement', 
        'email', 
        'file_folder', 
        'handwritten', 
        'news_article', 
        'presentation', 
        'specification'
    ]
}

def get_target_category(source_category):
    """
    Return the target category for a source category.
    
    Args:
        source_category (str): Category from training_data
        
    Returns:
        str or None: Target category or None if should be ignored
    """
    for target, sources in CATEGORY_MAPPING.items():
        if source_category in sources:
            if target == 'ignored':
                return None
            return target
    return None

def get_all_source_categories():
    """Return all mapped source categories."""
    mapped = []
    for sources in CATEGORY_MAPPING.values():
        mapped.extend(sources)
    return mapped
