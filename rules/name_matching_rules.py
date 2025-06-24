import re
from typing import Dict, Any

def normalize_name(name: str) -> str:
    """
    Normalize a name for comparison
    
    Args:
        name (str): Name to normalize
    
    Returns:
        str: Normalized name
    """
    if not name:
        return ''
    
    # Convert to lowercase
    normalized = name.lower().strip()
    
    # Remove special characters
    normalized = re.sub(r'[^a-z\s]', '', normalized)
    
    # Remove extra whitespaces
    normalized = ' '.join(normalized.split())
    
    return normalized

def check_name_match(
    name1: str, 
    name2: str, 
    strict: bool = False
) -> bool:
    """
    Compare two names for similarity
    
    Args:
        name1 (str): First name
        name2 (str): Second name
        strict (bool): Whether to use strict matching
    
    Returns:
        bool: Whether names match
    """
    # Normalize names
    norm_name1 = normalize_name(name1)
    norm_name2 = normalize_name(name2)
    
    if strict:
        # Exact match
        return norm_name1 == norm_name2
    
    # Fuzzy matching
    # Check if one name is a substring of another
    return (
        norm_name1 in norm_name2 or 
        norm_name2 in norm_name1 or
        # Levenshtein distance could be added here for more advanced matching
        _calculate_name_similarity(norm_name1, norm_name2)
    )

def _calculate_name_similarity(name1: str, name2: str, threshold: float = 0.8) -> bool:
    """
    Calculate name similarity using advanced techniques
    
    Args:
        name1 (str): First normalized name
        name2 (str): Second normalized name
        threshold (float): Similarity threshold
    
    Returns:
        bool: Whether names are similar
    """
    # Split names into words
    words1 = name1.split()
    words2 = name2.split()
    
    # Check common word ratio
    common_words = set(words1) & set(words2)
    max_words = max(len(words1), len(words2))
    
    # Calculate similarity ratio
    similarity_ratio = len(common_words) / max_words
    
    return similarity_ratio >= threshold

def validate_name_consistency(
    documents: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate name consistency across documents
    
    Args:
        documents (dict): Documents to check for name consistency
    
    Returns:
        dict: Name validation results
    """
    # Collect names from different documents
    names = []
    
    for doc_key, doc_info in documents.items():
        # Check if document is valid and has extracted data
        if doc_info.get('is_valid') and doc_info.get('extracted_data'):
            extracted_name = doc_info['extracted_data'].get('name')
            if extracted_name:
                names.append(extracted_name)
    
    # Check name consistency
    if not names:
        return {
            'is_consistent': False,
            'error': 'No valid names found in documents'
        }
    
    # Compare first name with others
    first_name = names[0]
    is_consistent = all(
        check_name_match(first_name, name) 
        for name in names[1:]
    )
    
    return {
        'is_consistent': is_consistent,
        'names': names,
        'error': 'Names are inconsistent' if not is_consistent else None
    }