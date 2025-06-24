from datetime import datetime
from typing import Dict, Any, Union
from dateutil import parser

def parse_date(date_str: str) -> Union[datetime, None]:
    """
    Parse a date string in multiple formats
    
    Args:
        date_str (str): Date string to parse
    
    Returns:
        datetime or None: Parsed date
    """
    if not date_str:
        return None
    
    # Date formats to try
    date_formats = [
        '%d/%m/%Y',  # DD/MM/YYYY
        '%Y-%m-%d',  # YYYY-MM-DD
        '%d-%m-%Y',  # DD-MM-YYYY
        '%m/%d/%Y',  # MM/DD/YYYY
        '%d %B %Y',  # DD Month YYYY
        '%d %b %Y'   # DD Mon YYYY
    ]
    
    # Try parsing with specific formats
    for fmt in date_formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    # Fallback to dateutil parser
    try:
        return parser.parse(date_str)
    except (ValueError, TypeError):
        return None

def validate_date_match(
    date1: str, 
    date2: str, 
    tolerance_days: int = 7
) -> bool:
    """
    Compare two dates for similarity
    
    Args:
        date1 (str): First date
        date2 (str): Second date
        tolerance_days (int): Allowed difference in days
    
    Returns:
        bool: Whether dates match
    """
    # Parse dates
    parsed_date1 = parse_date(date1)
    parsed_date2 = parse_date(date2)
    
    # Check if both dates are valid
    if not (parsed_date1 and parsed_date2):
        return False
    
    # Calculate date difference
    date_diff = abs((parsed_date1 - parsed_date2).days)
    
    return date_diff <= tolerance_days

def validate_dob_consistency(
    documents: Dict[str, Any],
    tolerance_days: int = 7
) -> Dict[str, Any]:
    """
    Validate Date of Birth consistency across documents
    
    Args:
        documents (dict): Documents to check for DOB consistency
        tolerance_days (int): Allowed difference in days
    
    Returns:
        dict: DOB validation results
    """
    # Collect dates of birth from different documents
    dobs = []
    
    for doc_key, doc_info in documents.items():
        # Check if document is valid and has extracted data
        if doc_info.get('is_valid') and doc_info.get('extracted_data'):
            extracted_dob = doc_info['extracted_data'].get('dob')
            if extracted_dob:
                dobs.append(extracted_dob)
    
    # Check DOB consistency
    if not dobs:
        return {
            'is_consistent': False,
            'error': 'No valid dates of birth found in documents'
        }
    
    # Compare first DOB with others
    first_dob = dobs[0]
    is_consistent = all(
        validate_date_match(first_dob, dob, tolerance_days) 
        for dob in dobs[1:]
    )
    
    return {
        'is_consistent': is_consistent,
        'dates_of_birth': dobs,
        'error': 'Dates of Birth are inconsistent' if not is_consistent else None
    }