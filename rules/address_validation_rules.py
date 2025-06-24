import re
from typing import Dict, Any

def normalize_address(address: str) -> str:
    """
    Normalize an address for comparison
    
    Args:
        address (str): Address to normalize
    
    Returns:
        str: Normalized address
    """
    if not address:
        return ''
    
    # Convert to lowercase
    normalized = address.lower().strip()
    
    # Remove special characters
    normalized = re.sub(r'[^a-z0-9\s]', '', normalized)
    
    # Remove extra whitespaces
    normalized = ' '.join(normalized.split())
    
    return normalized

def validate_address_match(
    documents: Dict[str, Any], 
    directors: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Validate address match between electricity bill and director addresses
    
    Args:
        documents (dict): All documents to check
        directors (dict, optional): Director information
    
    Returns:
        dict: Address validation results
    """
    # Collect addresses from different sources
    addresses = []
    
    # Extract electricity bill address
    electricity_bill_address = None
    for doc_key, doc_info in documents.items():
        if isinstance(doc_info, dict) and doc_info.get('is_valid'):
            extracted_data = doc_info.get('extracted_data', {})
            bill_address = extracted_data.get('address')
            if bill_address:
                electricity_bill_address = bill_address
                break
    
    # If no directors provided, use documents for address extraction
    if directors is None:
        directors = {}
    
    # Collect director addresses
    director_addresses = []
    for director_key, director_info in directors.items():
        # Check if director is authorised
        if director_info.get('is_authorised', False):
            # Look through director's documents
            for doc_key, doc_info in director_info.get('documents', {}).items():
                if isinstance(doc_info, dict) and doc_info.get('is_valid'):
                    extracted_data = doc_info.get('extracted_data', {})
                    dir_address = extracted_data.get('address')
                    if dir_address:
                        director_addresses.append(dir_address)
    
    # If no electricity bill address found
    if not electricity_bill_address:
        return {
            'is_consistent': False,
            'error': 'No electricity bill address found'
        }
    
    # If no director addresses found
    if not director_addresses:
        return {
            'is_consistent': False,
            'error': 'No director addresses found'
        }
    
    # Check address match
    is_consistent = any(
        _addresses_match(electricity_bill_address, dir_address)
        for dir_address in director_addresses
    )
    
    return {
        'is_consistent': is_consistent,
        'electricity_bill_address': electricity_bill_address,
        'director_addresses': director_addresses,
        'error': 'Addresses do not match' if not is_consistent else None
    }

def _addresses_match(address1: str, address2: str) -> bool:
    """
    Advanced address matching
    
    Args:
        address1 (str): First address
        address2 (str): Second address
    
    Returns:
        bool: Whether addresses match
    """
    if not address1 or not address2:
        return False
    
    # Normalize addresses
    def normalize_address(address: str) -> str:
        # Convert to lowercase
        norm_addr = address.lower()
        
        # Remove special characters and extra whitespaces
        norm_addr = re.sub(r'[^\w\s]', '', norm_addr)
        norm_addr = re.sub(r'\s+', ' ', norm_addr).strip()
        
        return norm_addr
    
    norm_addr1 = normalize_address(address1)
    norm_addr2 = normalize_address(address2)
    
    # Exact substring match
    if norm_addr1 in norm_addr2 or norm_addr2 in norm_addr1:
        return True
    
    # Check for common location identifiers
    common_keywords = ['flat', 'apartment', 'street', 'road', 'lane', 'block']
    
    # Check if any common keywords match
    for keyword in common_keywords:
        if keyword in norm_addr1 and keyword in norm_addr2:
            return True
    
    # Extract and compare postal codes if present
    postal_code1 = re.search(r'\b\d{6}\b', norm_addr1)
    postal_code2 = re.search(r'\b\d{6}\b', norm_addr2)
    
    if postal_code1 and postal_code2 and postal_code1.group() == postal_code2.group():
        return True
    
    return False

def _calculate_address_similarity(
    address1: str, 
    address2: str, 
    threshold: float = 0.7
) -> bool:
    """
    Calculate address similarity using advanced techniques
    
    Args:
        address1 (str): First normalized address
        address2 (str): Second normalized address
        threshold (float): Similarity threshold
    
    Returns:
        bool: Whether addresses are similar
    """
    # Extract key address components
    def extract_components(address):
        # Extract postal code
        postal_code = re.search(r'\b\d{6}\b', address)
        
        # Extract street/area name
        street = re.findall(r'\b[a-z]+\s*(?:street|road|avenue|lane|society)\b', address)
        
        return {
            'postal_code': postal_code.group(0) if postal_code else None,
            'street': street[0] if street else None
        }
    
    # Compare address components
    components1 = extract_components(address1)
    components2 = extract_components(address2)
    
    # Check postal code match
    postal_match = (
        components1['postal_code'] and 
        components1['postal_code'] == components2['postal_code']
    )
    
    # Check street name similarity
    street_match = (
        components1['street'] and 
        components1['street'] == components2['street']
    )
    
    return postal_match or street_match

def validate_address_consistency(
    documents: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate address consistency across documents
    
    Args:
        documents (dict): Documents to check for address consistency
    
    Returns:
        dict: Address validation results
    """
    # Collect addresses from different documents
    addresses = []
    
    for doc_key, doc_info in documents.items():
        # Check if document is valid and has extracted data
        if doc_info.get('is_valid') and doc_info.get('extracted_data'):
            extracted_address = doc_info['extracted_data'].get('address')
            if extracted_address:
                addresses.append(extracted_address)
    
    # Check address consistency
    if not addresses:
        return {
            'is_consistent': False,
            'error': 'No valid addresses found in documents'
        }
    
    # Compare first address with others
    first_address = addresses[0]
    is_consistent = all(
        validate_address_match(first_address, address) 
        for address in addresses[1:]
    )
    
    return {
        'is_consistent': is_consistent,
        'addresses': addresses,
        'error': 'Addresses are inconsistent' if not is_consistent else None
    }