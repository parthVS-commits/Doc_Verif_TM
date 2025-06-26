from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import re

class ComplianceValidationRules:
    """
    Comprehensive compliance validation utility methods
    """
    
    @staticmethod
    def validate_document_age(
        document_date: Optional[str], 
        max_age_days: int = 45
    ) -> Dict[str, Any]:
        """
        Validate document age
        
        Args:
            document_date (str): Date of the document
            max_age_days (int): Maximum allowed document age
        
        Returns:
            dict: Validation result
        """
        try:
            # Parse the date
            parsed_date = ComplianceValidationRules._parse_date(document_date)
            
            if not parsed_date:
                return {
                    "status": "failed",
                    "error_message": "Could not parse document date"
                }
            
            # Calculate document age
            today = datetime.now()
            document_age = (today - parsed_date).days
            
            if document_age > max_age_days:
                return {
                    "status": "failed",
                    "error_message": f"Document is {document_age} days old, exceeding {max_age_days} days limit"
                }
            
            return {
                "status": "passed",
                "document_age": document_age
            }
        
        except Exception as e:
            return {
                "status": "failed",
                "error_message": f"Error validating document age: {str(e)}"
            }
    
    @staticmethod
    def validate_name_match(
        name1: str, 
        name2: str, 
        strict: bool = False
    ) -> Dict[str, Any]:
        """
        Validate name matching with fuzzy logic
        
        Args:
            name1 (str): First name
            name2 (str): Second name
            strict (bool): Whether to use strict matching
        
        Returns:
            dict: Matching result
        """
        try:
            # Normalize names
            def normalize_name(name):
                return re.sub(r'[^a-z\s]', '', name.lower().strip())
            
            norm_name1 = normalize_name(name1)
            norm_name2 = normalize_name(name2)
            
            if strict:
                # Exact match
                is_match = norm_name1 == norm_name2
            else:
                # Fuzzy matching
                is_match = (
                    norm_name1 in norm_name2 or 
                    norm_name2 in norm_name1 or
                    ComplianceValidationRules._calculate_name_similarity(norm_name1, norm_name2)
                )
            
            return {
                "status": "passed" if is_match else "failed",
                "error_message": "Names do not match" if not is_match else None
            }
        
        except Exception as e:
            return {
                "status": "failed",
                "error_message": f"Error in name matching: {str(e)}"
            }
    
    @staticmethod
    def validate_age(
        dob: str, 
        min_age: int = 18, 
        max_age: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Validate age based on date of birth
        
        Args:
            dob (str): Date of birth
            min_age (int): Minimum required age
            max_age (int, optional): Maximum allowed age
        
        Returns:
            dict: Age validation result
        """
        try:
            # Parse date of birth
            parsed_dob = ComplianceValidationRules._parse_date(dob)
            
            if not parsed_dob:
                return {
                    "status": "failed",
                    "error_message": "Could not parse date of birth"
                }
            
            # Calculate age
            today = datetime.now()
            age = today.year - parsed_dob.year - (
                (today.month, today.day) < (parsed_dob.month, parsed_dob.day)
            )
            
            # Validate age
            if age < min_age:
                return {
                    "status": "failed",
                    "error_message": f"Age {age} is below minimum required age {min_age}"
                }
            
            if max_age and age > max_age:
                return {
                    "status": "failed",
                    "error_message": f"Age {age} exceeds maximum allowed age {max_age}"
                }
            
            return {
                "status": "passed",
                "age": age
            }
        
        except Exception as e:
            return {
                "status": "failed",
                "error_message": f"Error validating age: {str(e)}"
            }
    
    @staticmethod
    def validate_document_completeness(
        document_data: Dict[str, Any], 
        required_fields: list
    ) -> Dict[str, Any]:
        """
        Validate document completeness
        
        Args:
            document_data (dict): Extracted document data
            required_fields (list): List of required fields
        
        Returns:
            dict: Completeness validation result
        """
        try:
            missing_fields = [
                field for field in required_fields 
                if not document_data.get(field)
            ]
            
            if missing_fields:
                return {
                    "status": "failed",
                    "error_message": f"Missing fields: {', '.join(missing_fields)}"
                }
            
            return {
                "status": "passed"
            }
        
        except Exception as e:
            return {
                "status": "failed",
                "error_message": f"Error validating document completeness: {str(e)}"
            }
    
    @staticmethod
    def _parse_date(date_str: str) -> Optional[datetime]:
        """
        Parse date string in multiple formats
        
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
            from dateutil import parser
            return parser.parse(date_str)
        except (ValueError, TypeError):
            return None
    
    @staticmethod
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