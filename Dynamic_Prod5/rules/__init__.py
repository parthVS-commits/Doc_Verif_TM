"""
Initialization file for the rules package
"""
from .address_validation_rules import validate_address_match
from .compliance_validation_rules import ComplianceValidationRules

# Expose specific validation methods
def validate_name_match(name1, name2):
    """Wrapper for name matching validation"""
    result = ComplianceValidationRules.validate_name_match(name1, name2)
    return result['status'] == 'passed'

def validate_dob_match(dob1, dob2, tolerance_days=7):
    """Wrapper for date of birth matching"""
    from datetime import datetime
    from dateutil import parser

    def parse_date(date_str):
        try:
            return parser.parse(date_str)
        except:
            return None

    parsed_dob1 = parse_date(dob1)
    parsed_dob2 = parse_date(dob2)

    if not (parsed_dob1 and parsed_dob2):
        return False

    date_diff = abs((parsed_dob1 - parsed_dob2).days)
    return date_diff <= tolerance_days

def validate_address_match(address1, address2):
    """Wrapper for address matching"""
    return ComplianceValidationRules.validate_name_match(address1, address2)

def validate_bill_age(bill_date, max_age_days=45):
    """Wrapper for bill age validation"""
    result = ComplianceValidationRules.validate_document_age(bill_date, max_age_days)
    return result['status'] == 'passed'