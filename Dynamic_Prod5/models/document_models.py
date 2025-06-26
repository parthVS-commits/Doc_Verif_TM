from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum, auto

class DocumentType(Enum):
    """Enum for document types"""
    AADHAR_FRONT = auto()
    AADHAR_BACK = auto()
    PAN_CARD = auto()
    PASSPORT_PHOTO = auto()
    ADDRESS_PROOF = auto()
    ELECTRICITY_BILL = auto()
    SIGNATURE = auto()  # Added for signature document
    NOC = auto()  # Added for No Objection Certificate
    DRIVING_LICENSE = auto()  # Added for foreign nationals
    PASSPORT = auto()  # Added for foreign nationals

class NationalityType(Enum):
    """Enum for nationality types"""
    INDIAN = "Indian"
    FOREIGN = "Foreign"

@dataclass
class DocumentInfo:
    """
    Represents a single document's information
    """
    url: str
    document_type: DocumentType
    is_valid: bool = False
    extraction_data: Dict = field(default_factory=dict)
    clarity_score: Optional[float] = None  # Added for document clarity check
    is_recent: Optional[bool] = None  # Added for recency check
    is_masked: Optional[bool] = None  # Added for masked Aadhar check

@dataclass
class DirectorDocuments:
    """
    Represents documents for a single director
    """
    name: str
    nationality: NationalityType
    is_authorised: bool
    age: Optional[int] = None  # Added for age verification
    documents: Dict[DocumentType, DocumentInfo] = field(default_factory=dict)

@dataclass
class CompanyDocuments:
    """
    Represents company-level documents
    """
    address_proof_type: str
    address_proof: DocumentInfo
    noc: Optional[DocumentInfo] = None  # Added for NOC

@dataclass
class ValidationResult:
    """
    Comprehensive validation result
    """
    validation_rules: Dict[str, Dict[str, str]] = field(default_factory=dict)
    document_validation: Dict = field(default_factory=dict)
    is_compliant: bool = False
    processing_time: float = 0.0
    error_messages: List[str] = field(default_factory=list)

class ValidationRuleStatus:
    """
    Standardized statuses for validation rules
    """
    PASSED = "passed"
    FAILED = "failed"

class DocumentValidationError(Exception):
    """
    Custom exception for document validation errors
    """
    def __init__(self, message: str, error_code: Optional[str] = None):
        """
        Initialize validation error
        
        Args:
            message (str): Error description
            error_code (str, optional): Specific error code
        """
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

def validate_url(url: str) -> bool:
    """
    Basic URL validation
    
    Args:
        url (str): URL to validate
    
    Returns:
        bool: Whether URL is valid
    """
    try:
        from urllib.parse import urlparse
        result = urlparse(url)
        return all([
            result.scheme in ['http', 'https'],
            result.netloc
        ])
    except Exception:
        return False
    
# Add these classes to document_models.py

class ApplicantType(Enum):
    """Enum for applicant types"""
    INDIVIDUAL = "Individual"
    COMPANY = "Company"

@dataclass
class CertificateInfo:
    """Certificate validation information"""
    company_name_visible: bool = False
    certificate_is_valid_and_legible: bool = False

@dataclass
class ApplicantCompliance:
    """Applicant compliance requirements"""
    msme_or_dipp_required: bool = False
    certificate_requirements: CertificateInfo = field(default_factory=CertificateInfo)

@dataclass
class ApplicantDocuments:
    """Applicant document information"""
    msme_certificate: Optional[str] = None
    dipp_certificate: Optional[str] = None

@dataclass
class ApplicantInfo:
    """Complete applicant information"""
    applicant_type: ApplicantType
    applicant_name: str
    company_name: Optional[str] = None
    documents: ApplicantDocuments = field(default_factory=ApplicantDocuments)
    compliance: ApplicantCompliance = field(default_factory=ApplicantCompliance)

@dataclass
class VerificationDocument:
    """Trademark verification document"""
    url: str
    company_name_visible: bool = False
    logo_visible: bool = False
    brand_name_visible: bool = False
    brand_name_in_logo: bool = False
    extracted_text: str = ""
    clarity_score: float = 0.0

@dataclass
class TrademarkInfo:
    """Trademark information"""
    BrandName: str
    Logo: str  # "Yes" or "No"
    #LogoFile: Optional[str] = None  # <-- Add this
    AlreadyInUse: str  # "Yes" or "No"
    VerificationDocs: Dict[str, VerificationDocument] = field(default_factory=dict)

@dataclass
class TrademarkData:
    """Complete trademark data"""
    TrademarkNos: int
    trademarks: Dict[str, TrademarkInfo] = field(default_factory=dict)

@dataclass
class TrademarkValidationResult:
    """Trademark validation result"""
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    trademark_validations: Dict[str, Dict] = field(default_factory=dict)
    applicant_validation: Dict = field(default_factory=dict)
# from dataclasses import dataclass, field
# from typing import Dict, List, Optional
# from enum import Enum, auto

# class DocumentType(Enum):
#     """Enum for document types"""
#     AADHAR_FRONT = auto()
#     AADHAR_BACK = auto()
#     PAN_CARD = auto()
#     PASSPORT_PHOTO = auto()
#     ADDRESS_PROOF = auto()
#     ELECTRICITY_BILL = auto()
#     SIGNATURE = auto()
#     NOC = auto()
#     DRIVING_LICENSE = auto()
#     PASSPORT = auto()
#     MSME_CERTIFICATE = auto()
#     DIPP_CERTIFICATE = auto()
#     TRADEMARK_LOGO = auto()
#     TRADEMARK_VERIFICATION = auto()

# class NationalityType(Enum):
#     """Enum for nationality types"""
#     INDIAN = "Indian"
#     FOREIGN = "Foreign"

# @dataclass
# class DocumentInfo:
#     """
#     Represents a single document's information
#     """
#     url: str
#     document_type: DocumentType
#     is_valid: bool = False
#     extraction_data: Dict = field(default_factory=dict)
#     clarity_score: Optional[float] = None
#     is_recent: Optional[bool] = None
#     is_masked: Optional[bool] = None
#     company_name_visible: Optional[bool] = None

# @dataclass
# class DirectorDocuments:
#     """
#     Represents documents for a single director
#     """
#     name: str
#     nationality: NationalityType
#     is_authorised: bool
#     age: Optional[int] = None
#     documents: Dict[DocumentType, DocumentInfo] = field(default_factory=dict)

# @dataclass
# class CompanyDocuments:
#     """
#     Represents company-level documents
#     """
#     address_proof_type: str
#     address_proof: DocumentInfo
#     noc: Optional[DocumentInfo] = None

# @dataclass
# class ApplicantDocuments:
#     msme_certificate: Optional[DocumentInfo] = None
#     dipp_certificate: Optional[DocumentInfo] = None

# @dataclass
# class Applicant:
#     applicant_type: str  # "Individual" or "Company"
#     applicant_name: str
#     company_name: str
#     documents: ApplicantDocuments

# @dataclass
# class ComplianceCertificateRequirements:
#     company_name_visible: bool = True
#     certificate_is_valid_and_legible: bool = True

# @dataclass
# class Compliance:
#     msme_or_dipp_required: bool
#     certificate_requirements: ComplianceCertificateRequirements

# @dataclass
# class VerificationDoc:
#     url: str
#     company_name_visible: bool

# @dataclass
# class Trademark:
#     brand_name: str
#     logo: str  # "Yes" or "No"
#     logo_document: Optional[DocumentInfo] = None
#     already_in_use: str  # "Yes" or "No"
#     verification_docs: List[DocumentInfo] = field(default_factory=list)

# @dataclass
# class TrademarkInput:
#     trademark_nos: int
#     trademarks: List[Trademark] = field(default_factory=list)

# @dataclass
# class ValidationResult:
#     validation_rules: Dict[str, Dict[str, str]] = field(default_factory=dict)
#     document_validation: Dict = field(default_factory=dict)
#     is_compliant: bool = False
#     processing_time: float = 0.0
#     error_messages: List[str] = field(default_factory=list)

# class ValidationRuleStatus:
#     PASSED = "passed"
#     FAILED = "failed"

# class DocumentValidationError(Exception):
#     def __init__(self, message: str, error_code: Optional[str] = None):
#         self.message = message
#         self.error_code = error_code
#         super().__init__(self.message)

# def validate_url(url: str) -> bool:
#     try:
#         from urllib.parse import urlparse
#         result = urlparse(url)
#         return all([
#             result.scheme in ['http', 'https'],
#             result.netloc
#         ])
#     except Exception:
#         return False
