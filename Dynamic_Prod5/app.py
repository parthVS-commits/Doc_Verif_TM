from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import json
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

# Import our validation API
from api.document_validation_api import DocumentValidationAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_server.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("document-validation-api")

# Initialize the validation service
validation_api = DocumentValidationAPI()

# Create the FastAPI app
app = FastAPI(
    title="Document Validation API",
    description="API for validating director and company documents",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define input models
class DocumentInfo(BaseModel):
    """Document URL information"""
    url: str = Field(..., description="Document URL for validation")

class DirectorDocuments(BaseModel):
    """Director's documents"""
    aadharCardFront: Optional[str] = Field(None, description="Aadhar card front URL")
    aadharCardBack: Optional[str] = Field(None, description="Aadhar card back URL")
    panCard: Optional[str] = Field(None, description="PAN card URL")
    passportPhoto: Optional[str] = Field(None, description="Passport photo URL")
    address_proof: Optional[str] = Field(None, description="Address proof URL")
    signature: Optional[str] = Field(None, description="Signature URL")
    passport: Optional[str] = Field(None, description="Passport URL (for foreign directors)")
    drivingLicense: Optional[str] = Field(None, description="Driving license URL (for foreign directors)")

class Director(BaseModel):
    """Director information"""
    nationality: str = Field(..., description="Director's nationality (Indian/Foreign)")
    authorised: str = Field(..., description="Whether director is authorised (Yes/No)")
    documents: Dict[str, str] = Field(..., description="Director's documents URLs")

class CompanyDocuments(BaseModel):
    """Company documents"""
    address_proof_type: Optional[str] = Field(None, description="Type of address proof")
    addressProof: str = Field(..., description="Company address proof URL")
    noc: Optional[str] = Field(None, description="No Objection Certificate URL")

# TM-specific models
class CertificateRequirements(BaseModel):
    company_name_visible: bool
    certificate_is_valid_and_legible: bool

class ApplicantCompliance(BaseModel):
    msme_or_dipp_required: bool
    certificate_requirements: CertificateRequirements

class ApplicantDocuments(BaseModel):
    msme_certificate: str = ""
    dipp_certificate: str = ""

class Applicant(BaseModel):
    applicant_type: str
    applicant_name: str
    company_name: Optional[str] = None
    documents: ApplicantDocuments
    compliance: ApplicantCompliance

class VerificationDoc(BaseModel):
    url: str
    logo_url: Optional[str] = None  # URL to the logo file
    company_name_visible: Optional[bool] = True
    logo_visible: Optional[bool] = None  # Whether the logo is visible in this doc
    brand_name_visible: Optional[bool] = None  # Whether the brand name is visible in this doc

class Trademark(BaseModel):
    BrandName: str
    Logo: str  # "Yes" or "No"
    AlreadyInUse: str  # "Yes" or "No"
    VerificationDocs: Dict[str, VerificationDoc] = Field(default_factory=dict)

class TrademarkData(BaseModel):
    TrademarkNos: int
    Trademark1: Optional[Trademark] = None
    Trademark2: Optional[Trademark] = None
    Trademark3: Optional[Trademark] = None
    
    class Config:
        extra = "allow"  # Allow additional Trademark fields

# Combined ValidationRequest that supports both regular and TM validation
class ValidationRequest(BaseModel):
    service_id: str = "1"
    request_id: str
    preconditions: Optional[Dict[str, Any]] = None
    directors: Optional[Dict[str, Director]] = None
    companyDocuments: Optional[CompanyDocuments] = None
    applicant: Optional[Applicant] = None
    Trademarks: Optional[TrademarkData] = None

# Define error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc)
        }
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Document validation endpoint
@app.post("/validate", response_model=Dict[str, Any])
async def validate_documents(request: ValidationRequest):
    try:
        logger.info(f"Processing validation request: {request.request_id}")
        
        # Convert Pydantic model to dict
        input_data = request.dict()
        
        # For standard services, make sure required fields are present
        if request.service_id not in ["8"]:  # Not a TM service
            if not request.directors:
                raise HTTPException(
                    status_code=422,
                    detail="Directors information is required for non-TM services"
                )
            if not request.companyDocuments:
                raise HTTPException(
                    status_code=422,
                    detail="Company documents are required for non-TM services"
                )
        else:  # TM service
            if not request.Trademarks:
                raise HTTPException(
                    status_code=422,
                    detail="Trademarks information is required for TM service"
                )
            if not request.applicant:
                raise HTTPException(
                    status_code=422,
                    detail="Applicant information is required for TM service"
                )
        
        # Process validation
        api_response, _ = validation_api.validate_document(input_data)
        
        # Log completion
        logger.info(f"Validation completed for request: {request.request_id}")
        
        return api_response
    
    except HTTPException:
        # Re-raise FastAPI HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error processing validation request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Validation processing error: {str(e)}"
        )
    
# Add this after your existing route definitions
@app.get("/")
async def root():
    return {
        "message": "Document Validation API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "validate": "/validate"
        }
    }

# Run the app
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)