from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel

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
    description="API for validating director, company, and TM documents (base64 or URL accepted for all files)",
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

# Flexible input model for all services
class ValidationRequest(BaseModel):
    service_id: str = "1"
    request_id: str
    preconditions: Optional[Dict[str, Any]] = None
    directors: Optional[Dict[str, Any]] = None
    companyDocuments: Optional[Dict[str, Any]] = None
    applicant: Optional[Dict[str, Any]] = None
    Trademarks: Optional[Dict[str, Any]] = None

    class Config:
        extra = "allow"  # Allow any extra fields

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

@app.post("/validate", response_model=Dict[str, Any])
async def validate_documents(request: ValidationRequest):
    try:
        logger.info(f"Processing validation request: {request.request_id}")
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
            # Only keep relevant fields
            filtered_input = {
                "service_id": request.service_id,
                "request_id": request.request_id,
                "preconditions": request.preconditions,
                "directors": request.directors,
                "companyDocuments": request.companyDocuments
            }
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
            # Only keep relevant fields
            filtered_input = {
                "service_id": request.service_id,
                "request_id": request.request_id,
                "preconditions": request.preconditions,
                "applicant": request.applicant,
                "Trademarks": request.Trademarks
            }

        # Process validation
        api_response, _ = validation_api.validate_document(filtered_input)
        logger.info(f"Validation completed for request: {request.request_id}")
        return api_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing validation request: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Validation processing error: {str(e)}"
        )
# # Document validation endpoint
# @app.post("/validate", response_model=Dict[str, Any])
# async def validate_documents(request: ValidationRequest):
#     try:
#         logger.info(f"Processing validation request: {request.request_id}")
#         input_data = request.dict()

#         # For standard services, make sure required fields are present
#         if request.service_id not in ["8"]:  # Not a TM service
#             if not request.directors:
#                 raise HTTPException(
#                     status_code=422,
#                     detail="Directors information is required for non-TM services"
#                 )
#             if not request.companyDocuments:
#                 raise HTTPException(
#                     status_code=422,
#                     detail="Company documents are required for non-TM services"
#                 )
#         else:  # TM service
#             if not request.Trademarks:
#                 raise HTTPException(
#                     status_code=422,
#                     detail="Trademarks information is required for TM service"
#                 )
#             if not request.applicant:
#                 raise HTTPException(
#                     status_code=422,
#                     detail="Applicant information is required for TM service"
#                 )

#         # Process validation
#         api_response, _ = validation_api.validate_document(input_data)
#         logger.info(f"Validation completed for request: {request.request_id}")
#         return api_response

#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Error processing validation request: {str(e)}", exc_info=True)
#         raise HTTPException(
#             status_code=500,
#             detail=f"Validation processing error: {str(e)}"
#         )

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

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)