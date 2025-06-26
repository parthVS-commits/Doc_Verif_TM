import logging
from typing import Dict, Any, Optional

from utils.file_utils import DocumentDownloader, APIDocumentFetcher
from services.extraction_service import ExtractionService
from models.document_models import DocumentType, NationalityType
from config.settings import Config

class DocumentProcessor:
    """
    Comprehensive document processing service
    
    Handles document retrieval, validation, and data extraction
    """
    
    def __init__(
        self, 
        extraction_service: Optional[ExtractionService] = None,
        api_key: Optional[str] = None,
        api_token: Optional[str] = None
    ):
        """
        Initialize document processor
        
        Args:
            extraction_service (ExtractionService, optional): Custom extraction service
            api_key (str, optional): API authentication key
            api_token (str, optional): API authentication token
        """
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize services
        self.extraction_service = extraction_service or ExtractionService(
            Config.OPENAI_API_KEY
        )
        
        # API credentials
        self.api_key = api_key
        self.api_token = api_token
    
    def process_director_documents(
        self, 
        directors_data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Process and validate documents for all directors
        
        Args:
            directors_data (dict): Director document information
        
        Returns:
            dict: Processed and validated director documents
        """
        processed_directors = {}
        
        for director_key, director_info in directors_data.items():
            try:
                # Validate and process individual director
                processed_director = self._process_single_director(director_info)
                processed_directors[director_key] = processed_director
            
            except Exception as e:
                self.logger.error(f"Error processing director {director_key}: {e}")
                processed_directors[director_key] = {
                    'error': str(e)
                }
        
        return processed_directors
    
    def _process_single_director(
        self, 
        director_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process documents for a single director
        
        Args:
            director_info (dict): Director document information
        
        Returns:
            dict: Processed director document details
        """
        # Validate nationality
        nationality = self._validate_nationality(
            director_info.get('nationality', '')
        )
        
        # Validate authorization
        is_authorised = self._validate_authorization(
            director_info.get('authorised', '')
        )
        
        # Process individual documents
        documents = director_info.get('documents', {})
        processed_documents = self._process_director_document_set(documents)
        
        return {
            'nationality': nationality,
            'is_authorised': is_authorised,
            'documents': processed_documents
        }
    
    def _process_director_document_set(
        self, 
        documents: Dict[str, str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Process and validate a set of director documents
        
        Args:
            documents (dict): Document URLs
        
        Returns:
            dict: Processed document details
        """
        processed_docs = {}
        
        # Define required document types
        required_docs = [
            'adhereCardFront', 
            'adhereCardBack', 
            'passportPhoto', 
            'address_proof'
        ]
        
        for doc_key in required_docs:
            try:
                # Get document URL
                doc_url = documents.get(doc_key)
                
                if not doc_url:
                    processed_docs[doc_key] = {
                        'error': f'Missing document: {doc_key}'
                    }
                    continue
                
                # Validate document URL
                if not self._validate_document_url(doc_url):
                    processed_docs[doc_key] = {
                        'error': f'Invalid document URL: {doc_url}'
                    }
                    continue
                
                # Determine document type
                doc_type = self._get_document_type(doc_key)
                
                # Extract document data
                extracted_data = self.extraction_service.extract_document_data(
                    doc_url, 
                    doc_type
                )
                
                processed_docs[doc_key] = {
                    'url': doc_url,
                    'document_type': doc_type,
                    'is_valid': True,
                    'extracted_data': extracted_data
                }
            
            except Exception as e:
                self.logger.error(f"Error processing document {doc_key}: {e}")
                processed_docs[doc_key] = {
                    'error': str(e)
                }
        
        return processed_docs
    
    def process_company_documents(
        self, 
        company_docs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process and validate company-level documents
        
        Args:
            company_docs (dict): Company document information
        
        Returns:
            dict: Processed company document details
        """
        try:
            # Validate address proof type
            address_proof_type = company_docs.get('address_proof_type', '')
            
            # Validate address proof document
            address_proof_url = company_docs.get('addressProof')
            
            if not address_proof_url:
                return {
                    'error': 'Missing address proof document'
                }
            
            # Validate document URL
            if not self._validate_document_url(address_proof_url):
                return {
                    'error': f'Invalid address proof URL: {address_proof_url}'
                }
            
            # Extract document data
            extracted_data = self.extraction_service.extract_document_data(
                address_proof_url, 
                'address_proof'
            )
            
            return {
                'address_proof_type': address_proof_type,
                'addressProof': {
                    'url': address_proof_url,
                    'is_valid': True,
                    'extracted_data': extracted_data
                }
            }
        
        except Exception as e:
            self.logger.error(f"Error processing company documents: {e}")
            return {
                'error': str(e)
            }
    
    def _validate_nationality(self, nationality: str) -> str:
        """
        Validate director nationality
        
        Args:
            nationality (str): Nationality to validate
        
        Returns:
            str: Validated nationality
        """
        try:
            return NationalityType(nationality.strip()).value
        except ValueError:
            self.logger.warning(f"Invalid nationality: {nationality}")
            return 'Unknown'
    
    def _validate_authorization(self, authorisation: str) -> bool:
        """
        Validate director authorization status
        
        Args:
            authorisation (str): Authorization status
        
        Returns:
            bool: Validated authorization status
        """
        valid_authorizations = {'Yes', 'No'}
        
        try:
            parsed_auth = authorisation.strip().capitalize()
            
            if parsed_auth not in valid_authorizations:
                self.logger.warning(f"Invalid authorization status: {authorisation}")
                return False
            
            return parsed_auth == 'Yes'
        
        except Exception:
            self.logger.warning(f"Invalid authorization status: {authorisation}")
            return False
    
    def _validate_document_url(self, url: str) -> bool:
        """
        Validate document URL
        
        Args:
            url (str): Document URL to validate
        
        Returns:
            bool: Whether URL is valid
        """
        return DocumentDownloader.validate_url(url)
    
    def _get_document_type(self, doc_key: str) -> str:
        """
        Determine document type from document key
        
        Args:
            doc_key (str): Document key
        
        Returns:
            str: Detected document type
        """
        doc_type_mapping = {
            'adhereCardFront': 'aadhar',
            'adhereCardBack': 'aadhar',
            'passportPhoto': 'passport',
            'address_proof': 'address_proof'
        }
        
        return doc_type_mapping.get(doc_key, 'unknown')