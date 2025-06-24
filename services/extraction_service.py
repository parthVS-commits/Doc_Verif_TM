import os
import re
import json
import logging
import base64
import io
from datetime import datetime
from typing import Dict, Any, Optional

import requests
import openai
from PIL import Image
import PyPDF2
from pdf2image import convert_from_bytes

# Import extraction prompts
from .extraction_prompts import (
    get_aadhar_extraction_prompt,
    get_pan_extraction_prompt,
    get_passport_extraction_prompt,
    get_driving_license_extraction_prompt,
    get_address_proof_extraction_prompt,
    get_bill_extraction_prompt,
    get_passport_photo_extraction_prompt,
    get_signature_extraction_prompt,
    get_noc_extraction_prompt,
    get_generic_extraction_prompt,
    get_consent_letter_extraction_prompt,
    get_board_resolution_extraction_prompt,
    get_msme_certificate_extraction_prompt,
    get_dipp_certificate_extraction_prompt,
    get_trademark_verification_document_prompt

)

class ExtractionService:
    """
    Advanced document data extraction service using AI Vision
    """
    
    def __init__(self, openai_api_key=None):
        """
        Initialize the extraction service
        
        Args:
            openai_api_key (str, optional): OpenAI API key
        """
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        
        # Logging handler for detailed tracking
        file_handler = logging.FileHandler('document_extraction.log')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # API Key configuration
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
            self.logger.info("OpenAI API key initialized successfully")

    def _convert_pdf_to_image(self, pdf_data):
        """
        Convert PDF to image
        
        Args:
            pdf_data (bytes): PDF document data
        
        Returns:
            bytes: Converted image data
        """
        try:
            # Convert first page of PDF to image
            images = convert_from_bytes(
                pdf_data, 
                first_page=1, 
                last_page=1,
                fmt='png'
            )
            
            if not images:
                self.logger.error("PDF to image conversion produced no images")
                return None
            
            # Convert to bytes
            byte_arr = io.BytesIO()
            images[0].save(byte_arr, format='PNG')
            return byte_arr.getvalue()
        
        except Exception as e:
            self.logger.error(f"PDF conversion error: {str(e)}")
            return None

    def _verify_aadhar_data(self, data):
        """
        Verify Aadhar document data
        
        Args:
            data (dict): Extracted Aadhar data
        
        Returns:
            dict: Verified data or None
        """
        required_fields = ['name', 'aadhar_number']
        
        for field in required_fields:
            if not data.get(field):
                self.logger.warning(f"Missing required Aadhar field: {field}")
                return None
        
        # Optional validations
        if data.get('is_masked', False):
            self.logger.warning("Masked Aadhar not allowed")
            return None
        
        return data

    def _verify_pan_data(self, data):
        """
        Verify PAN card document data
        
        Args:
            data (dict): Extracted PAN data
        
        Returns:
            dict: Verified data or None
        """
        required_fields = ['name', 'pan_number', 'dob']
        
        for field in required_fields:
            if not data.get(field):
                self.logger.warning(f"Missing required PAN field: {field}")
                return None
        
        # Validate PAN number format
        if not re.match(r'^[A-Z]{5}\d{4}[A-Z]{1}$', data['pan_number']):
            self.logger.warning("Invalid PAN number format")
            return None
        
        return data

    def _verify_passport_data(self, data):
        """
        Verify passport document data
        
        Args:
            data (dict): Extracted passport data
        
        Returns:
            dict: Verified data or None
        """
        required_fields = ['name', 'passport_number', 'dob', 'expiry_date']
        
        for field in required_fields:
            if not data.get(field):
                self.logger.warning(f"Missing required passport field: {field}")
                return None
        
        # Check passport validity
        try:
            expiry_date = datetime.strptime(data['expiry_date'], '%d/%m/%Y')
            if expiry_date < datetime.now():
                self.logger.warning("Passport has expired")
                return None
        except Exception:
            self.logger.warning("Invalid passport expiry date")
            return None
        
        return data

    def _verify_passport_photo_data(self, data):
        """
        Verify passport photo data
        
        Args:
            data (dict): Extracted passport photo data
        
        Returns:
            dict: Verified data or None
        """
        required_fields = ['clarity_score', 'is_passport_style', 'face_visible']
        
        for field in required_fields:
            if field not in data:
                self.logger.warning(f"Missing required passport photo field: {field}")
                return None
        
        # Validate clarity
        if data['clarity_score'] < 0.7:
            self.logger.warning("Passport photo clarity too low")
            return None
        
        # Validate photo requirements
        if not data['is_passport_style'] or not data['face_visible']:
            self.logger.warning("Passport photo does not meet requirements")
            return None
        
        return data

    def _verify_signature_data(self, data):
        """
        Verify signature data
        
        Args:
            data (dict): Extracted signature data
        
        Returns:
            dict: Verified data or None
        """
        required_fields = ['clarity_score', 'is_handwritten', 'is_complete']
        
        for field in required_fields:
            if field not in data:
                self.logger.warning(f"Missing required signature field: {field}")
                return None
        
        # Validate clarity
        if data['clarity_score'] < 0.7:
            self.logger.warning("Signature clarity too low")
            return None
        
        # Validate signature requirements
        if not data['is_handwritten'] or not data['is_complete']:
            self.logger.warning("Signature does not meet requirements")
            return None
        
        return data
    
    def extract_document_data(self, document_url, document_type):
        """
        Extract data from a document with comprehensive validation
        
        Args:
            document_url (str): URL of the document
            document_type (str): Type of document
        
        Returns:
            dict: Extracted document data with verification metrics
        """
        extraction_start_time = datetime.now()
        
        try:
            # Log extraction attempt
            self.logger.info(f"Starting document extraction: {document_type}")
            self.logger.debug(f"Document URL: {document_url}")
            
            # Download document
            document_data = self._download_document(document_url)
            
            if not document_data:
                self.logger.error(f"Failed to download {document_type} document")
                return self._create_extraction_failure_record(document_type, "Download failed")
            
            # Convert to supported image format
            image_data = self._convert_to_supported_image(document_data)
            
            if not image_data:
                self.logger.error(f"Failed to convert {document_type} to supported image format")
                return self._create_extraction_failure_record(document_type, "Image conversion failed")
            
            # Choose extraction prompt based on document type
            extraction_prompt = self._select_extraction_prompt(document_type)
            
            # Extract data using AI
            extracted_data = self._extract_with_ai(
                image_data, 
                document_type,
                extraction_prompt
            )
            
            # Verify extracted data
            verified_data = self._verify_extracted_data(
                extracted_data, 
                document_type
            )
            
            # Log extraction metrics
            extraction_end_time = datetime.now()
            extraction_duration = (extraction_end_time - extraction_start_time).total_seconds()
            
            self.logger.info(
                f"Document extraction completed: {document_type}, "
                f"Duration: {extraction_duration:.2f} seconds"
            )
            
            return verified_data or self._create_extraction_failure_record(document_type, "Verification failed")
        
        except Exception as e:
            self.logger.error(
                f"Comprehensive document extraction error for {document_type}: {str(e)}", 
                exc_info=True
            )
            return self._create_extraction_failure_record(document_type, str(e))
    
    def _select_extraction_prompt(self, document_type):
        """
        Select appropriate extraction prompt based on document type
        
        Args:
            document_type (str): Type of document
        
        Returns:
            str: Extraction prompt
        """
        extraction_prompts = {
            'aadhar': get_aadhar_extraction_prompt(),
            'aadhar_front': get_aadhar_extraction_prompt(),
            'aadhar_back': get_aadhar_extraction_prompt(),
            'pan': get_pan_extraction_prompt(),
            'passport': get_passport_extraction_prompt(),
            'passport_photo': get_passport_photo_extraction_prompt(),
            'address_proof': get_address_proof_extraction_prompt(),
            'electricity_bill': get_bill_extraction_prompt(),
            'signature': get_signature_extraction_prompt(),
            'driving_license': get_driving_license_extraction_prompt(),
            'noc': get_noc_extraction_prompt(),
            'consent_letter': get_consent_letter_extraction_prompt(),
            'board_resolution': get_board_resolution_extraction_prompt(),
            'msme_certificate': get_msme_certificate_extraction_prompt(),
            'dipp_certificate': get_dipp_certificate_extraction_prompt(),
            'trademark_verification': get_trademark_verification_document_prompt()
        }
        
        return extraction_prompts.get(
            document_type.lower(), 
            get_generic_extraction_prompt()
        )
    
    def _verify_extracted_data(self, extracted_data, document_type):
        """
        Verify extracted data for consistency and completeness
        
        Args:
            extracted_data (dict): Extracted document data
            document_type (str): Type of document
        
        Returns:
            dict: Verified document data
        """
        if not extracted_data:
            return None
        
        # Implement type-specific verification logic
        verifications = {
            'aadhar': self._verify_aadhar_data,
            'pan': self._verify_pan_data,
            'passport': self._verify_passport_data,
            # Add more verification methods for different document types
        }
        
        verification_method = verifications.get(
            document_type.lower(), 
            self._generic_data_verification
        )
        
        return verification_method(extracted_data)
    
    def _verify_aadhar_data(self, data):
        """
        Verify Aadhar document data
        
        Args:
            data (dict): Extracted Aadhar data
        
        Returns:
            dict: Verified data or None
        """
        required_fields = ['name', 'aadhar_number', 'address']
        
        for field in required_fields:
            if not data.get(field):
                self.logger.warning(f"Missing required Aadhar field: {field}")
                return None
        
        return data
    
    # Similar verification methods for other document types...
    
    def _generic_data_verification(self, data):
        """
        Generic data verification method
        
        Args:
            data (dict): Extracted document data
        
        Returns:
            dict: Verified data or None
        """
        if not data:
            return None
        
        # Basic verification: ensure data is not empty
        non_empty_fields = [
            value for value in data.values() 
            if value and value not in [None, '', 'Not Extracted']
        ]
        
        if len(non_empty_fields) < len(data) / 2:
            self.logger.warning("Insufficient meaningful data extracted")
            return None
        
        return data
    
    def _create_extraction_failure_record(self, document_type, error_message):
        """
        Create a standardized failure record for extraction
        
        Args:
            document_type (str): Type of document
            error_message (str): Description of extraction failure
        
        Returns:
            dict: Failure record
        """
        self.logger.error(f"Extraction failure for {document_type}: {error_message}")
        
        return {
            'extraction_status': 'failed',
            'document_type': document_type,
            'error_message': error_message,
            'clarity_score': 0.0
        }
    
    def _download_document(self, url):
        try:
            # Enhanced Google Drive link handling
            if 'drive.google.com' in url:
                # Extract file ID more robustly
                file_id_match = re.search(r'/d/([a-zA-Z0-9_-]+)', url)
                if file_id_match:
                    file_id = file_id_match.group(1)
                    url = f'https://drive.google.com/uc?export=download&id={file_id}'
            
            # Robust download with multiple retries
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Accept": "*/*"
            }
            
            response = requests.get(
                url, 
                headers=headers, 
                allow_redirects=True,
                timeout=30
            )
            
            # Validate response
            if response.status_code == 200:
                return response.content
            
            self.logger.error(f"Download failed: {response.status_code}")
            return None
        
        except Exception as e:
            self.logger.error(f"Document download error: {str(e)}")
            return None
        
    def _convert_to_supported_image(self, document_data):
        """
        Convert document to a supported image format with comprehensive logging
        
        Args:
            document_data (bytes): Original document data
        
        Returns:
            bytes: Converted image data
        """
        # Log initial document data details
        self.logger.info(f"Document data length: {len(document_data)} bytes")
        
        # Log first 100 bytes to inspect content
        self.logger.info(f"First 100 bytes: {document_data[:100].hex()}")
        
        try:
            # Try to identify file type
            def identify_file_type(data):
                # Check common file signatures
                signatures = {
                    'PDF': b'%PDF-',
                    'PNG': b'\x89PNG\r\n\x1a\n',
                    'JPEG': b'\xff\xd8\xff',
                    'GIF': b'GIF87a',
                    'BMP': b'BM',
                    'TIFF': b'\x49\x49\x2a\x00'
                }
                
                for name, sig in signatures.items():
                    if data.startswith(sig):
                        return name
                return "Unknown"
            
            file_type = identify_file_type(document_data)
            self.logger.info(f"Identified file type: {file_type}")
            
            # Try opening as an image first
            try:
                with Image.open(io.BytesIO(document_data)) as img:
                    # Convert to RGB mode to ensure compatibility
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    
                    # Save to a bytes buffer in PNG format
                    byte_arr = io.BytesIO()
                    img.save(byte_arr, format='PNG')
                    return byte_arr.getvalue()
            except (Image.UnidentifiedImageError, IOError) as img_err:
                self.logger.warning(f"Image opening failed: {img_err}")
                
                # Try PDF conversion
                try:
                    return self._convert_pdf_to_image(document_data)
                except Exception as pdf_err:
                    self.logger.error(f"PDF conversion failed: {pdf_err}")
                    
                    # Attempt to decode as text
                    try:
                        text_content = document_data.decode('utf-8', errors='ignore')
                        self.logger.info(f"Decoded text content (first 500 chars): {text_content[:500]}")
                    except Exception as decode_err:
                        self.logger.error(f"Text decoding error: {decode_err}")
                    
                    return None
        
        except Exception as e:
            self.logger.error(f"Comprehensive document conversion error: {str(e)}")
            return None
        
    def _extract_with_ai(self, image_data, document_type, extraction_prompt):
        """
        Extract document data using AI with improved error handling
        
        Args:
            image_data (bytes): Image data to extract
            document_type (str): Type of document being extracted
            extraction_prompt (str): Specific prompt for document extraction
        
        Returns:
            dict or None: Extracted document data
        """
        try:
            # Encode image to base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a precise document data extraction assistant."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": extraction_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                        ]
                    }
                ],
                max_tokens=300
            )
            
            # Parse response
            extracted_text = response.choices[0].message.content
            return self._parse_extraction_result(extracted_text, document_type)
        
        except Exception as e:
            self.logger.error(f"AI extraction error for {document_type}: {str(e)}")
            return None

    def _parse_extraction_result(self, extraction_text, document_type):
        """
        Parse AI extraction result with more robust error handling
        
        Args:
            extraction_text (str): Text returned by AI
            document_type (str): Type of document being extracted
        
        Returns:
            dict or None: Parsed extraction result
        """
        try:
            # Log the full extraction text for debugging
            self.logger.info(f"Full extraction text for {document_type}: {extraction_text}")
            
            # More flexible JSON extraction
            json_match = re.search(r'{.*}', extraction_text, re.DOTALL | re.MULTILINE)
            
            if json_match:
                try:
                    # Try to clean up the JSON string
                    json_str = json_match.group(0)
                    
                    # Remove trailing commas and extra whitespaces
                    json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas in objects
                    json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
                    json_str = re.sub(r'\s+', ' ', json_str)  # Reduce whitespaces
                    
                    parsed_data = json.loads(json_str)
                    
                    # Log parsed data
                    self.logger.info(f"Parsed data for {document_type}: {parsed_data}")
                    
                    # Ensure standard boolean values 
                    for key, value in parsed_data.items():
                        if isinstance(value, str):
                            if value.lower() == 'true':
                                parsed_data[key] = True
                            elif value.lower() == 'false':
                                parsed_data[key] = False
                            elif value.lower() == 'yes':
                                parsed_data[key] = True
                            elif value.lower() == 'no':
                                parsed_data[key] = False
                    
                    return parsed_data
                except json.JSONDecodeError as e:
                    self.logger.error(f"JSON parsing error for {document_type}: {e}")
                    self.logger.error(f"Problematic JSON string: {json_str}")
            
            return None
        
        except Exception as e:
            self.logger.error(f"Result parsing error for {document_type}: {str(e)}")
            return None

        # Existing methods like _download_document, _convert_to_supported_image, 
        # _extract_with_ai would remain largely the same