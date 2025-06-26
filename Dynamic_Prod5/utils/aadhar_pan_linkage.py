import time
import requests
import logging
import random
from typing import Dict, Any
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import re

class AadharPanLinkageService:
    """
    Enhanced service to verify Aadhar and PAN linkage with robust error handling
    """
    
    @staticmethod
    def _create_retry_session(retries=3, backoff_factor=0.3):
        """
        Create a robust requests session with retry mechanism
        
        Args:
            retries (int): Number of retries
            backoff_factor (float): Backoff multiplier between attempts
        
        Returns:
            requests.Session: Configured session
        """
        # Configure retry strategy
        retry_strategy = Retry(
            total=retries,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],   #method_whitelist
            backoff_factor=backoff_factor
        )
        
        # Create adapter
        adapter = HTTPAdapter(max_retries=retry_strategy)
        
        # Create session
        session = requests.Session()
        session.mount("https://", adapter)
        
        return session
    
    @staticmethod
    def verify_linkage(
        aadhar_number: str, 
        pan_number: str,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Advanced Aadhar and PAN linkage verification
        
        Args:
            aadhar_number (str): Aadhar number
            pan_number (str): PAN number
            max_retries (int): Maximum number of retries
        
        Returns:
            dict: Linkage verification result
        """
        # Validate input
        if not aadhar_number or not pan_number:
            return {
                'is_linked': False,
                'message': 'Invalid Aadhar or PAN number',
                'error': 'invalid_input'
            }
        
        # Clean and validate Aadhar number
        cleaned_aadhar = re.sub(r'\D', '', aadhar_number)
        if len(cleaned_aadhar) != 12:
            return {
                'is_linked': False,
                'message': 'Invalid Aadhar number format',
                'error': 'invalid_aadhar'
            }
        
        # Clean and validate PAN number
        cleaned_pan = pan_number.strip().upper()
        if not re.match(r'^[A-Z]{5}\d{4}[A-Z]{1}$', cleaned_pan):
            return {
                'is_linked': False,
                'message': 'Invalid PAN number format',
                'error': 'invalid_pan'
            }
        
        try:
            # Create robust session
            session = AadharPanLinkageService._create_retry_session(max_retries)
            
            # Prepare request with better error handling
            url = 'https://eportal.incometax.gov.in/iec/servicesapi/getEntity'
            
            # Enhanced request payload
            payload = {
                "aadhaarNumber": cleaned_aadhar,
                "pan": cleaned_pan,
                "preLoginFlag": "Y",
                "serviceName": "linkAadhaarPreLoginService"
            }
            
            # More robust headers
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
                'Accept': 'application/json',
                'X-Requested-With': 'XMLHttpRequest'
            }
            
            # Add randomized delays between attempts
            time.sleep(random.uniform(0.5, 1.5))
            
            # Make request with timeout and error handling
            try:
                response = session.post(
                    url, 
                    json=payload, 
                    headers=headers,
                    timeout=(10, 30)  # Connection and read timeout
                )
                
                # Log raw response for debugging
                logging.info(f"Linkage API Response Status: {response.status_code}")
                logging.info(f"Response Content: {response.text}")
                
                # Enhanced response parsing
                if response.status_code == 200:
                    try:
                        result = response.json()
                        
                        # Check for success messages
                        if 'messages' in result and isinstance(result['messages'], list):
                            for message in result['messages']:
                                # Success scenarios
                                if (message.get('code') == 'EF40124' or 
                                    ('desc' in message and 'already linked' in message.get('desc', '').lower())):
                                    return {
                                        'is_linked': True,
                                        'message': message.get('desc', 'Aadhar and PAN are successfully linked'),
                                        'details': message
                                    }
                                
                                # Rate limiting or temporary error
                                if (message.get('code') == 'EF00077' or 
                                    'exceeded the limit' in message.get('desc', '').lower()):
                                    return {
                                        'is_linked': False,
                                        'message': 'API rate limit or temporary error',
                                        'is_rate_limited': True,
                                        'details': message
                                    }
                        
                        # Default fallback for unhandled scenarios
                        return {
                            'is_linked': False,
                            'message': 'Linkage verification inconclusive',
                            'raw_response': result
                        }
                    
                    except ValueError as json_err:
                        logging.error(f"JSON parsing error: {json_err}")
                        return {
                            'is_linked': False,
                            'message': 'Unable to parse API response',
                            'error': 'json_parse_error'
                        }
                
                # Non-200 response handling
                return {
                    'is_linked': False,
                    'message': f'API returned status code {response.status_code}',
                    'error': 'api_error',
                    'status_code': response.status_code
                }
            
            except requests.exceptions.RequestException as req_err:
                logging.error(f"Request error: {req_err}")
                return {
                    'is_linked': False,
                    'message': f'Network error: {str(req_err)}',
                    'error': 'network_error',
                    'details': str(req_err)
                }
        
        except Exception as e:
            logging.error(f"Comprehensive linkage verification error: {e}")
            return {
                'is_linked': False,
                'message': 'Unexpected error during verification',
                'error': 'unexpected_error',
                'details': str(e)
            }