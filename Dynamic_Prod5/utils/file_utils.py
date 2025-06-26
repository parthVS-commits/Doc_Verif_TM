import requests
import logging
import json
from urllib.parse import urlparse

class DocumentDownloader:
    """
    Utility for downloading and validating documents
    """
    
    @staticmethod
    def download_document(url, timeout=30):
        """
        Download a document from a given URL
        
        Args:
            url (str): Document URL
            timeout (int): Request timeout in seconds
        
        Returns:
            bytes or None: Document content
        """
        try:
            # Validate URL
            if not DocumentDownloader.validate_url(url):
                logging.error(f"Invalid URL: {url}")
                return None
            
            # Prepare headers
            headers = {
                "User-Agent": "DocumentValidationService/1.0",
                "Accept": "*/*"
            }
            
            # For S3 signed URLs, use specific headers
            if 's3.amazonaws.com' in url:
                headers["Range"] = "bytes=0-"
            
            # Download document
            response = requests.get(
                url, 
                headers=headers, 
                timeout=timeout,
                stream=True
            )
            
            # Check response
            if response.status_code not in [200, 206]:  # 206 is Partial Content for range requests
                logging.error(f"Failed to download document. Status: {response.status_code}")
                return None
            
            return response.content
        
        except requests.exceptions.RequestException as e:
            logging.error(f"Download error: {str(e)}")
            return None
    
    @staticmethod
    def validate_url(url):
        """
        Validate document URL
        
        Args:
            url (str): URL to validate
        
        Returns:
            bool: Whether URL is valid
        """
        try:
            # Parse URL
            result = urlparse(url)
            
            # Check scheme and netloc
            return all([
                result.scheme in ['http', 'https'],
                result.netloc
            ])
        
        except Exception as e:
            logging.error(f"URL validation error: {str(e)}")
            return False
    
    @staticmethod
    def verify_document_access(url):
        """
        Verify document URL accessibility
        
        Args:
            url (str): Document URL
        
        Returns:
            bool: Whether document is accessible
        """
        try:
            # HEAD request to verify access
            response = requests.head(
                url, 
                timeout=10, 
                allow_redirects=True
            )
            
            return response.status_code in [200, 206]
        
        except Exception as e:
            logging.error(f"Document access verification error: {str(e)}")
            return False

class APIDocumentFetcher:
    """
    Utility for fetching documents via API
    """
    
    @staticmethod
    def fetch_documents(
        document_id, 
        api_key, 
        api_token, 
        base_url="https://qe-vsapi.vakilsearch.com/api/v1"
    ):
        """
        Fetch documents from API
        
        Args:
            document_id (str): Document identifier
            api_key (str): API authentication key
            api_token (str): API authentication token
            base_url (str): Base API URL
        
        Returns:
            dict or None: Fetched document data
        """
        try:
            # Validate inputs
            if not all([document_id, api_key, api_token]):
                logging.error("Missing required API parameters")
                return None
            
            # Construct URL
            url = f"{base_url}/get_documents/{document_id}"
            
            # Prepare headers
            headers = {
                "key": api_key.strip(),
                "token": api_token.strip()
            }
            
            # Log request details (with sensitive info masked)
            logging.info(f"Fetching documents for ID: {document_id}")
            logging.info(f"API Key (first 5 chars): {api_key[:5]}...")
            
            # Make API request
            response = requests.get(
                url, 
                headers=headers, 
                timeout=30
            )
            
            # Check response status
            if response.status_code == 401 or response.status_code == 403:
                logging.error("Authentication failed - invalid API credentials")
                return None
            
            if response.status_code != 200:
                logging.error(f"API returned status code: {response.status_code}")
                logging.error(f"Response content: {response.text}")
                return None
            
            # Parse JSON response
            try:
                documents_data = response.json()
                
                # Validate response structure
                if not isinstance(documents_data, dict):
                    logging.error(f"API returned unexpected data type: {type(documents_data)}")
                    return None
                
                # Check for required fields
                if 'director_documents' not in documents_data:
                    logging.error("API response missing 'director_documents' field")
                    return None
                
                logging.info("Documents fetched successfully")
                return documents_data
            
            except json.JSONDecodeError:
                logging.error("Failed to decode JSON response from API")
                return None
        
        except requests.exceptions.Timeout:
            logging.error("API request timed out")
            return None
        
        except requests.exceptions.ConnectionError:
            logging.error("Connection error when contacting API")
            return None
        
        except Exception as e:
            logging.error(f"Error fetching documents: {str(e)}")
            return None
    
    @staticmethod
    def get_fresh_document_url(
        document_id, 
        api_key, 
        api_token, 
        director_name, 
        document_type
    ):
        """
        Get a fresh S3 URL for a specific document
        
        Args:
            document_id (str): Main document ID
            api_key (str): API key
            api_token (str): API token
            director_name (str): Name of the director
            document_type (str): Type of document
        
        Returns:
            str or None: Fresh document URL
        """
        try:
            # Fetch fresh document data
            documents_data = APIDocumentFetcher.fetch_documents(
                document_id, api_key, api_token
            )
            
            if not documents_data:
                logging.error("Failed to fetch fresh document data")
                return None
            
            # Find the specified director
            for director in documents_data.get('director_documents', []):
                if director.get('name') == director_name:
                    # Find the specified document
                    for doc in director.get('documents', []):
                        if doc.get('document_category') == document_type:
                            doc_urls = doc.get('document_url', [])
                            if doc_urls and len(doc_urls) > 0:
                                logging.info(f"Found fresh URL for {document_type}")
                                return doc_urls[0]
            
            logging.warning(f"Could not find {document_type} for director {director_name}")
            return None
        
        except Exception as e:
            logging.error(f"Error getting fresh document URL: {str(e)}")
            return None

# Configure basic logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)