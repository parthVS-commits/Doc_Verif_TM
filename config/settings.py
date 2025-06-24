import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """
    Centralized configuration management for the document validation system
    
    This class provides:
    - Elasticsearch connection settings
    - OpenAI API configuration
    - Logging settings
    - API and validation rule configurations
    """
    
    # Elasticsearch Configuration
    ELASTICSEARCH_HOST = os.getenv(
        'ELASTICSEARCH_HOST', 
        'https://vs-production.es.ap-south-1.aws.elastic-cloud.com:9243'
    )
    ELASTICSEARCH_USERNAME = os.getenv('ELASTICSEARCH_USERNAME', 'elastic')
    ELASTICSEARCH_PASSWORD = os.getenv('ELASTICSEARCH_PASSWORD', '')

    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'document_validation.log')

    # API Configurations
    API_BASE_URL = os.getenv('API_BASE_URL', 'https://qe-vsapi.vakilsearch.com/api/v1')

    # Document Validation Rules
    VALIDATION_RULES_INDEX = os.getenv('VALIDATION_RULES_INDEX', 'compliance_rules')

    @classmethod
    def get_elasticsearch_config(cls):
        """
        Get Elasticsearch connection configuration
        
        Returns:
            dict: Elasticsearch connection parameters
        """
        return {
            'hosts': [cls.ELASTICSEARCH_HOST],
            'http_auth': (cls.ELASTICSEARCH_USERNAME, cls.ELASTICSEARCH_PASSWORD)
        }

    @classmethod
    def validate_config(cls):
        """
        Validate critical configuration parameters
        
        Returns:
            bool: Whether configuration is valid
        """
        errors = []
        
        # Check critical configurations
        if not cls.ELASTICSEARCH_HOST:
            errors.append("Missing Elasticsearch Host")
        
        if not cls.ELASTICSEARCH_USERNAME:
            errors.append("Missing Elasticsearch Username")
        
        if not cls.ELASTICSEARCH_PASSWORD:
            errors.append("Missing Elasticsearch Password")
        
        # Log errors if any
        if errors:
            print("Configuration Errors:")
            for error in errors:
                print(f"- {error}")
            return False
        
        return True