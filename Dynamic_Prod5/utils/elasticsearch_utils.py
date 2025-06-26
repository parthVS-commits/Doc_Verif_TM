from elasticsearch import Elasticsearch
from utils.logging_utils import logger
from config.settings import Config

class ElasticsearchClient:
    """
    Elasticsearch connection and query management
    """
    
    def __init__(self, config=None):
        """
        Initialize Elasticsearch client
        
        Args:
            config (dict, optional): Custom Elasticsearch configuration
        """
        self.config = config or Config.get_elasticsearch_config()
        self.client = self._create_client()
    
    def _create_client(self):
        """
        Create Elasticsearch client
        
        Returns:
            Elasticsearch: Configured Elasticsearch client
        """
        try:
            es_client = Elasticsearch(**self.config)
            
            # Verify connection
            if not es_client.ping():
                logger.error("Elasticsearch connection failed")
                return None
            
            logger.info("Elasticsearch connection established successfully")
            return es_client
        
        except Exception as e:
            logger.error(f"Elasticsearch connection error: {str(e)}")
            return None
    
    def get_compliance_rules(self, service_id):
        """
        Retrieve compliance rules for a specific service ID
        
        Args:
            service_id (str): Service identifier
        
        Returns:
            list: Matching compliance rules
        """
        try:
            # Search query with exact service_id match
            search_query = {
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"service_id.keyword": str(service_id)}}
                        ]
                    }
                }
            }
            
            # Execute search
            results = self.client.search(
                index=Config.VALIDATION_RULES_INDEX, 
                body=search_query
            )
            
            # Convert Elasticsearch response to standard list
            rules = [hit['_source'] for hit in results.body['hits']['hits']]
            
            logger.info(f"Retrieved {len(rules)} rules for service ID: {service_id}")
            
            return rules
        
        except Exception as e:
            logger.error(f"Error retrieving compliance rules: {str(e)}")
            return []
    
    def validate_index_exists(self, index_name=None):
        """
        Check if an Elasticsearch index exists
        
        Args:
            index_name (str, optional): Index to check. 
                                        Uses default from config if not provided
        
        Returns:
            bool: Whether the index exists
        """
        index_to_check = index_name or Config.VALIDATION_RULES_INDEX
        
        try:
            return self.client.indices.exists(index=index_to_check)
        except Exception as e:
            logger.error(f"Error checking index existence: {str(e)}")
            return False
    
    def create_index_if_not_exists(
        self, 
        index_name=None, 
        mapping=None
    ):
        """
        Create an Elasticsearch index if it doesn't exist
        
        Args:
            index_name (str, optional): Index name
            mapping (dict, optional): Index mapping
        
        Returns:
            bool: Whether index was created or already exists
        """
        index_to_create = index_name or Config.VALIDATION_RULES_INDEX
        
        # Default mapping if not provided
        default_mapping = {
            "mappings": {
                "properties": {
                    "service_id": {"type": "keyword"},
                    "rules": {"type": "nested"},
                    "name": {"type": "text"}
                }
            }
        }
        
        mapping = mapping or default_mapping
        
        try:
            # Check if index exists
            if not self.validate_index_exists(index_to_create):
                # Create index
                self.client.indices.create(
                    index=index_to_create, 
                    body=mapping
                )
                logger.info(f"Created index: {index_to_create}")
                return True
            
            logger.info(f"Index {index_to_create} already exists")
            return True
        
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            return False

# Global Elasticsearch client
es_client = ElasticsearchClient()