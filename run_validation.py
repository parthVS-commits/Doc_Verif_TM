import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to Python path
import sys
import os
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from api.document_validation_api import DocumentValidationAPI

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('document_validation.log'),
        logging.StreamHandler()
    ]
)

def print_api_response(api_response):
    """
    Print API response for testing and debugging
    
    Args:
        api_response (dict): Formatted API response
    """
    print("\n==== API Response ====")
    print(json.dumps(api_response, indent=2))

def main():
    # Sample input data with updated document structure
    sample_input = {
        "service_id": "1",
        "preconditions": {
          "owner_name": "Parth"
        },
        "request_id": "REQ_12345",
        "directors": {
            "director1": {
                "nationality": "Indian",
                "authorised": "Yes",
                "documents": {
                    "aadharCardFront": "https://drive.google.com/file/d/13VuASJkX9SFFTiRWYaJXP68PqEweQgNH/view?usp=sharing",
                    "aadharCardBack": "https://drive.google.com/file/d/13VuASJkX9SFFTiRWYaJXP68PqEweQgNH/view?usp=sharing",
                    "panCard": "https://drive.google.com/file/d/1OvxgK1Ndelcih0AMiGWBuZoxW6MawD5M/view?usp=sharing",
                    "passportPhoto": "https://drive.google.com/file/d/15ePWxHU016AcKCexKflH--I7zMCxQsMA/view?usp=sharing",
                    "address_proof": "https://drive.google.com/file/d/1S1bonPYt8cs9JwvJFeetNjRCnhHeE09e/view?usp=sharing",
                    "signature": "https://drive.google.com/file/d/1wDAY45fd6bVmPs0AbejzVxFR42uAxS1-/view?usp=sharing"
                }
            },
            "director2": {
                "nationality": "Foreign",
                "authorised": "Yes",
                "documents": {
                    "panCard": "https://drive.google.com/file/d/1l5bun1D9ISYqAYImj2VAwaoOlRPWv9pS/view?usp=sharing",
                    "passportPhoto": "https://drive.google.com/file/d/1Y85m-9233Mx6pq-y-FObD2cQc6AmO2Om/view?usp=sharing",
                    "address_proof": "https://drive.google.com/file/d/1pMokK6ZhlYFbydFsRJJgHi7Szy9k9FxD/view?usp=sharing",
                    "signature": "https://drive.google.com/file/d/1iOig2z_PDHkKFMU4m0_sLZngrmjHYrAD/view?usp=sharing"
                }
            },
            "director3": {
                "nationality": "Indian",
                "authorised": "Yes",
                "documents": {
                    "aadharCardFront": "https://drive.google.com/file/d/1Bu4OAIBunwj2GT9tjnOtH00yu8EIoywE/view?usp=sharing",
                    "aadharCardBack": "https://drive.google.com/file/d/1MgKSiyNhdYcSFBO1IhLBnV_o7QmR_UiL/view?usp=sharing",
                    "panCard": "https://drive.google.com/file/d/1CC7aWWG-RCDT-z1VWohlEEsHWND22LWQ/view?usp=sharing",
                    "passportPhoto": "https://drive.google.com/file/d/15ePWxHU016AcKCexKflH--I7zMCxQsMA/view?usp=sharing",
                    "address_proof": "https://drive.google.com/file/d/1NFi-kr6Zenp2kwdYRQ4rRlc7uO8fo0tv/view?usp=sharing",
                    "signature": "https://drive.google.com/file/d/1RXn40z-xONfuxg2Jic0IZ7fvVXvTKdSV/view?usp=sharing"
                }
            }
        },
        "companyDocuments": {
            "address_proof_type": "Utility Bill",
            "addressProof": "https://drive.google.com/file/d/1S1bonPYt8cs9JwvJFeetNjRCnhHeE09e/view?usp=sharing",
            "noc": "https://drive.google.com/file/d/1rkGAKSnsLNfUlDudwj25B9nQSQTntzPe/view?usp=sharing"
        }
    }

    # Initialize API
    validation_api = DocumentValidationAPI()

    try:
        # Validate documents
        api_response, detailed_result = validation_api.validate_document(sample_input)

        # Print results in the required API format
        print_api_response(api_response)

        # Optional: Print processing time
        if 'metadata' in detailed_result:
            print(f"\nProcessing Time: {detailed_result['metadata'].get('processing_time', 0):.2f} seconds")

        # Optional: Save results to files
        with open('api_response.json', 'w') as f:
            json.dump(api_response, f, indent=2)
        
        with open('detailed_validation_results.json', 'w') as f:
            json.dump(detailed_result, f, indent=2)
        
        print("\nResults saved to api_response.json and detailed_validation_results.json")

    except Exception as e:
        print(f"Validation error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()