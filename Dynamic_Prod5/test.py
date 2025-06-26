import requests
import json

# API endpoint
url = "http://localhost:8000/validate-documents"

# Sample input data
payload = {
    "service_id": "1",
    "request_id": "REQ_12345",
    "directors": {
        "director1": {
            "nationality": "Indian",
            "authorised": "Yes",
            "documents": {
                "aadharCardFront": "https://drive.google.com/file/d/1l4E404heg2wplLB6NzTW1SvcJ1Mp07tO/view?usp=sharing",
                "aadharCardBack": "https://drive.google.com/file/d/19YBInQb4Z6yWFJ1g-Atc1pMaZ0mrSrV4/view?usp=share_link",
                "panCard": "https://drive.google.com/file/d/1BIl0_NLdmDLowDlNGUOP28NDPvi78neX/view?usp=sharing",
                "passportPhoto": "https://drive.google.com/file/d/1m3k6UrypP9Ctws5be4ywZ2Hmu8_rA6_3/view?usp=sharing",
                "address_proof": "https://drive.google.com/file/d/1pYJJ19EvfHJJs7WHSjYXEesj19zX6KMW/view?usp=sharing"
            }
        }
    },
    "companyDocuments": {
        "address_proof_type": "Utility Bill",
        "addressProof": "https://drive.google.com/file/d/18mlfWiQE5mMzOxgoiF3LCyZh68gCROsS/view?usp=sharing"
    }
}

# Print payload to verify
print("Payload to be sent:")
print(json.dumps(payload, indent=2))

# Send POST request
try:
    response = requests.post(url, json=payload)
    
    # Print full response details
    print("\nResponse Status Code:", response.status_code)
    print("\nResponse Headers:")
    for key, value in response.headers.items():
        print(f"{key}: {value}")
    
    print("\nResponse Content:")
    print(json.dumps(response.json(), indent=2))

except requests.exceptions.RequestException as e:
    print(f"Request error: {e}")