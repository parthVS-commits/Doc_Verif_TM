import logging
import time
from datetime import datetime, timedelta
import traceback
from typing import Dict, Any, Optional, List, Tuple
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from dateutil import parser
import re
import json
import base64
import tempfile
import os

from services.extraction_service import ExtractionService
from utils.elasticsearch_utils import ElasticsearchClient
from utils.aadhar_pan_linkage import AadharPanLinkageService
from config.settings import Config
from models.document_models import (
    ValidationResult, 
    DocumentValidationError,
    ValidationRuleStatus
)
from rules.compliance_validation_rules import ComplianceValidationRules


class DocumentValidationService:
    """
    Comprehensive document validation service
    """
    
    def __init__(
        self, 
        es_client: Optional[ElasticsearchClient] = None,
        extraction_service: Optional[ExtractionService] = None
    ):
        """
        Initialize the validation service
        
        Args:
            es_client (ElasticsearchClient, optional): Elasticsearch client
            extraction_service (ExtractionService, optional): Document extraction service
        """
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize services
        self.es_client = es_client or ElasticsearchClient()
        self.extraction_service = extraction_service or ExtractionService(
            Config.OPENAI_API_KEY
        )
        self.aadhar_pan_linkage_service = AadharPanLinkageService()

    def _get_compliance_rules(self, service_id: str) -> Dict:
        """
        Retrieve compliance rules for a service
        
        Args:
            service_id (str): Service identifier
        
        Returns:
            dict: Compliance rules
        """
        try:
            # Add explicit None check
            if self.es_client is None:
                self.logger.warning("Elasticsearch client is None, using default rules")
                return self._get_default_compliance_rules()
            
            # Retrieve rules
            rules = self.es_client.get_compliance_rules(service_id)
            
            # Log full details for debugging
            self.logger.info(f"DEBUG: Retrieved rules for service ID {service_id}: {json.dumps(rules, indent=2)}")
            
            # Explicitly filter and find the matching rule set
            matching_rules = [
                rule_set for rule_set in rules 
                if str(rule_set.get('service_id')) == str(service_id)
            ]
            
            # Add additional logging
            self.logger.info(f"DEBUG: Matching rules found: {len(matching_rules)}")
            
            # If matching rules found, return the first one
            if matching_rules:
                selected_rules = matching_rules[0]
                
                # CRITICAL: ALWAYS return rules for the specific service ID
                return {
                    "service_id": service_id,
                    "service_name": selected_rules.get('service_name', 'Unknown Service'),
                    "rules": selected_rules.get('rules', [])
                }
            
            # If no matching rules, log warning and use default
            self.logger.warning(f"No compliance rules found for service ID: {service_id}")
            default_rules = self._get_default_compliance_rules()
            
            # Override service ID in default rules
            default_rules['service_id'] = service_id
            default_rules['service_name'] = f"Default Rules for Service {service_id}"
            
            return default_rules
        
        except Exception as e:
            self.logger.error(f"Error retrieving compliance rules: {e}")
            self.logger.info("Using default compliance rules")
            default_rules = self._get_default_compliance_rules()
            default_rules['service_id'] = service_id
            default_rules['service_name'] = f"Default Rules for Service {service_id}"
            return default_rules
        
        
    def format_validation_results(self, standard_result: Dict) -> str:
        """Format validation results for display"""
        output = []
        output.append("==== Comprehensive Validation Results ====")
        
        # Overall Status
        overall_status = "✅ Passed" if standard_result.get('metadata', {}).get('is_compliant', False) else "❌ Failed"
        output.append(f"Overall Compliance Status: {overall_status}")
        
        # Rule Validations
        output.append("\n=== Rule Validations ===")
        for rule_name, rule_info in standard_result.get('validation_rules', {}).items():
            status = "✅" if rule_info.get('status') == 'passed' else "❌"
            error = f"\n   Error: {rule_info.get('error_message')}" if rule_info.get('error_message') else ""
            output.append(f"{rule_name}: {status}{error}")
        
        # Directors Validation
        output.append("\n=== Directors Validation ===")
        for director_key, director_info in standard_result.get('document_validation', {}).get('directors', {}).items():
            output.append(f"{director_key}:")
            output.append(f"  Nationality: {director_info.get('nationality', 'Unknown')}")
            output.append(f"  Authorized: {director_info.get('authorized', False)}")
            output.append("  Documents:")
            
            for doc_key, doc_info in director_info.get('documents', {}).items():
                status = "✅" if doc_info.get('status') == 'Valid' else "❌"
                reason = f" ({doc_info.get('reason')})" if doc_info.get('reason') else ""
                output.append(f"    {doc_key}: {status}{reason}")
        
        # Company Documents
        output.append("\n=== Company Documents ===")
        company_docs = standard_result.get('document_validation', {}).get('companyDocuments', {})
        
        # Address Proof
        address_status = "✅" if company_docs.get('addressProof', {}).get('status') == 'Valid' else "❌"
        address_reason = f" ({company_docs.get('addressProof', {}).get('reason')})" if company_docs.get('addressProof', {}).get('reason') else ""
        output.append(f"  Address Proof: {address_status}{address_reason}")
        
        # NOC
        noc_status = "✅" if company_docs.get('noc', {}).get('status') == 'Valid' else "❌"
        noc_reason = f" ({company_docs.get('noc', {}).get('reason')})" if company_docs.get('noc', {}).get('reason') else ""
        output.append(f"  NOC: {noc_status}{noc_reason}")
        
        return "\n".join(output)

    def _get_default_compliance_rules(self) -> Dict:
        """
        Get default compliance rules when Elasticsearch rules are unavailable
        
        Returns:
            dict: Default compliance rules
        """
        default_rules = {
            "rules": [
                {
                    "rule_id": "DIRECTOR_COUNT",
                    "rule_name": "Director Count",
                    "description": "Number of directors must be between 2 and 5",
                    "severity": "high",
                    "is_active": True,
                    "conditions": {
                        "min_directors": 2,
                        "max_directors": 5
                    }
                },
                {
                    "rule_id": "PASSPORT_PHOTO",
                    "rule_name": "Passport Photo",
                    "description": "Passport photo must be clear and properly formatted",
                    "severity": "medium",
                    "is_active": True,
                    "conditions": {
                        "min_clarity_score": 0.7,
                        "is_passport_style": True,
                        "face_visible": True
                    }
                },
                {
                    "rule_id": "SIGNATURE",
                    "rule_name": "Signature",
                    "description": "Signature must be clear and handwritten",
                    "severity": "medium",
                    "is_active": True,
                    "conditions": {
                        "min_clarity_score": 0.7,
                        "is_handwritten": True,
                        "is_complete": True
                    }
                },
                {
                    "rule_id": "ADDRESS_PROOF",
                    "rule_name": "Address Proof",
                    "description": "Address proof must be valid and recent",
                    "severity": "high",
                    "is_active": True,
                    "conditions": {
                        "max_age_days": 45,
                        "name_match_required": True,
                        "complete_address_required": True
                    }
                },
                {
                    "rule_id": "INDIAN_DIRECTOR_PAN",
                    "rule_name": "Indian Director PAN Card",
                    "description": "Indian directors must provide a valid PAN card",
                    "severity": "high",
                    "is_active": True,
                    "conditions": {
                        "min_age": 18
                    }
                },
                {
                    "rule_id": "INDIAN_DIRECTOR_AADHAR",
                    "rule_name": "Indian Director Aadhar Card",
                    "description": "Indian directors must provide valid Aadhar cards",
                    "severity": "high",
                    "is_active": True,
                    "conditions": {
                        "masked_not_allowed": True,
                        "different_images_required": True
                    }
                },
                {
                    "rule_id": "FOREIGN_DIRECTOR_DOCS",
                    "rule_name": "Foreign Director Documents",
                    "description": "Foreign directors must provide valid identification",
                    "severity": "high",
                    "is_active": True,
                    "conditions": {
                        "passport_required": True,
                        "passport_validity_check": True,
                        "driving_license_required": False
                    }
                },
                {
                    "rule_id": "COMPANY_ADDRESS_PROOF",
                    "rule_name": "Company Address Proof",
                    "description": "Company must have valid address proof",
                    "severity": "high",
                    "is_active": True,
                    "conditions": {
                        "max_age_days": 45,
                        "complete_address_required": True,
                        "name_match_required": False
                    }
                },
                {
                    "rule_id": "NOC_VALIDATION",
                    "rule_name": "No Objection Certificate",
                    "description": "NOC from property owner is required",
                    "severity": "medium",
                    "is_active": True,
                    "conditions": {
                        "noc_required": True,
                        "signature_required": True
                    }
                },
                {
                    "rule_id": "AADHAR_PAN_LINKAGE",
                    "rule_name": "Aadhar PAN Linkage",
                    "description": "Aadhar and PAN must be linked for Indian directors",
                    "severity": "high",
                    "is_active": True,
                    "conditions": {
                        "linkage_api_check_required": True
                    }
                }
            ]
        }
        
        self.logger.info(f"Using default compliance rules: {json.dumps(default_rules, indent=2)}")
        return default_rules
    
    # def validate_documents(
    #     self, 
    #     service_id: str, 
    #     request_id: str, 
    #     input_data: Dict[str, Any]
    # ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    #     """
    #     Main document validation method
    #     """
    #     start_time = time.time()
        
    #     try:
    #         # Retrieve compliance rules
    #         compliance_rules = self._get_compliance_rules(service_id)
            
    #         # Validate directors
    #         directors_validation = self._validate_directors(
    #             input_data.get('directors', {}), 
    #             compliance_rules
    #         )
            
    #         # Validate company documents
    #         company_docs_validation = self._validate_company_documents(
    #             input_data.get('companyDocuments', {}),
    #             input_data.get('directors', {}),
    #             compliance_rules
    #         )
            
    #         # Calculate processing time
    #         processing_time = time.time() - start_time
            
    #         # Ensure directors_validation is a dictionary
    #         if isinstance(directors_validation, list):
    #             directors_validation = {str(idx): info for idx, info in enumerate(directors_validation)}
            
    #         # Determine overall compliance
    #         is_compliant = all(
    #             director.get('is_valid', False) 
    #             for director in directors_validation.values() 
    #             if isinstance(director, dict)
    #         )
            
    #         # Prepare standard result with type checking
    #         standard_result = {
    #             "validation_rules": self._prepare_validation_rules(directors_validation, company_docs_validation),
    #             "document_validation": {
    #                 "directors": {
    #                     director_key: {
    #                         "nationality": director_info.get('nationality', 'Unknown'),
    #                         "authorized": director_info.get('is_authorised', False),
    #                         "documents": {
    #                             doc_key: {
    #                                 "status": self._get_document_status(doc_info),
    #                                 "reason": self._get_document_reason(doc_info)
    #                             } for doc_key, doc_info in director_info.get('documents', {}).items()
    #                         }
    #                     } for director_key, director_info in directors_validation.items() 
    #                     if isinstance(director_info, dict)
    #                 },
    #                 "companyDocuments": {
    #                     "addressProof": {
    #                         "status": "Valid" if company_docs_validation.get('is_valid', False) else "Not Valid",
    #                         "reason": company_docs_validation.get('validation_errors', [None])[0]
    #                     },
    #                     "noc": {
    #                         "status": "Valid" if company_docs_validation.get('noc', {}).get('is_valid', False) else "Not Valid",
    #                         "reason": None
    #                     }
    #                 }
    #             }
    #         }
            
    #         # Prepare detailed result
    #         detailed_result = {
    #             "validation_rules": self._prepare_detailed_validation_rules(directors_validation, company_docs_validation),
    #             "document_validation": {
    #                 "directors": directors_validation,
    #                 "companyDocuments": company_docs_validation
    #             },
    #             "metadata": {
    #                 "service_id": service_id,
    #                 "request_id": request_id,
    #                 "timestamp": datetime.now().isoformat(),
    #                 "processing_time": processing_time,
    #                 "is_compliant": is_compliant
    #             }
    #         }
            
    #         return standard_result, detailed_result
            
    #     except Exception as e:
    #         self.logger.error(f"Comprehensive validation error: {str(e)}", exc_info=True)
            
    #         # Prepare error results
    #         error_result = {
    #             "validation_rules": {
    #                 "global_error": {
    #                     "status": "failed",
    #                     "error_message": str(e)
    #                 }
    #             },
    #             "document_validation": {
    #                 "directors": {},
    #                 "companyDocuments": {}
    #             }
    #         }
            
    #         error_detailed_result = {
    #             "validation_rules": {
    #                 "global_error": {
    #                     "status": "failed",
    #                     "error_message": str(e),
    #                     "stacktrace": traceback.format_exc()
    #                 }
    #             },
    #             "document_validation": {
    #                 "directors": {},
    #                 "companyDocuments": {}
    #             },
    #             "metadata": {
    #                 "service_id": service_id,
    #                 "request_id": request_id,
    #                 "timestamp": datetime.now().isoformat(),
    #                 "error": str(e)
    #             }
    #         }
            
    #         return error_result, error_detailed_result
    def _get_expected_documents_for_service(self, service_id: str, compliance_rules: Dict) -> Dict[str, List[str]]:
        """
        Determine which documents are expected based on service ID and rules
        
        Args:
            service_id (str): Service identifier
            compliance_rules (dict): Compliance rules for the service
        
        Returns:
            dict: Dictionary of expected documents by type
        """
        # Common documents expected for all service types
        expected_documents = {
            "director": [
                "aadharCardFront", 
                "aadharCardBack", 
                "panCard", 
                "passportPhoto",
                "signature",
                "address_proof"
            ],
            "company": [
                "addressProof",
                "noc"
            ]
        }
        
        # Add specific documents based on service ID
        if service_id == "4" or service_id == "5":  # GST Rental or GST Own
            # No additional documents required
            pass
        elif service_id == "6":  # GST Family Owned
            expected_documents["company"].append("consent_letter")
        elif service_id == "7":  # GST PVT/LLP
            expected_documents["company"].append("board_resolution")
        
        # Check rules to refine expected documents
        rules = compliance_rules.get('rules', [])
        
        # Check for TENANT_EB_NAME_MATCH rule
        has_tenant_eb_match = any(rule.get('rule_id') == 'TENANT_EB_NAME_MATCH' and rule.get('is_active', True) for rule in rules)
        if has_tenant_eb_match:
            expected_documents["director"].append("electricityBill")
            
        # Check for specific foreign documents
        has_foreign_docs = any(rule.get('rule_id') == 'FOREIGN_DIRECTOR_DOCS' and rule.get('is_active', True) for rule in rules)
        if has_foreign_docs:
            # These would only be required for foreign directors
            expected_documents["foreign_director"] = ["passport", "drivingLicense"]
            
        return expected_documents

    def _identify_missing_documents(self, input_data: Dict[str, Any], expected_documents: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Identify which required documents are missing
        
        Args:
            input_data (dict): Input validation data
            expected_documents (dict): Dictionary of expected documents by type
        
        Returns:
            dict: Dictionary of missing documents by document type
        """
        missing_documents = {}
        
        # Check director documents
        directors = input_data.get('directors', {})
        for director_key, director_info in directors.items():
            # Determine if this is a foreign director
            is_foreign = director_info.get('nationality', '').lower() == 'foreign'
            
            # Get expected documents for this director type
            director_expected_docs = expected_documents.get("director", [])
            
            # If foreign director, add foreign-specific docs
            if is_foreign and "foreign_director" in expected_documents:
                director_expected_docs.extend(expected_documents["foreign_director"])
            
            # Check each expected document
            documents = director_info.get('documents', {})
            for doc_type in director_expected_docs:
                if doc_type not in documents or not documents[doc_type]:
                    # Add to missing documents list
                    if doc_type not in missing_documents:
                        missing_documents[doc_type] = []
                    missing_documents[doc_type].append(director_key)
        
        # Check company documents
        company_docs = input_data.get('companyDocuments', {})
        for doc_type in expected_documents.get("company", []):
            if doc_type not in company_docs or not company_docs[doc_type]:
                # Add to missing documents list with "company" as the director
                if doc_type not in missing_documents:
                    missing_documents[doc_type] = []
                missing_documents[doc_type].append("company")
        
        return missing_documents
    
    def validate_documents(
        self, 
        service_id: str, 
        request_id: str, 
        input_data: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Main document validation method with FORCED service ID rule selection
        
        Args:
            service_id (str): Service identifier
            request_id (str): Unique request identifier
            input_data (Dict[str, Any]): Input validation data
        
        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: Validation results
        """
        start_time = time.time()

        self._current_preconditions = input_data.get('preconditions', {})
        
        # CRITICAL: FORCE the service ID rules
        def force_service_id_rules(rules, target_service_id):
            """
            Forcibly select rules for a specific service ID
            
            Args:
                rules (list): Retrieved rules
                target_service_id (str): Target service ID
            
            Returns:
                dict: Selected rules for the target service
            """
            for rule_set in rules:
                if str(rule_set.get('service_id')) == str(target_service_id):
                    return {
                        "service_id": target_service_id,
                        "service_name": rule_set.get('service_name', f'Service {target_service_id}'),
                        "rules": rule_set.get('rules', [])
                    }
            
            # If no rules found, use default
            default_rules = self._get_default_compliance_rules()
            default_rules['service_id'] = target_service_id
            return default_rules

        try:
            # Retrieve ALL rules from Elasticsearch
            all_rules = self.es_client.get_compliance_rules(service_id)
            
            # FORCE selection of rules for specific service ID
            compliance_rules = force_service_id_rules(all_rules, service_id)
            
            # Log forced rule selection for debugging
            self.logger.info(f"FORCED Rule Selection for Service ID {service_id}: {json.dumps(compliance_rules, indent=2)}")
            if service_id == "8":  # TM Service
                return self._validate_tm_documents(service_id, request_id, input_data, compliance_rules, start_time)
            else:
                # Extract preconditions if available
                preconditions = input_data.get('preconditions', {})

                # Validate directors
                directors_validation = self._validate_directors(
                    input_data.get('directors', {}), 
                    compliance_rules
                )
                
                # Validate company documents
                company_docs_validation = self._validate_company_documents(
                    input_data.get('companyDocuments', {}),
                    input_data.get('directors', {}),
                    compliance_rules,
                    preconditions
                )
                # company_docs_validation = self._process_company_documents(
                #     input_data.get('companyDocuments', {})#,
                #     #input_data.get('directors', {}),
                #     #compliance_rules,
                #     #preconditions
                # )
                # company_docs_extracted = self._process_company_documents(
                #     input_data.get('companyDocuments', {})
                # )

                # company_docs_validation = self._apply_company_rules(
                #     company_docs_extracted,
                #     compliance_rules,
                #     preconditions
                # )

                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Ensure directors_validation is a dictionary
                if isinstance(directors_validation, list):
                    directors_validation = {str(idx): info for idx, info in enumerate(directors_validation)}
                
                # Determine overall compliance
                is_compliant = all(
                    director.get('is_valid', False) 
                    for director in directors_validation.values() 
                    if isinstance(director, dict)
                )
                
                # Prepare standard result
                # standard_result = {
                #     "validation_rules": self._prepare_validation_rules(directors_validation, company_docs_validation, compliance_rules),
                #     "document_validation": {
                #         "directors": directors_validation,
                #         "companyDocuments": company_docs_validation
                #     }
                # }
                standard_result = {
                    "validation_rules": self._prepare_detailed_validation_rules(directors_validation, company_docs_validation, compliance_rules),
                    "document_validation": {
                        "directors": directors_validation,
                        "companyDocuments": company_docs_validation                    }
                    }
                
                
                # Prepare detailed result
                detailed_result = {
                    "validation_rules": self._prepare_detailed_validation_rules(directors_validation, company_docs_validation, compliance_rules),
                    "document_validation": {
                        "directors": directors_validation,
                        "companyDocuments": company_docs_validation
                    },
                    "metadata": {
                        "service_id": service_id,
                        "request_id": request_id,
                        "timestamp": datetime.now().isoformat(),
                        "processing_time": processing_time,
                        "is_compliant": is_compliant
                    }
                }
                
                # Save detailed results to JSON file
                with open('detailed_validation_results.json', 'w') as f:
                    json.dump(detailed_result, f, indent=2)

                # Print simplified output to terminal
                print("\n=== Document Validation Summary ===")
                print(f"Service ID: {service_id}")
                print(f"Request ID: {request_id}")
                print(f"Processing Time: {processing_time:.2f} seconds")
                
                # Print only critical validation errors
                print("\nValidation Status:")
                errors_found = False
                
                # Print director-related errors
                if "global_errors" in directors_validation:
                    for error in directors_validation["global_errors"]:
                        print(f"❌ {error}")
                        errors_found = True
                
                # Print company document errors
                if company_docs_validation.get("validation_errors"):
                    for error in company_docs_validation["validation_errors"]:
                        print(f"❌ {error}")
                        errors_found = True
                #print(company_docs_validation["addressProof"]["error_messages"])
                if not errors_found:
                    print("✅ All validations passed")
                        
                print("\nDetailed results saved to detailed_validation_results.json")
                #print(detailed_result["document_validation"]["companyDocuments"]["addressProof"]["error_messages"])
                return standard_result, detailed_result
            
        except Exception as e:
            self.logger.error(f"Comprehensive validation error: {str(e)}", exc_info=True)
            
            # Prepare error results
            error_result = {
                "validation_rules": {
                    "global_error": {
                        "status": "failed",
                        "error_message": str(e)
                    }
                },
                "document_validation": {
                    "directors": {},
                    "companyDocuments": {}
                }
            }
            
            error_detailed_result = {
                "validation_rules": {
                    "global_error": {
                        "status": "failed",
                        "error_message": str(e),
                        "stacktrace": traceback.format_exc()
                    }
                },
                "document_validation": {
                    "directors": {},
                    "companyDocuments": {}
                },
                "metadata": {
                    "service_id": service_id,
                    "request_id": request_id,
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                }
            }
            
            # Save error results to JSON
            with open('detailed_validation_results.json', 'w') as f:
                json.dump(error_detailed_result, f, indent=2)
                    
            print("\n❌ Error during validation:")
            print(f"Error: {str(e)}")
            print("Check detailed_validation_results.json for more information")
            
            return error_result, error_detailed_result
        
    def _get_document_status(self, doc_info: Dict) -> str:
        """Determine document validation status"""
        if not isinstance(doc_info, dict):
            return "Not Valid"
            
        # Check basic validity
        if not doc_info.get('is_valid', False):
            return "Not Valid"
            
        # Get extracted data
        extracted_data = doc_info.get('extracted_data', {})
        
        # If extraction failed
        if extracted_data.get('extraction_status') == 'failed':
            return "Not Valid"
            
        # Check clarity score
        clarity_score = extracted_data.get('clarity_score', 0)
        if clarity_score > 0.7:
            return "Valid"
            
        return "Not Valid"

    def _get_document_reason(self, doc_info: Dict) -> Optional[str]:
        """Get reason for document validation status"""
        if not isinstance(doc_info, dict):
            return "Invalid document format"
            
        if not doc_info.get('is_valid', False):
            return "Document validation failed"
            
        extracted_data = doc_info.get('extracted_data', {})
        
        if extracted_data.get('extraction_status') == 'failed':
            return extracted_data.get('error_message', 'Extraction failed')
            
        clarity_score = extracted_data.get('clarity_score', 0)
        if clarity_score < 0.7:
            return f"Low clarity score: {clarity_score}"
            
        return None

        
    def _get_validation_reason(self, doc_info):
        """Helper to get validation failure reason"""
        if not isinstance(doc_info, dict):
            return "Invalid document data"
            
        if not doc_info.get('is_valid', False):
            return "Document validation failed"
            
        extracted_data = doc_info.get('extracted_data', {})
        
        if 'extraction_status' in extracted_data:
            return extracted_data.get('error_message', 'Extraction failed')
            
        clarity_score = extracted_data.get('clarity_score', 0)
        if clarity_score < 0.7:
            return f"Low clarity score: {clarity_score}"
            
        return None

    def _prepare_validation_rules(self, directors_validation, company_docs_validation, compliance_rules):
        """
        Prepare validation rules summary dynamically based on compliance rules
        
        Args:
            directors_validation (dict or list): Directors validation data
            company_docs_validation (dict): Company documents validation
            compliance_rules (dict): Compliance rules for the service
                
        Returns:
            dict: Processed validation rules
        """
        validation_rules = {}
        
        # Extract rules from compliance rules
        rules = compliance_rules.get('rules', [])
        
        # Standard rule ID mapping for API response keys
        rule_id_mapping = {
            'director_count': 'director_count',
            'passport_photo': 'passport_photo_validation',
            'signature': 'signature_validation',
            'address_proof': 'address_proof_validation',
            'indian_director_pan': 'pan_validation',
            'indian_director_aadhar': 'aadhar_validation',
            'foreign_director_docs': 'foreign_director_docs_validation',
            'company_address_proof': 'company_address_proof',
            'noc_validation': 'noc_validation',
            'aadhar_pan_linkage': 'aadhar_pan_linkage',
            'noc_owner_validation': 'noc_owner_validation'  # Added this mapping
        }
        
        # Initialize validation results with default values
        validation_defaults = {}
        
        # Dynamically build default validation rules based on the compliance rules
        for rule in rules:
            rule_id = rule.get('rule_id', '').lower()
            api_rule_id = rule_id_mapping.get(rule_id.lower(), rule_id.lower())
            
            validation_defaults[api_rule_id] = {
                "status": "passed",
                "error_message": None
            }
        
        # Get rule validations from directors
        if isinstance(directors_validation, dict):
            # Check for global rule validations from directors
            if 'rule_validations' in directors_validation:
                for rule_id, rule_result in directors_validation['rule_validations'].items():
                    api_rule_id = rule_id_mapping.get(rule_id.lower(), rule_id.lower())
                    validation_defaults[api_rule_id] = {
                        "status": rule_result.get('status', 'failed').lower(),
                        "error_message": rule_result.get('error_message')
                    }
            
            # Check individual directors for rule validations
            for director_key, director_info in directors_validation.items():
                if isinstance(director_info, dict) and director_key not in ['global_errors', 'rule_validations']:
                    rule_validations = director_info.get('rule_validations', {})
                    for rule_id, rule_result in rule_validations.items():
                        api_rule_id = rule_id_mapping.get(rule_id.lower(), rule_id.lower())
                        # validation_defaults[api_rule_id] = {
                        #     "status": rule_result.get('status', 'failed').lower(),
                        #     "error_message": rule_result.get('error_message')
                        # }
                        if isinstance(rule_result, list):
                            failed = [r for r in rule_result if r.get("status") != "passed"]
                            if failed:
                                combined_msg = "; ".join(f"{r['director']}: {r['error_message']}" for r in failed)
                                validation_defaults[api_rule_id] = {
                                    "status": "failed",
                                    "error_message": combined_msg
                                }
                            else:
                                validation_defaults[api_rule_id] = {
                                    "status": "passed",
                                    "error_message": None
                                }
                        elif isinstance(rule_result, dict):
                            validation_defaults[api_rule_id] = {
                                "status": rule_result.get('status', 'failed').lower(),
                                "error_message": rule_result.get('error_message')
                            }

        
        # Get validation errors from company documents
        if isinstance(company_docs_validation, dict):
            # Check for validation errors in company documents
            validation_errors = company_docs_validation.get('validation_errors', [])
            
            # # Company address proof validation
            # if 'company_address_proof' in validation_defaults and validation_errors:
            #     validation_defaults['company_address_proof'] = {
            #         "status": "failed",
            #         "error_message": validation_errors[0] if validation_errors else None
            #     }
            
            # # NOC validation
            # if 'noc_validation' in validation_defaults:
            #     noc_validation = company_docs_validation.get('noc_validation', {})
            #     if noc_validation:
            #         validation_defaults['noc_validation'] = {
            #             "status": noc_validation.get('status', 'failed').lower(),
            #             "error_message": noc_validation.get('error_message')
            #         }
            #     elif validation_errors:
            #         validation_defaults['noc_validation'] = {
            #             "status": "failed",
            #             "error_message": validation_errors[0] if validation_errors else None
            #         }
            
            if 'noc_validation' in validation_defaults:
                noc_data = company_docs_validation.get('noc', {})
                noc_errors = noc_data.get("error_messages", [])
                validation_defaults['noc_validation'] = {
                    "status": "failed" if noc_errors else "passed",
                    "error_message": noc_errors[0] if noc_errors else None
                }

            if 'company_address_proof' in validation_defaults:
                addr_data = company_docs_validation.get('addressProof', {})
                addr_errors = addr_data.get("error_messages", [])
                validation_defaults['company_address_proof'] = {
                    "status": "failed" if addr_errors else "passed",
                    "error_message": addr_errors[0] if addr_errors else None
                }



            # NOC Owner validation
            if 'noc_owner_validation' in validation_defaults:
                noc_owner_validation = company_docs_validation.get('noc_owner_validation', {})
                if noc_owner_validation:
                    validation_defaults['noc_owner_validation'] = {
                        "status": noc_owner_validation.get('status', 'failed').lower(),
                        "error_message": noc_owner_validation.get('error_message')
                    }
                elif validation_errors and any("owner" in error.lower() for error in validation_errors):
                    # Find owner-related error
                    owner_error = next((error for error in validation_errors if "owner" in error.lower()), None)
                    if owner_error:
                        validation_defaults['noc_owner_validation'] = {
                            "status": "failed",
                            "error_message": owner_error
                        }
        
        return validation_defaults


    def _prepare_detailed_validation_rules(self, directors_validation, company_docs_validation, compliance_rules):
        """
        Generate detailed validation rule information with conditions and evaluation

        Args:
            directors_validation (dict): Full validation results of directors
            company_docs_validation (dict): Company documents validation
            compliance_rules (dict): Full compliance rules

        Returns:
            dict: Detailed rule evaluation
        """
        validation_rules = {}
        rules = compliance_rules.get('rules', [])

        rule_result_map = {}

        # Gather all rule validations from directors
        if isinstance(directors_validation, dict):
            for key, value in directors_validation.items():
                if key in ['global_errors', 'rule_validations']:
                    continue
                if isinstance(value, dict):
                    rule_validations = value.get('rule_validations', {})
                    if isinstance(rule_validations, dict):
                        for rule_id, rule_result in rule_validations.items():
                            if rule_id not in rule_result_map:
                                rule_result_map[rule_id] = []
                            rule_result_map[rule_id].append({
                                "director": key,
                                "status": rule_result.get("status", "failed"),
                                "error_message": rule_result.get("error_message")
                            })

            # Global/shared validations
            shared_rules = directors_validation.get("rule_validations", {})
            if isinstance(shared_rules, dict):
                for rule_id, rule_result in shared_rules.items():
                    if rule_id not in rule_result_map:
                        rule_result_map[rule_id] = []
                    rule_result_map[rule_id].append({
                        "director": "all",
                        "status": rule_result.get("status", "failed"),
                        "error_message": rule_result.get("error_message")
                    })

        # Company document validations
        if isinstance(company_docs_validation, dict):
            if "noc_validation" in company_docs_validation:
                rule_result_map["noc_validation"] = [{
                    "director": "company",
                    "status": company_docs_validation["noc_validation"].get("status", "failed"),
                    "error_message": company_docs_validation["noc_validation"].get("error_message")
                }]
            elif "noc" in company_docs_validation:
                noc_errors = company_docs_validation["noc"].get("error_messages", [])
                if noc_errors:
                    rule_result_map["noc_validation"] = [{
                        "director": "company",
                        "status": "failed",
                        "error_message": noc_errors[0]
                    }]
            if "noc_owner_validation" in company_docs_validation:
                rule_result_map["noc_owner_validation"] = [{
                    "director": "company",
                    "status": company_docs_validation["noc_owner_validation"].get("status", "failed"),
                    "error_message": company_docs_validation["noc_owner_validation"].get("error_message")
                }]
            if "addressProof" in company_docs_validation:
                addr_errors = company_docs_validation["addressProof"].get("error_messages", [])
                if addr_errors:
                    rule_result_map["company_address_proof"] = [{
                        "director": "company",
                        "status": "failed",
                        "error_message": addr_errors[0]
                    }]
            if "validation_errors" in company_docs_validation and "company_address_proof" not in rule_result_map:
                errors = company_docs_validation.get("validation_errors", [])
                if any("address proof" in e.lower() for e in errors):
                    rule_result_map["company_address_proof"] = [{
                        "director": "company",
                        "status": "failed",
                        "error_message": errors[0]
                    }]

        # Build final validation summary
        for rule in rules:
            rule_id = rule.get("rule_id", "").lower()
            rule_result = rule_result_map.get(rule_id, [])

            if isinstance(rule_result, list):
                failed = [r for r in rule_result if r.get("status") != "passed"]
                status = "failed" if failed else "passed"
                error_message = "; ".join(
                    f"{r['director']}: {r['error_message']}" for r in failed if r.get("error_message")
                ) or None
                details = rule_result
            elif isinstance(rule_result, dict):  # fallback, rarely used now
                status = rule_result.get("status", "failed")
                error_message = rule_result.get("error_message")
                details = [rule_result]
            else:
                status = "passed"
                error_message = None
                details = []

            validation_rules[rule_id] = {
                "rule_id": rule.get('rule_id'),
                "rule_name": rule.get('rule_name'),
                "description": rule.get('description'),
                "severity": rule.get('severity', 'medium'),
                "is_active": rule.get('is_active', True),
                "conditions": rule.get('conditions', {}),
                "status": status,
                "error_message": error_message,
                "details": details
            }

        return validation_rules
    

    
    
    def _validate_noc_owner_name_rule(self, company_docs_validation, conditions, preconditions=None):
        """
        Validate NOC owner name against provided precondition
        
        Args:
            company_docs_validation (dict): Company document validation data
            conditions (dict): Rule conditions
            preconditions (dict): Input preconditions
        
        Returns:
            dict: Validation result
        """
        # Default to passed if no validation needed
        if not preconditions:
            return {
                "status": "passed",
                "error_message": None
            }
            
        # Check if API check is required
        api_check_required = conditions.get('api_check_required', True)
        if not api_check_required:
            return {
                "status": "passed",
                "error_message": None
            }
        
        # Check if preconditions contain owner name
        expected_owner_name = preconditions.get('owner_name')
        if not expected_owner_name:
            return {
                "status": "passed",
                "error_message": None
            }
        
        # Get NOC document
        noc = company_docs_validation.get('noc', {})
        
        # If no NOC document found
        if not noc:
            return {
                "status": "failed",
                "error_message": "No Objection Certificate (NOC) is required"
            }
        
        # Get extracted NOC data
        extracted_data = noc.get('extracted_data', {})
        
        # Get actual owner name from NOC
        actual_owner_name = extracted_data.get('owner_name')
        
        # Validate name matching
        if not actual_owner_name:
            return {
                "status": "failed",
                "error_message": "Could not extract owner name from NOC"
            }
        
        # Normalize names for comparison
        def normalize_name(name):
            # Convert to lowercase, remove punctuation
            import re
            return re.sub(r'[^\w\s]', '', name.lower()).strip()
        
        # Compare normalized names
        if normalize_name(expected_owner_name) != normalize_name(actual_owner_name):
            return {
                "status": "failed",
                "error_message": f"NOC owner name '{actual_owner_name}' does not match expected name '{expected_owner_name}'"
            }
        
        # Names match
        return {
            "status": "passed",
            "error_message": None
        }
    def _map_rule_id_to_doc_key(self, rule_id: str) -> str:
        mapping = {
            "ADDRESS_PROOF": "address_proof",
            "PASSPORT_PHOTO": "passportPhoto",
            "SIGNATURE": "signature",
            "INDIAN_DIRECTOR_PAN": "panCard",
            "INDIAN_DIRECTOR_AADHAR": "aadharCardFront",
            "AADHAR_PAN_LINKAGE": "aadharCardFront",  # optional mapping
            "FOREIGN_DIRECTOR_DOCS": "passport"       # or drivinglicense depending on extraction
        }
        return mapping.get(rule_id.upper())

    def _validate_directors(self, directors: Dict, compliance_rules: Dict) -> Dict:
        if not isinstance(directors, dict):
            error_msg = f"Invalid directors input. Expected dict, got {type(directors)}"
            self.logger.error(error_msg)
            return {
                "validation_error": error_msg,
                "global_errors": [error_msg],
                "director_errors": {},
                "raw_input": str(directors)
            }

        rules = self._extract_rules_from_compliance_data(compliance_rules)
        validation_results = {}
        global_errors = []
        rule_validations = {}

        # Director count rule
        director_count_rule = next((r for r in rules if r.get('rule_id') == 'DIRECTOR_COUNT'), None)
        if director_count_rule:
            min_directors = director_count_rule.get('conditions', {}).get('min_directors', 2)
            max_directors = director_count_rule.get('conditions', {}).get('max_directors', 5)
            count = len(directors)
            if count < min_directors:
                msg = f"Insufficient directors. Found {count}, minimum required is {min_directors}."
                global_errors.append(msg)
                rule_validations['director_count'] = {"status": "failed", "error_message": msg}
            elif count > max_directors:
                msg = f"Too many directors. Found {count}, maximum allowed is {max_directors}."
                global_errors.append(msg)
                rule_validations['director_count'] = {"status": "failed", "error_message": msg}
            else:
                rule_validations['director_count'] = {"status": "passed", "error_message": None}

        # Preprocess docs in parallel
        processed_directors = {}
        with ThreadPoolExecutor(max_workers=min(5, len(directors))) as executor:
            futures = {
                executor.submit(self._process_director_documents_parallel, info.get('documents', {})): key
                for key, info in directors.items()
            }
            for future in as_completed(futures):
                key = futures[future]
                try:
                    processed_directors[key] = future.result()
                except Exception as e:
                    self.logger.error(f"Doc processing error for {key}: {e}", exc_info=True)
                    processed_directors[key] = {}

        full_director_data = {
            key: {
                **info,
                "documents": processed_directors.get(key, {})
            }
            for key, info in directors.items()
        }

        nationality_map = {
            "indian": ["INDIAN_DIRECTOR_PAN", "INDIAN_DIRECTOR_AADHAR", "AADHAR_PAN_LINKAGE"],
            "foreign": ["FOREIGN_DIRECTOR_DOCS"]
        }
        common_rules = ["PASSPORT_PHOTO", "SIGNATURE", "ADDRESS_PROOF"]

        rule_method_map = {
            "INDIAN_DIRECTOR_PAN": self._validate_indian_pan_rule,
            "INDIAN_DIRECTOR_AADHAR": self._validate_indian_aadhar_rule,
            "AADHAR_PAN_LINKAGE": self._validate_aadhar_pan_linkage_rule,
            "FOREIGN_DIRECTOR_DOCS": self._validate_foreign_director_rule,
            "PASSPORT_PHOTO": self._validate_passport_photo_rule,
            "SIGNATURE": self._validate_signature_rule,
            "ADDRESS_PROOF": self._validate_address_proof_rule
        }

        all_director_keys = list(full_director_data.keys())

        for rule_id, method in rule_method_map.items():
            applicable_keys = []
            if rule_id in nationality_map['indian']:
                applicable_keys = [k for k, v in full_director_data.items() if v.get("nationality", "").lower() == "indian"]
            elif rule_id in nationality_map['foreign']:
                applicable_keys = [k for k, v in full_director_data.items() if v.get("nationality", "").lower() == "foreign"]
            elif rule_id in common_rules:
                applicable_keys = all_director_keys

            applicable_data = {k: full_director_data[k] for k in applicable_keys}
            rule_conditions = next((r.get('conditions', {}) for r in rules if r.get('rule_id') == rule_id), {})

            try:
                result = method(applicable_data, rule_conditions)
                if result["status"] == "failed" and isinstance(result.get("details"), list):
                    for failure in result["details"]:
                        director_key = failure["director"]
                        if director_key not in validation_results:
                            validation_results[director_key] = {
                                "nationality": full_director_data[director_key].get("nationality", "Unknown"),
                                "is_authorised": full_director_data[director_key].get("authorised", "No") == "Yes",
                                "documents": full_director_data[director_key].get("documents", {}),
                                "validation_errors": [],
                                "rule_validations": {}
                            }
                        validation_results[director_key]["rule_validations"][rule_id.lower()] = {
                            "status": "failed",
                            "error_message": failure.get("error_message")
                        }
                        # # Sync document-level status for streamlit UI display
                        # if rule_id == "ADDRESS_PROOF":
                        #     doc_section = validation_results[director_key].get("documents", {})
                        #     if "address_proof" in doc_section:
                        #         doc_section["address_proof"]["status"] = "Failed"

                        validation_results[director_key]["validation_errors"].append(failure.get("error_message"))
                     


                elif result["status"] == "passed":
                    for k in applicable_keys:
                        if k not in validation_results:
                            validation_results[k] = {
                                "nationality": full_director_data[k].get("nationality", "Unknown"),
                                "is_authorised": full_director_data[k].get("authorised", "No") == "Yes",
                                "documents": full_director_data[k].get("documents", {}),
                                "validation_errors": [],
                                "rule_validations": {}
                            }
                        validation_results[k]["rule_validations"][rule_id.lower()] = {
                            "status": "passed",
                            "error_message": None
                        }
                        # if rule_id == "ADDRESS_PROOF":
                        #     doc_section = validation_results[k].get("documents", {})
                        #     if "address_proof" in doc_section:
                        #         doc_section["address_proof"]["status"] = "Valid"


            except Exception as e:
                self.logger.error(f"Error applying rule {rule_id}: {e}", exc_info=True)
                for k in applicable_keys:
                    validation_results.setdefault(k, {"rule_validations": {}, "validation_errors": []})
                    validation_results[k]["rule_validations"][rule_id.lower()] = {
                        "status": "failed",
                        "error_message": str(e)
                    }
                    validation_results[k]["validation_errors"].append(str(e))

        for key in validation_results:
            errors = validation_results[key].get("validation_errors", [])
            validation_results[key]["is_valid"] = len(errors) == 0

        if global_errors:
            validation_results['global_errors'] = global_errors
        if rule_validations:
            validation_results['rule_validations'] = rule_validations
        print("----------------------------------------------")
        print(f"Validation results: {validation_results}")
        print("----------------------------------------------")
        return validation_results


    def _validate_single_director(
        self, 
        director_key: str, 
        director_info: Dict[str, Any], 
        rules: List
    ) -> Dict:
        """
        Comprehensive validation for a single director
        
        Args:
            director_key (str): Director identifier
            director_info (dict): Director information
            rules (list): Validation rules
        
        Returns:
            dict: Detailed validation results
        """
        # Validate basic structure
        validation_errors = []
        required_keys = ['nationality', 'authorised', 'documents']
        for key in required_keys:
            if key not in director_info:
                validation_errors.append(f"Missing required key: {key}")
        
        # Get nationality and documents
        nationality = director_info.get('nationality', '').lower()
        documents = director_info.get('documents', {})
        
        # Prepare full document validation with extraction results in parallel
        full_documents = self._process_director_documents_parallel(documents)
        
        # Specific nationality-based rules mapping
        nationality_rules = {
            'indian': [
                'INDIAN_DIRECTOR_PAN', 
                'INDIAN_DIRECTOR_AADHAR', 
                'AADHAR_PAN_LINKAGE'
            ],
            'foreign': ['FOREIGN_DIRECTOR_DOCS']
        }
        
        # Common rules for all directors
        common_rules = ['PASSPORT_PHOTO', 'SIGNATURE', 'ADDRESS_PROOF']
        
        # Get applicable rules based on nationality
        applicable_rules = nationality_rules.get(nationality, []) + common_rules
        
        # Rule processing map
        rule_processing_map = {
            "INDIAN_DIRECTOR_PAN": self._validate_indian_pan_rule,
            "INDIAN_DIRECTOR_AADHAR": self._validate_indian_aadhar_rule,
            "FOREIGN_DIRECTOR_DOCS": self._validate_foreign_director_rule,
            "AADHAR_PAN_LINKAGE": self._validate_aadhar_pan_linkage_rule,
            "PASSPORT_PHOTO": self._validate_passport_photo_rule,
            "SIGNATURE": self._validate_signature_rule,
            "ADDRESS_PROOF": self._validate_address_proof_rule
        }
        
        # Storage for rule validations
        rule_validations = {}
        
        # Apply each relevant rule
        for rule_id in applicable_rules:
            try:
                # Get rule conditions
                rule_conditions = next(
                    (rule.get('conditions', {}) for rule in rules if rule.get('rule_id') == rule_id), 
                    {}
                )
                
                # Prepare director data with full documents
                director_validation_data = {
                    director_key: {
                        **director_info,
                        'documents': full_documents
                    }
                }
                
                # Apply validation method
                validation_method = rule_processing_map.get(rule_id)
                if validation_method:
                    result = validation_method(
                        director_validation_data, 
                        rule_conditions
                    )
                    
                    # Store the rule validation result
                    #rule_validations[rule_id.lower()] = result
                    if rule_id.lower() not in rule_validations:
                        rule_validations[rule_id.lower()] = []

                    rule_validations[rule_id.lower()].append({
                        "director": director_key,
                        "status": result.get("status"),
                        "error_message": result.get("error_message")
                    })

                    # Collect errors if validation fails
                    if result.get('status') != 'passed':
                        validation_errors.append(
                            result.get('error_message', f'Validation failed for {rule_id}')
                        )
            
            except Exception as e:
                self.logger.error(f"Rule validation error for {rule_id}: {str(e)}", exc_info=True)
                validation_errors.append(f"Error in {rule_id} validation: {str(e)}")
                rule_validations[rule_id.lower()] = {
                    "status": "failed",
                    "error_message": str(e)
                }
        
        # Determine overall validation status
        is_valid = len(validation_errors) == 0
        
        # Create comprehensive director validation result
        return {
            'nationality': director_info.get('nationality', 'Unknown'),
            'is_authorised': director_info.get('authorised', 'No') == 'Yes',
            'is_valid': is_valid,
            'validation_errors': validation_errors,
            'documents': full_documents,
            'rule_validations': rule_validations
        }
    
    def _process_director_documents_parallel(
        self, 
        documents: Dict[str, str]
    ) -> Dict[str, Dict[str, Any]]:
        processed_docs = {}

        if not documents:
            return processed_docs

        futures = {}
        with ThreadPoolExecutor(max_workers=min(len(documents), 10)) as executor:
            for doc_key, doc_content in documents.items():
                if isinstance(doc_content, str) and doc_content:
                    future = executor.submit(
                        self._extract_document_data_safe,
                        doc_key,
                        doc_content
                    )
                    futures[future] = doc_key

        for future in as_completed(futures):
            doc_key = futures[future]
            try:
                result = future.result()
                processed_docs[doc_key] = result
            except Exception as e:
                self.logger.error(f"Error processing document {doc_key}: {str(e)}", exc_info=True)
                processed_docs[doc_key] = {
                    "is_valid": False,
                    "error": str(e)
                }

        return processed_docs

    def _process_company_documents(self, company_docs: Dict[str, str]) -> Dict[str, Any]:
        processed_docs = {}
        
        for doc_key, doc_content in company_docs.items():
            try:
                # Save base64 string to temp file (if not URL)
                if isinstance(doc_content, str):
                    if doc_content.startswith("http://") or doc_content.startswith("https://"):
                        source = doc_content
                    else:
                        # Determine file extension
                        file_ext = "pdf" if "JVBER" in doc_content[:20] else "jpg"
                        decoded = base64.b64decode(doc_content)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
                            tmp_file.write(decoded)
                            source = tmp_file.name

                    # Extract data
                    result = self.extraction_service.extract_document_data(source, doc_key)
                    processed_docs[doc_key] = result

            except Exception as e:
                self.logger.error(f"Error processing company document {doc_key}: {e}")
                processed_docs[doc_key] = {
                    "is_valid": False,
                    "error": str(e)
                }
        
        return processed_docs


    def _apply_company_rules(
        self,
        extracted_docs: Dict[str, Any],
        compliance_rules: Dict,
        preconditions: Dict = None
    ) -> Dict[str, Any]:
        validation_result = extracted_docs.copy()
        validation_errors = []
        
        rules = self._extract_rules_from_compliance_data(compliance_rules)
        
        # Apply COMPANY_ADDRESS_PROOF rule
        if "addressProof" in extracted_docs:
            address_rule = next(
                (rule for rule in rules if rule.get('rule_id') == 'COMPANY_ADDRESS_PROOF'),
                None
            )
            if address_rule:
                result = self._validate_company_address_proof_rule(
                    extracted_docs,
                    address_rule.get('conditions', {})
                )
                if result["status"] != "passed":
                    validation_errors.append(result["error_message"])
                    validation_result["addressProof"]["is_valid"] = False
                else:
                    validation_result["addressProof"]["is_valid"] = True
        
        # Optionally handle NOC or other company docs here too...

        if validation_errors:
            validation_result["validation_errors"] = validation_errors
            validation_result["is_valid"] = False
        else:
            validation_result["is_valid"] = True

        return validation_result


    def _extract_document_data_safe(
        self, 
        doc_key: str, 
        doc_content: str  # either base64 string or URL
    ) -> Dict[str, Any]:
        """
        Thread-safe method to extract document data from base64 or URL
        
        Args:
            doc_key (str): Document key
            doc_content (str): base64-encoded file or URL
        
        Returns:
            dict: Document validation result
        """
        try:
            doc_type = self._get_document_type(doc_key)

            # Detect base64 string (naive but works well)
            if doc_content.startswith("http://") or doc_content.startswith("https://"):
                input_source = doc_content
            else:
                # Save base64 to temp file
                file_ext = "pdf" if "JVBER" in doc_content[:20] else "jpg"
                decoded = base64.b64decode(doc_content)
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
                    tmp_file.write(decoded)
                    input_source = tmp_file.name
            if doc_type == "passport_photo":
                extracted_data = self.extraction_service.extract_document_data(input_source, doc_type)
                            # Check if extraction failed or returned invalid data
                should_use_fallback = (
                    not isinstance(extracted_data, dict) or
                    extracted_data.get('extraction_status') == 'failed' or
                    "clarity_score" not in extracted_data or
                    extracted_data.get("error") or
                    # Check for the specific error messages from your logs
                    isinstance(extracted_data, str) and "unable to analyze" in extracted_data.lower()
                )
                
                if should_use_fallback:
                    self.logger.warning(f"Using OpenCV fallback for passport photo. Original result: {extracted_data}")
                    opencv_result = self.extraction_service.assess_passport_photo_opencv(input_source, doc_type)
                    
                    # Mark as OpenCV-based result
                    opencv_result["extraction_method"] = "opencv_fallback"
                    extracted_data = opencv_result
                else:
                    extracted_data["extraction_method"] = "primary_extraction"
                # Step 2: Fallback to OpenCV if result is empty or lacks clarity_score
                # if not isinstance(extracted_data, dict) or "clarity_score" not in extracted_data:
                #     self.logger.warning("Fallback to OpenCV for passport photo due to missing clarity_score")
                #     extracted_data = self.extraction_service.assess_passport_photo_opencv(input_source, doc_type)
            else:
                # Extract data using the extraction service
             
                extracted_data = self.extraction_service.extract_document_data(
                    input_source, doc_type
                )
            #  # ⚠️ Fallback for passport_photo if extraction gives no usable result
            # if doc_type == "passport_photo":
            #     if not isinstance(extracted_data, dict) or "clarity_score" not in extracted_data:
            #         self.logger.warning("Fallback to OpenCV for passport photo due to missing clarity_score")
            #         extracted_data = self.extraction_service.assess_passport_photo_opencv(input_source)


            return {
                "source": input_source,
                "document_type": doc_type,
                "is_valid": extracted_data is not None and not (
                    isinstance(extracted_data, dict) and extracted_data.get('extraction_status') == 'failed'
                ),
                "extracted_data": extracted_data or {}
            }

        except Exception as e:
            self.logger.error(f"Document extraction error for {doc_key}: {str(e)}", exc_info=True)
            return {
                "document_type": self._get_document_type(doc_key),
                "is_valid": False,
                "error": str(e)
            }


    def _get_document_type(self, doc_key: str) -> str:
        """
        Map document key to standard document type
        
        Args:
            doc_key (str): Document key from input
        
        Returns:
            str: Standardized document type
        """
        doc_type_mapping = {
            'aadharCardFront': 'aadhar_front',
            'aadharCardBack': 'aadhar_back',
            'panCard': 'pan',
            'passportPhoto': 'passport_photo',
            'passport': 'passport',
            'address_proof': 'address_proof',
            'signature': 'signature',
            'drivingLicense': 'driving_license',
            # New document types
            'consent_letter': 'consent_letter',
            'board_resolution': 'board_resolution',
            'electricityBill': 'electricity_bill'
        }
        
        return doc_type_mapping.get(doc_key, 'unknown')

    def _get_rule_conditions(self, rules: List, rule_id: str) -> Dict:
        """
        Extract conditions for a specific rule
        
        Args:
            rules (list): List of rules
            rule_id (str): Rule identifier
        
        Returns:
            dict: Rule conditions
        """
        for rule in rules:
            if rule.get('rule_id', '').upper() == rule_id.upper():
                return rule.get('conditions', {})
        return {}
    
    def _extract_rules_from_compliance_data(self, compliance_rules: Dict) -> List:
        """
        Extract rules from compliance data, handling various structures
        
        Args:
            compliance_rules (dict): Compliance rules data
        
        Returns:
            list: Extracted rules list
        """
        try:
            # Get the rules list
            rules = compliance_rules.get('rules', [])
            
            # If rules is a list of dicts and the first item has a 'rules' key
            if (isinstance(rules, list) and len(rules) > 0 and 
                isinstance(rules[0], dict) and 'rules' in rules[0]):
                return rules[0].get('rules', [])
            
            return rules
        
        except Exception as e:
            self.logger.error(f"Error extracting rules: {str(e)}", exc_info=True)
            return []
    
    def _process_director(self, director_key: str, director_info: Dict) -> Dict:
        """
        Process and validate a single director
        
        Args:
            director_key (str): Director identifier
            director_info (dict): Director information
        
        Returns:
            dict: Processed director information
        """
        # Validate director info structure
        if not isinstance(director_info, dict):
            raise DocumentValidationError(f"Invalid director information structure for {director_key}")
        
        # Extract basic director information
        nationality = director_info.get('nationality', 'Unknown')
        is_authorised = director_info.get('authorised', 'No') == 'Yes'
        
        # Process documents
        documents = director_info.get('documents', {})
        processed_documents = {}
        
        # Validate and extract data from each document
        for doc_key, doc_url in documents.items():
            if isinstance(doc_url, str) and doc_url:
                # Get document type
                doc_type = self._get_document_type(doc_key)
                
                try:
                    # Extract data from document
                    extracted_data = self.extraction_service.extract_document_data(
                        doc_url, 
                        doc_type
                    )
                    
                    # Prepare document validation result
                    doc_validation = {
                        "url": doc_url,
                        "document_type": doc_type,
                        "is_valid": extracted_data is not None and not ('extraction_status' in extracted_data and extracted_data['extraction_status'] == 'failed'),
                        "extracted_data": extracted_data or {}
                    }
                    
                    # Add optional fields from extraction if available
                    if extracted_data:
                        if 'clarity_score' in extracted_data:
                            doc_validation['clarity_score'] = extracted_data['clarity_score']
                        
                        if 'is_masked' in extracted_data:
                            doc_validation['is_masked'] = extracted_data['is_masked']
                        
                        if 'is_recent' in extracted_data:
                            doc_validation['is_recent'] = extracted_data['is_recent']
                    
                    processed_documents[doc_key] = doc_validation
                
                except Exception as e:
                    self.logger.error(f"Error processing document {doc_key} for {director_key}: {str(e)}", exc_info=True)
                    processed_documents[doc_key] = {
                        "url": doc_url,
                        "document_type": doc_type,
                        "is_valid": False,
                        "error": str(e)
                    }
            else:
                self.logger.warning(f"Invalid document URL for {doc_key} in {director_key}")
                processed_documents[doc_key] = {
                    "error": "Invalid or missing document URL",
                    "is_valid": False
                }
        
        # Return processed director information
        return {
            "nationality": nationality,
            "is_authorised": is_authorised,
            "documents": processed_documents
        }
    
    def _get_document_type(self, doc_key: str) -> str:
        """
        Determine document type from document key
        
        Args:
            doc_key (str): Document key
        
        Returns:
            str: Document type
        """
        doc_type_mapping = {
            'aadharCardFront': 'aadhar_front',
            'aadharCardBack': 'aadhar_back',
            'panCard': 'pan',
            'passport': 'passport',
            'passportPhoto': 'passport_photo',
            'address_proof': 'address_proof',
            'signature': 'signature',
            'drivingLicense': 'driving_license'
        }
        
        return doc_type_mapping.get(doc_key, 'unknown')

    def _save_base64_to_tempfile(self, base64_str: str, doc_type_hint: str = "pdf") -> str:
        """
        Convert base64 string to temp file and return file path.
        
        Args:
            base64_str (str): Base64 encoded string.
            doc_type_hint (str): 'pdf' or 'jpg'
        
        Returns:
            str: Temporary file path
        """
        try:
            decoded = base64.b64decode(base64_str)
            suffix = ".pdf" if "JVBER" in base64_str[:20] or doc_type_hint == "pdf" else ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(decoded)
                return temp_file.name
        except Exception as e:
            self.logger.error(f"Failed to convert base64 to temp file: {e}")
            raise

    def _validate_company_documents(
        self, 
        company_docs: Dict[str, Any],
        directors: Dict,
        compliance_rules: Dict,
        preconditions: Dict = None
    ) -> Dict[str, Any]:
        """
        Validate company-level documents
        
        Args:
            company_docs (dict): Company document information
            directors (dict): Director information
            compliance_rules (dict): Compliance rules
            preconditions (dict, optional): Additional validation preconditions
        
        Returns:
            dict: Validation results for company documents
        """
        try:
            # Extract rules
            rules = self._extract_rules_from_compliance_data(compliance_rules)
            
            validation_result = {}
            # validation_errors = []
            # doc_specific_errors = {
            #     "addressProof": [],
            #     "noc": [],
            # }
            addprf_error = []
            noc_error = []
            # Use ThreadPoolExecutor for parallel processing of company documents
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit address proof task
                address_proof_future = None
                address_proof_input = company_docs.get('addressProof')
                if not address_proof_input:
                    addprf_error.append("Company Address Proof not uploaded")
                    validation_result["addressProof"] = {
                        "url": None,
                        "is_valid": False,
                        "status": "Failed",
                        "error_messages": addprf_error
                    }
                else:
                    try:
                        if address_proof_input.startswith("http://") or address_proof_input.startswith("https://"):
                            source = address_proof_input
                        else:
                            source = self._save_base64_to_tempfile(address_proof_input, "pdf")
                        address_proof_future = executor.submit(
                            self.extraction_service.extract_document_data,
                            source,
                            'address_proof'
                        )
                    except Exception as e:
                        self.logger.error(f"Failed to process addressProof input: {e}")
                        addprf_error.append(f"Failed to process address proof: {str(e)}")

                # if 'addressProof' in company_docs:
                #     address_proof_input = company_docs.get('addressProof')
                #     if address_proof_input:
                #         try:
                #             if address_proof_input.startswith("http://") or address_proof_input.startswith("https://"):
                #                 source = address_proof_input
                #             else:
                #                 source = self._save_base64_to_tempfile(address_proof_input, "pdf")
                            
                #             address_proof_future = executor.submit(
                #                 self.extraction_service.extract_document_data,
                #                 source,
                #                 'address_proof'
                #             )
                #         except Exception as e:
                #             self.logger.error(f"Failed to process addressProof input: {e}")
                #             addprf_error.append(f"Failed to process address proof: {str(e)}")

                # address_proof_future = None
                # if 'addressProof' in company_docs:
                #     address_proof_url = company_docs.get('addressProof')
                #     if address_proof_url:
                #         address_proof_future = executor.submit(
                #             self.extraction_service.extract_document_data,
                #             address_proof_url,
                #             'address_proof'
                #         )
                
                # Submit NOC task
                # noc_future = None
                # if 'noc' in company_docs:
                #     noc_url = company_docs.get('noc')
                #     if noc_url:
                #         noc_future = executor.submit(
                #             self.extraction_service.extract_document_data,
                #             noc_url,
                #             'noc'
                #         )

                # Submit noc task
                noc_future = None
                noc_input = company_docs.get('noc')
                if not noc_input:
                    noc_error.append("NOC not uploaded")
                    validation_result["noc"] = {
                        "source": None,
                        "is_valid": False,
                        "status": "Failed",
                        "error_messages": noc_error
                    }
                else:
                    try:
                        if noc_input.startswith("http://") or noc_input.startswith("https://"):
                            source = noc_input
                        else:
                            source = self._save_base64_to_tempfile(noc_input, "pdf")
                        noc_future = executor.submit(
                            self.extraction_service.extract_document_data,
                            source,
                            'noc'
                        )
                    except Exception as e:
                        self.logger.error(f"Failed to process noc input: {e}")
                        noc_error.append(f"Failed to process NOC: {str(e)}")

                # if 'noc' in company_docs:
                #     noc_input = company_docs.get('noc')
                #     if noc_input:
                #         try:
                #             if noc_input.startswith("http://") or noc_input.startswith("https://"):
                #                 source = noc_input
                #             else:
                #                 source = self._save_base64_to_tempfile(noc_input, "pdf")
                            
                #             noc_future = executor.submit(
                #                 self.extraction_service.extract_document_data,
                #                 source,
                #                 'noc'
                #             )
                #         except Exception as e:
                #             self.logger.error(f"Failed to process noc input: {e}")
                #             noc_error.append(f"Failed to process NOC: {str(e)}")

                
                # Process address proof result
                if address_proof_future:
                    try:
                        address_proof_data = address_proof_future.result()
                        
                        # Get clarity score
                        clarity_score = 0.0
                        if address_proof_data and "clarity_score" in address_proof_data:
                            clarity_score = float(address_proof_data.get("clarity_score", 0.0))
                        
                        # Check for complete address
                        complete_address = False
                        if address_proof_data:
                            complete_address = address_proof_data.get("complete_address_visible", False)
                        
                        validation_result["addressProof"] = {
                            "url": company_docs.get('addressProof'),
                            "is_valid": address_proof_data is not None,
                            "clarity_score": clarity_score,
                            "complete_address_visible": complete_address,
                            "extracted_data": address_proof_data,
                            "status": "Not Valid",
                            "error_messages": [],
                        }
                        
                        # Validate age if extracted data available
                        if address_proof_data:
                            # Get company address proof rule
                            # Step 1: Validate required fields
                            required_fields = ['address', 'date']
                            missing_fields = [field for field in required_fields if not address_proof_data.get(field)]

                            if missing_fields:
                                addprf_error.append(f"Address proof missing required fields: {', '.join(missing_fields)}")

                            company_address_rule = next(
                                (rule for rule in rules if rule.get('rule_id') == 'COMPANY_ADDRESS_PROOF'), 
                                None
                            )
                            
                            if company_address_rule:
                                # Get conditions
                                conditions = company_address_rule.get('conditions', {})
                                max_age_days = conditions.get('max_age_days', 45)
                                
                                # Check address proof age
                                date_str = address_proof_data.get("date") or address_proof_data.get("bill_date")
                                if date_str:
                                    try:
                                        doc_date = self._parse_date(date_str)
                                        if doc_date:
                                            today = datetime.now()
                                            doc_age = (today - doc_date).days
                                            # Use max_age_days from rule conditions
                                            if doc_age > max_age_days:
                                                addprf_error.append(f"Address proof is {doc_age} days old (exceeds {max_age_days} days limit)")
                                    except Exception as e:
                                        self.logger.error(f"Error calculating document age: {e}")
                                        addprf_error.append(f"Error validating address proof date: {str(e)}")
                        else:
                            addprf_error.append("Address proof data extraction failed")
                        # validation_result["addressProof"]["is_valid"] = not any(
                        #     err.lower().startswith("address proof") for err in doc_specific_errors["addressProof"]
                        # )
                        validation_result["addressProof"]["is_valid"] = len(addprf_error) == 0

                    except Exception as e:
                        self.logger.error(f"Error processing address proof: {str(e)}", exc_info=True)
                        validation_result["addressProof"] = {
                            "url": company_docs.get('addressProof'),
                            "is_valid": False,
                            "error": str(e)
                        }
                        addprf_error.append(f"Address proof error: {str(e)}")
                
                # Process NOC result
                if noc_future:
                    try:
                        noc_data = noc_future.result()
                        
                        
                        if not isinstance(noc_data, dict) or not noc_data.get("has_signature", False):
                            noc_error.append("NOC verification failed or invalid format")
                            validation_result["noc"] = {
                                "source": company_docs.get('noc'),
                                "is_valid": False,
                                "has_signature": False,
                                "extracted_data": noc_data,
                                                               "status": "Failed",
                                "error_messages": noc_error
                            }
                        else:
                            validation_result["noc"] = {
                                "source": company_docs.get('noc'),
                                "has_signature": noc_data.get('has_signature', True) if noc_data else False,
                                "extracted_data": noc_data
                            }
                            is_noc_valid = True
                            
                            # Validate NOC Owner Name if preconditions are provided
                            if noc_data and preconditions and 'owner_name' in preconditions:
                                noc_owner_rule = next(
                                    (rule for rule in rules if rule.get('rule_id') == 'NOC_OWNER_VALIDATION'), 
                                    None
                                )
                                
                                if noc_owner_rule:
                                    expected_owner_name = preconditions.get('owner_name')
                                    actual_owner_name = noc_data.get('owner_name')
                                    
                                    # Store the validation result for NOC owner
                                    noc_owner_validation = self._validate_noc_owner_name(
                                        actual_owner_name, 
                                        expected_owner_name
                                    )
                                    
                                    # Store the validation in result
                                    validation_result["noc_owner_validation"] = noc_owner_validation
                                    
                                    # Add to validation errors if failed
                                    if noc_owner_validation["status"] != "passed":
                                        noc_error.append(noc_owner_validation["error_message"])
                            if not noc_data or not validation_result["noc"]["has_signature"]:
                                noc_error.append("NOC does not contain a valid signature")
                                is_noc_valid = False

                        # validation_result["noc"]["is_valid"] = is_noc_valid
                        # validation_result["noc"]["is_valid"] = not any(
                        #     err.lower().startswith("noc") for err in doc_specific_errors["noc"]
                        # )
                        validation_result["noc"]["is_valid"] = len(noc_error) == 0


                    
                    except Exception as e:
                        self.logger.error(f"Error processing NOC: {str(e)}", exc_info=True)
                        validation_result["noc"] = {
                            "url": company_docs.get('noc'),
                            "is_valid": False,
                            "error": str(e)
                        }
                        noc_error.append(f"NOC error: {str(e)}")
            
            # Add validation errors
            # if validation_errors:
            #     validation_result["validation_errors"] = validation_errors
            #     validation_result["is_valid"] = False
            # else:
            #     validation_result["is_valid"] = True
            # Convert internal result into display-friendly format
            # for doc_key in ["addressProof", "noc"]:
            #     if doc_key in validation_result:
            #         doc_entry = validation_result[doc_key]
            #         errors = []

            #         # Append global validation_errors if they relate to this doc
            #         # if "validation_errors" in validation_result:
            #         #     for err in validation_result["validation_errors"]:
            #         #         if doc_key.lower() in err.lower():
            #         #             errors.append(err)
            #         for err in validation_result.get("validation_errors", []):
            #             if doc_key == "addressProof" and err.lower().startswith("address proof"):
            #                 errors.append(err)
            #             elif doc_key == "noc" and err.lower().startswith("noc"):
            #                 errors.append(err)

            #         doc_entry["status"] = "Valid" if doc_entry.get("is_valid") and not errors else "Failed"
            #         doc_entry["error_messages"] = errors
            # Format statuses only for keys that exist

            # for doc_key in validation_result.keys():
            #     if doc_key not in ["addressProof", "noc"]:
            #         continue  # skip any extra keys like validation_errors, etc.
                
            #     doc_entry = validation_result[doc_key]
            #     errors = []

            #     for err in validation_result.get("validation_errors", []):
            #         if doc_key == "addressProof" and err.lower().startswith("address proof"):
            #             errors.append(err)
            #         elif doc_key == "noc" and err.lower().startswith("noc"):
            #             errors.append(err)

            #     doc_entry["status"] = "Valid" if doc_entry.get("is_valid") and not errors else "Failed"
            #     doc_entry["error_messages"] = errors
            # Assign errors per document
            # for doc_key in doc_specific_errors:
            #     if doc_key in validation_result:
            #         errors = doc_specific_errors.get(doc_key, [])
            #         validation_result[doc_key]["error_messages"] = errors
            #         validation_result[doc_key]["status"] = "Valid" if validation_result[doc_key].get("is_valid") and not errors else "Failed"

            # # Combine all document-level errors into one validation_errors array
            # all_errors = []
            # for errs in doc_specific_errors.values():
            #     all_errors.extend(errs)

            # validation_result["validation_errors"] = all_errors
            # validation_result["is_valid"] = len(all_errors) == 0
            # Attach errors to result
            if "addressProof" in validation_result:
                validation_result["addressProof"]["error_messages"] = addprf_error
                validation_result["addressProof"]["status"] = "Valid" if validation_result["addressProof"]["is_valid"] else "Failed"

            if "noc" in validation_result:
                validation_result["noc"]["error_messages"] = noc_error
                validation_result["noc"]["status"] = "Valid" if validation_result["noc"]["is_valid"] else "Failed"

            # Final decision
            validation_result["validation_errors"] = addprf_error + noc_error
            validation_result["is_valid"] = len(addprf_error + noc_error) == 0
            #print(validation_result["addressProof"]["error_messages"])
            return validation_result
                
        except Exception as e:
            self.logger.error(f"Company document validation error: {str(e)}", exc_info=True)
            return {
                "error": str(e),
                "is_valid": False,
                "validation_errors": [str(e)]
            }
        
    def _validate_noc_owner_name(self, actual_owner_name: str, expected_owner_name: str) -> Dict[str, Any]:
        """
        Validate NOC owner name against expected name
        
        Args:
            actual_owner_name (str): Owner name from NOC
            expected_owner_name (str): Expected owner name
        
        Returns:
            dict: Validation result
        """
        # Handle None values
        if not actual_owner_name:
            return {
                "status": "failed",
                "error_message": "Could not extract owner name from NOC"
            }
        
        if not expected_owner_name:
            return {
                "status": "passed",
                "error_message": None
            }
        
        # Normalize names for comparison
        def normalize_name(name):
            # Convert to lowercase, remove punctuation
            import re
            return re.sub(r'[^\w\s]', '', name.lower()).strip()
        
        # Compare normalized names
        if normalize_name(expected_owner_name) != normalize_name(actual_owner_name):
            return {
                "status": "failed",
                "error_message": f"NOC owner name '{actual_owner_name}' does not match expected name '{expected_owner_name}'"
            }
        
        # Names match
        return {
            "status": "passed",
            "error_message": None
        }
    
    def _apply_compliance_rules(
        self, 
        directors_validation: Dict, 
        company_docs_validation: Dict,
        compliance_rules: Dict
    ) -> Dict:
        """
        Apply compliance rules dynamically based on Elasticsearch configuration
        
        Args:
            directors_validation (dict): Director document validation results
            company_docs_validation (dict): Company document validation results
            compliance_rules (dict): Compliance rules from Elasticsearch
        
        Returns:
            dict: Detailed compliance rule validation results
        """
        validation_rules = {}
        
        # Check for validation error in directors validation
        if isinstance(directors_validation, dict) and "validation_error" in directors_validation:
            return {
                "global_error": {
                    "status": "failed",
                    "error_message": directors_validation.get("validation_error", "Unknown validation error"),
                    "global_errors": directors_validation.get("global_errors", []),
                    "director_errors": directors_validation.get("director_errors", {})
                }
            }
        
        try:
            # Extract rules, handling potential nested structure
            rules = self._extract_rules_from_compliance_data(compliance_rules)
            
            # Log processing rules for debugging
            self.logger.info(f"Processing {len(rules)} compliance rules")
            
            # Rule processing map
            rule_processing_map = {
                "DIRECTOR_COUNT": self._validate_director_count_rule,
                "PASSPORT_PHOTO": self._validate_passport_photo_rule,
                "SIGNATURE": self._validate_signature_rule,
                "ADDRESS_PROOF": self._validate_address_proof_rule,
                "INDIAN_DIRECTOR_PAN": self._validate_indian_pan_rule,
                "INDIAN_DIRECTOR_AADHAR": self._validate_indian_aadhar_rule,
                "FOREIGN_DIRECTOR_DOCS": self._validate_foreign_director_rule,
                "COMPANY_ADDRESS_PROOF": self._validate_company_address_proof_rule,
                "NOC_VALIDATION": self._validate_noc_rule,
                "AADHAR_PAN_LINKAGE": self._validate_aadhar_pan_linkage_rule,
                "NOC_OWNER_VALIDATION": self._validate_noc_owner_name_rule
            }
            # Only add new GST rules if they exist in the validation service
            if hasattr(self, '_validate_aadhar_pan_name_match_rule'):
                rule_processing_map["AADHAR_PAN_NAME_MATCH"] = self._validate_aadhar_pan_name_match_rule
            
            if hasattr(self, '_validate_tenant_eb_name_match_rule'):
                rule_processing_map["TENANT_EB_NAME_MATCH"] = self._validate_tenant_eb_name_match_rule
            
            if hasattr(self, '_validate_document_signatures_rule'):
                rule_processing_map["DOCUMENT_SIGNATURES"] = self._validate_document_signatures_rule
            
            if hasattr(self, '_validate_noc_multiple_signatures_rule'):
                rule_processing_map["NOC_MULTIPLE_SIGNATURES"] = self._validate_noc_multiple_signatures_rule
            
            if hasattr(self, '_validate_consent_letter_validation_rule'):
                rule_processing_map["CONSENT_LETTER_VALIDATION"] = self._validate_consent_letter_validation_rule
            
            if hasattr(self, '_validate_board_resolution_validation_rule'):
                rule_processing_map["BOARD_RESOLUTION_VALIDATION"] = self._validate_board_resolution_validation_rule
            
            # Process each rule
            for rule in rules:
                rule_id = rule.get('rule_id', '')
                
                # Skip inactive rules
                if not rule.get('is_active', True):
                    self.logger.info(f"Skipping inactive rule: {rule_id}")
                    continue
                
                # Find appropriate validation method
                validation_method = rule_processing_map.get(rule_id)
                
                if not validation_method:
                    self.logger.warning(f"No validation method found for rule: {rule_id}")
                    continue
                
                try:
                    # Prepare conditions for the rule
                    conditions = rule.get('conditions', {})
                    
                    # For NOC Owner validation, we need preconditions
                    if rule_id == "NOC_OWNER_VALIDATION":
                        # Get preconditions from the last called validate_documents method
                        preconditions = getattr(self, '_current_preconditions', {})
                        validation_result = validation_method(company_docs_validation, conditions, preconditions)
                    # Determine which data to pass based on rule type
                    elif rule_id in ["DIRECTOR_COUNT", "PASSPORT_PHOTO", "SIGNATURE", 
                                "ADDRESS_PROOF", "INDIAN_DIRECTOR_PAN", 
                                "INDIAN_DIRECTOR_AADHAR", "FOREIGN_DIRECTOR_DOCS", 
                                "AADHAR_PAN_LINKAGE"]:
                        validation_result = validation_method(directors_validation, conditions)
                    elif rule_id in ["COMPANY_ADDRESS_PROOF", "NOC_VALIDATION"]:
                        validation_result = validation_method(company_docs_validation, conditions)
                    else:
                        self.logger.warning(f"Unhandled rule type: {rule_id}")
                        continue
                    
                    # Store validation result under lowercase rule_id
                    validation_rules[rule_id.lower()] = validation_result
                    
                    # Log rule validation result for debugging
                    self.logger.info(f"Rule {rule_id} validation result: {validation_result}")
                
                except Exception as rule_error:
                    self.logger.error(f"Error processing rule {rule_id}: {rule_error}", exc_info=True)
                    validation_rules[rule_id.lower()] = {
                        "status": "error",
                        "error_message": str(rule_error)
                    }
            
            return validation_rules
        
        except Exception as e:
            self.logger.error(f"Comprehensive compliance rules application error: {e}", exc_info=True)
            return {
                "global_error": {
                    "status": "error",
                    "error_message": str(e)
                }
            }
    
    def _safe_validate_directors(self, directors_validation):
        """
        Safely convert directors validation to a dictionary
        
        Args:
            directors_validation (dict or str): Input validation data
        
        Returns:
            dict: Processed directors validation
        """
        # If it's already a dictionary, return as-is
        if isinstance(directors_validation, dict):
            return directors_validation
        
        # If it's a string error, return an empty dictionary
        if isinstance(directors_validation, str):
            self.logger.error(f"Received string instead of directors validation: {directors_validation}")
            return {}
        
        # For any other unexpected type
        self.logger.error(f"Unexpected directors validation type: {type(directors_validation)}")
        return {}

    def _validate_director_count_rule(self, directors_validation, conditions):
        """
        Validate director count rule
        
        Args:
            directors_validation (dict): Directors validation data
            conditions (dict): Rule conditions
        
        Returns:
            dict: Validation result
        """
        # Safely process directors validation
        safe_directors = self._safe_validate_directors(directors_validation)
        
        # Get director count
        director_count = len(safe_directors)
        
        # Get conditions
        min_directors = conditions.get('min_directors', 2)
        max_directors = conditions.get('max_directors', 5)
        
        # Validate count
        if director_count < min_directors:
            return {
                "status": "failed",
                "error_message": f"Insufficient directors. Found {director_count}, minimum required is {min_directors}."
            }
        
        if director_count > max_directors:
            return {
                "status": "failed",
                "error_message": f"Too many directors. Found {director_count}, maximum allowed is {max_directors}."
            }
        
        return {
            "status": "passed",
            "error_message": None
        }
    
    def _validate_passport_photo_rule(self, directors_validation: Dict, conditions: Dict) -> Dict:
        """
        Validate passport photo for all directors.

        Args:
            directors_validation (dict): All directors' validation data.
            conditions (dict): Rule-specific conditions.

        Returns:
            dict: Rule result with overall status, message, and per-director details.
        """
        failed_directors = []
        safe_directors = self._safe_validate_directors(directors_validation)
        # Get conditions with more leniency
        min_clarity_score = conditions.get('min_clarity_score', 0.1)  # Lower threshold
        require_passport_style = conditions.get('is_passport_style', False)  # Make optional
        require_face_visible = conditions.get('face_visible', True)  # Keep this requirement
            
        for director_key, director_info in safe_directors.items():
            if not isinstance(director_info, dict):
                continue

            documents = director_info.get("documents", {})
            passport_photo = documents.get("passportPhoto", {})

            if not passport_photo:
                failed_directors.append({
                    "director": director_key,
                    "status": "failed",
                    "error_message": f"Passport photo not uploaded for {director_key}"
                })
                continue

            if not passport_photo.get("is_valid", False):
                failed_directors.append({
                    "director": director_key,
                    "status": "failed",
                    "error_message": f"Invalid passport photo for {director_key}"
                })
                continue
            # Get extraction data
            extracted_data = passport_photo.get('extracted_data', {})
            
            # Only check face visibility as a strict requirement
            if require_face_visible and 'face_visible' in extracted_data:
                if not extracted_data.get('face_visible', False):
                    failed_directors.append({
                        "director": director_key,
                        "status": "failed",
                        "error_message": f"Face not clearly visible in passport photo for {director_key}"
                    })

        if failed_directors:
            return {
                "status": "failed",
                "error_message": "; ".join(f"{d['director']}: {d['error_message']}" for d in failed_directors),
                "details": failed_directors
            }

        return {
            "status": "passed",
            "error_message": None,
            "details": None
        }
    
    def _validate_signature_rule(self, directors_validation: Dict, conditions: Dict) -> Dict:
        """
        Validate signature documents for all directors with leniency.

        Args:
            directors_validation (dict): All directors' validation data.
            conditions (dict): Rule-specific conditions.

        Returns:
            dict: Rule result with overall status, message, and per-director details.
        """
        safe_directors = self._safe_validate_directors(directors_validation)
        min_clarity_score = 0.5 # conditions.get('min_clarity_score', 0.3)
        require_handwritten = conditions.get('is_handwritten', True)
        require_complete = conditions.get('is_complete', True)

        failed_directors = []

        for director_key, director_info in safe_directors.items():
            if not isinstance(director_info, dict):
                continue

            documents = director_info.get("documents", {})
            signature = documents.get("signature", {})

            if not signature:
                failed_directors.append({
                    "director": director_key,
                    "status": "failed",
                    "error_message": f"No signature uploaded for {director_key}"
                })
                continue

            if not signature.get("is_valid", False):
                failed_directors.append({
                    "director": director_key,
                    "status": "failed",
                    "error_message": f"Signature extraction failed for {director_key}"
                })
                continue

            data = signature.get("extracted_data", {})
            clarity_score = data.get("clarity_score", 0)
            is_handwritten = data.get("is_handwritten", False)
            is_complete = data.get("is_complete", False)

            if clarity_score < min_clarity_score:
                failed_directors.append({
                    "director": director_key,
                    "status": "failed",
                    "error_message": f"Low clarity ({clarity_score:.2f}) for {director_key}, minimum is {min_clarity_score}"
                })
            elif require_handwritten and not is_handwritten:
                failed_directors.append({
                    "director": director_key,
                    "status": "failed",
                    "error_message": f"Signature not handwritten for {director_key}"
                })
            elif require_complete and not is_complete:
                failed_directors.append({
                    "director": director_key,
                    "status": "failed",
                    "error_message": f"Incomplete signature for {director_key}"
                })
            
        if failed_directors:
            return {
                "status": "failed",
                "error_message": "; ".join(d["error_message"] for d in failed_directors),
                "details": failed_directors
            }

        return {
            "status": "passed",
            "error_message": None,
            "details": None
        }

    # def _validate_signature_rule(self, directors_validation, conditions):
    #     """
    #     Validate signature rule for all directors with more leniency
        
    #     Args:
    #         directors_validation (dict): Directors validation data
    #         conditions (dict): Rule conditions
        
    #     Returns:
    #         dict: Validation result
    #     """
    #     # Safely process directors validation
    #     safe_directors = self._safe_validate_directors(directors_validation)
        
    #     # Get conditions with lower default thresholds for leniency
    #     min_clarity_score = conditions.get('min_clarity_score', 0.1)  # Lower threshold
    #     require_handwritten = conditions.get('is_handwritten', False)  # Make optional
    #     require_complete = conditions.get('is_complete', False)  # Make optional
        
    #     # Check each director
    #     for director_key, director_info in safe_directors.items():
    #         documents = director_info.get('documents', {})
    #         signature = documents.get('signature', {})
            
    #         # Skip if no signature
    #         if not signature:
    #             # Just log a warning but don't fail
    #             self.logger.warning(f"No signature document found for {director_key}")
    #             continue
            
    #         # If extraction failed, be lenient
    #         if not signature.get('is_valid', False):
    #             self.logger.warning(f"Signature extraction issues for {director_key}, but proceeding with validation")
    #             continue
                
    #         # Get extraction data
    #         extracted_data = signature.get('extracted_data', {})
            
    #         # Check clarity score if available
    #         if 'clarity_score' in extracted_data:
    #             clarity_score = float(extracted_data.get('clarity_score', 0))
    #             if clarity_score < min_clarity_score:
    #                 return {
    #                     "status": "failed",
    #                     "error_message": f"Signature for {director_key} has low clarity ({clarity_score:.2f}), required: {min_clarity_score:.2f}"
    #                 }


    #     # All directors pass the check
    #     return {
    #         "status": "passed",
    #         "error_message": None
    #     }

    def _validate_address_proof_rule(self, directors_validation: Dict, conditions: Dict) -> Dict:
        """
        Validate address proof documents for all directors.

        Args:
            directors_validation (dict): All directors' validation data.
            conditions (dict): Rule-specific conditions.

        Returns:
            dict: Rule result with status, error_message, and details.
        """
        max_age_days = conditions.get('max_age_days', 45)
        name_match_required = conditions.get('name_match_required', True)
        complete_address_required = conditions.get('complete_address_required', True)    
        failed_directors = []
        safe_directors = self._safe_validate_directors(directors_validation)

        for director_key, director_info in safe_directors.items():
            if not isinstance(director_info, dict):
                continue

            documents = director_info.get("documents", {})
            addr_doc = documents.get("address_proof", {})

            if not addr_doc:
                failed_directors.append({
                    "director": director_key,
                    "status": "failed",
                    "error_message": f"No address proof uploaded for {director_key}"
                })
                continue

            if not addr_doc.get("is_valid", False):
                failed_directors.append({
                    "director": director_key,
                    "status": "failed",
                    "error_message": f"Address proof extraction failed for {director_key}"
                })
                continue

            extracted_data = addr_doc.get("extracted_data", {})
            date_str = extracted_data.get("date") or extracted_data.get("bill_date")
            if not date_str:
                failed_directors.append({
                    "director": director_key,
                    "status": "failed",
                    "error_message": f"Missing date in address proof for {director_key}"
                })
                continue

            try:
                doc_date = self._parse_date(date_str)
                if not doc_date:
                    raise ValueError("Date parsing returned None")
                doc_age = (datetime.now() - doc_date).days
                if doc_age > max_age_days:
                    failed_directors.append({
                        "director": director_key,
                        "status": "failed",
                        "error_message": f"Address proof for {director_key} is {doc_age} days old (exceeds {max_age_days} days limit)"
                    })
            except Exception as e:
                failed_directors.append({
                    "director": director_key,
                    "status": "failed",
                    "error_message": f"Error parsing date for {director_key}: {str(e)}"
                })
            if complete_address_required:
                address = extracted_data.get('address', '')
                if not address or len(address.strip()) < 10:
                    failed_directors.append({
                        "director": director_key,
                        "status": "failed",
                        "error_message": f"Address proof for {director_key} does not contain a complete address"
                    })
                            
            # Check name matching
            if name_match_required:
                # Get director name from other documents
                director_name = self._extract_director_name(director_info)
                address_name = extracted_data.get('name') or extracted_data.get('consumer_name')
                
                if director_name and address_name and not self._names_match(director_name, address_name):
                    failed_directors.append({
                        "director": director_key,
                        "status": "failed",
                        "error_message": f"Address proof name '{address_name}' for {director_key} does not match director name '{director_name}'"
                    })
                elif not director_name or not address_name:
                    failed_directors.append({
                        "director": director_key,
                        "status": "failed",
                        "error_message": f"Missing name in address proof for {director_key}"
                    })
                elif director_name != address_name:
                    # If names do not match but are not required to match, log a warning
                    self.logger.warning(f"Address proof name '{address_name}' for {director_key} does not match director name '{director_name}', but name matching is not required.")
        # print("---------------------------------------")
        # print(failed_directors)
        # print("---------------------------------------")
        # print("; ".join([d["error_message"] for d in failed_directors]))
        # print("---------------------------------------")
        if failed_directors:
            return {
                "status": "failed",
                "error_message": "; ".join([d["error_message"] for d in failed_directors]),
                "details": failed_directors
            }

        return {
            "status": "passed",
            "error_message": None,
            "details": None
        }


    def _validate_indian_pan_rule(self, directors_validation: Dict, conditions: Dict) -> Dict:
        """
        Validate PAN for all Indian directors in one go.

        Args:
            directors_validation (dict): Validation data for all directors.
            conditions (dict): PAN validation conditions (e.g., required format, length).

        Returns:
            dict: Rule result with per-director failure details.
        """
        failed_directors = []
        min_age = conditions.get('min_age', 18)
        # Safely process directors validation
        safe_directors = self._safe_validate_directors(directors_validation)
        for director_key, director_info in safe_directors.items():
            if not isinstance(director_info, dict):
                continue

            if director_info.get("nationality", "").lower() != "indian":
                continue

            documents = director_info.get("documents", {})
            pan = documents.get("panCard", {})

            if not pan or not pan.get("extracted_data"):
                failed_directors.append({
                    "director": director_key,
                    "status": "failed",
                    "error_message": f"PAN card not uploaded or extraction failed for {director_key}"
                })
                continue
            extracted_data = pan.get('extracted_data', {})
            pan_number = extracted_data.get('pan_number', '')
            if not pan_number:
                failed_directors.append({
                    "director": director_key,
                    "status": "failed",
                    "error_message": f"PAN number not found for {director_key}"
                })
                continue

            if not re.match(r'^[A-Z]{5}\d{4}[A-Z]{1}$', pan_number):
                failed_directors.append({
                    "director": director_key,
                    "status": "failed",
                    "error_message": f"Invalid PAN format for {director_key}: {pan_number}"
                })
            dob_str = extracted_data.get('dob')
            if dob_str:
                dob_date = self._parse_date(dob_str)
                if dob_date:
                    today = datetime.now()
                    age = today.year - dob_date.year - ((today.month, today.day) < (dob_date.month, dob_date.day))
                    if age < min_age:
                        failed_directors.append({
                            "director": director_key,
                            "status": "failed",
                            "error_message": f"Director {director_key} is {age} years old, below minimum age of {min_age}"
                        })
        if failed_directors:
            return {
                "status": "failed",
                "error_message": "; ".join(f"{d['director']}: {d['error_message']}" for d in failed_directors),
                "details": failed_directors
            }

        return {
            "status": "passed",
            "error_message": None,
            "details": None
        }

    def _validate_applicant_aadhaar(self, extracted_front: dict, extracted_back: dict, conditions: dict = None) -> dict:
        """
        Validate applicant Aadhaar card (front and back) using similar logic as _validate_indian_aadhar_rule.
        """
        errors = []
        masked_not_allowed = True if conditions is None else conditions.get('masked_not_allowed', True)
        different_images_required = True if conditions is None else conditions.get('different_images_required', True)

        # Check both sides present and valid
        if not extracted_front :
            errors.append("Aadhaar front is missing or invalid.")
        if not extracted_back :
            errors.append("Aadhaar back is missing or invalid.")

        if errors:
            return {"is_valid": False, "validation_errors": errors}

        front_data = extracted_front #.get('extracted_data', {})
        back_data = extracted_back #.get('extracted_data', {})

        if not front_data or not back_data:
            errors.append("Missing extracted data for Aadhaar front or back.")
            return {"is_valid": False, "validation_errors": errors}

        # Masked check
        if masked_not_allowed:
            if front_data.get('is_masked', False) and back_data.get('is_masked', False):
                errors.append("Both Aadhaar front and back are masked, need at least one unmasked.")

        # # Consistency check
        # key_fields = ['name', 'dob', 'aadhar_number', 'gender']
        # inconsistent_fields = [
        #     field for field in key_fields
        #     if front_data.get(field) != back_data.get(field)
        # ]
        # if inconsistent_fields:
        #     errors.append(f"Inconsistent Aadhaar fields between front and back: {', '.join(inconsistent_fields)}")

        # Different images check (optional, can be skipped for applicant)
        if different_images_required:
            import hashlib
            front_raw = extracted_front.get('base64') or extracted_front.get('content') or ''
            back_raw = extracted_back.get('base64') or extracted_back.get('content') or ''
            front_hash = hashlib.md5(front_raw.encode()).hexdigest() if front_raw else ''
            back_hash = hashlib.md5(back_raw.encode()).hexdigest() if back_raw else ''
            if front_hash and front_hash == back_hash:
                errors.append("Same image used for Aadhaar front and back.")

        return {"is_valid": len(errors) == 0, "validation_errors": errors}   
    

    def _validate_indian_aadhar_rule(self, directors_validation: Dict, conditions: Dict) -> Dict:
        """
        Validate Aadhar card (front and back) for Indian directors.

        Args:
            directors_validation (dict): Full validation data for all directors.
            conditions (dict): Rule-specific conditions (if any).

        Returns:
            dict: Rule validation result with per-director error reporting.
        """
        failed_directors = []
        safe_directors = self._safe_validate_directors(directors_validation)
        masked_not_allowed = conditions.get('masked_not_allowed', True)
        different_images_required = conditions.get('different_images_required', True)
            
        for director_key, director_info in safe_directors.items():
            if not isinstance(director_info, dict):
                continue

            if director_info.get('nationality', '').lower() != 'indian':
                continue

            documents = director_info.get('documents', {})
            aadhar_front = documents.get('aadharCardFront', {})
            aadhar_back = documents.get('aadharCardBack', {})

            missing_parts = []
            if not aadhar_front or not aadhar_front.get("is_valid", False):
                missing_parts.append("Aadhar front")
            if not aadhar_back or not aadhar_back.get("is_valid", False):
                missing_parts.append("Aadhar back")

            if missing_parts:
                failed_directors.append({
                    "director": director_key,
                    "status": "failed",
                    "error_message": f"{', '.join(missing_parts)} is missing or invalid"
                })
            # Advanced image comparison logic
            front_data = aadhar_front.get('extracted_data', {})
            back_data = aadhar_back.get('extracted_data', {}) if aadhar_back else {}
            # Intelligent image comparison
            # If no extracted data, we can't compare
            if not front_data or not back_data:
                failed_directors.append({
                    "director": director_key,
                    "status": "failed",
                    "error_message": f"Missing extracted data for Aadhar front or back for {director_key}"
                })
                continue
            # Check for masked Aadhar
            if masked_not_allowed:
                # Only check if both are masked - be more lenient
                # if front_data.get('is_masked', False) and back_data.get('is_masked', False):
                #     failed_directors.append({
                #         "director": director_key,
                #         "status": "failed",
                #         "error_message": f"Both Aadhar front and back are masked for {director_key}, need at least one unmasked"
                #     })
                #     continue  # Skip further checks for this director
                
                # If only one is masked, we can still proceed
                if front_data.get('is_masked', False) and not back_data.get('is_masked', False):
                    self.logger.warning(f"Aadhar front is masked for {director_key}, but back is unmasked")
                elif back_data.get('is_masked', False) and not front_data.get('is_masked', False):
                    self.logger.warning(f"Aadhar back is masked for {director_key}, but front is unmasked")
                elif not front_data.get('is_masked', False) and not back_data.get('is_masked', False):
                    self.logger.info(f"Both Aadhar front and back are unmasked for {director_key}")
                else:
                    # If both are masked, we fail the validation
                    failed_directors.append({
                        "director": director_key,
                        "status": "failed",
                        "error_message": f"Both Aadhar front and back are masked for {director_key}, need at least one unmasked"
                    })
                    continue  # Skip further checks for this director
            
            
            # Compare key data points instead of just image URLs
            key_fields = ['name', 'dob', 'aadhar_number', 'gender']
            
            # Check if key information is consistent
            inconsistent_fields = [
                field for field in key_fields 
                if front_data.get(field) != back_data.get(field)
            ]
            
            # If different_images_required is True, do stricter checking
            if different_images_required:
                # Check URL or file uniqueness
                # Try comparing base64 content if URL is missing
                front_raw = aadhar_front.get('base64') or aadhar_front.get('content') or ''
                back_raw = aadhar_back.get('base64') or aadhar_back.get('content') or ''

                # fallback: compare by hash (short hash, to avoid long strings)
                import hashlib
                front_hash = hashlib.md5(front_raw.encode()).hexdigest() if front_raw else ''
                back_hash = hashlib.md5(back_raw.encode()).hexdigest() if back_raw else ''

                # If hashes are equal, assume same image
                if front_hash and front_hash == back_hash:
                    if len(inconsistent_fields) > 1:
                        failed_directors.append({
                            "director": director_key,
                            "status": "failed",
                            "error_message": f"Same image used for Aadhar front and back for {director_key}"
                        })
                        continue  # Skip further checks for this director

                    # Log a warning about potential duplicate
                    self.logger.warning(f"Potential duplicate Aadhar images for {director_key}")
            
            # Optional: Add logging for inconsistent fields
            if inconsistent_fields:
                self.logger.warning(f"Inconsistent Aadhar fields for {director_key}: {inconsistent_fields}")
        
        if failed_directors:
            return {
                    "status": "failed",
                    "error_message": "; ".join(
                        f"{d['director']}: {d['error_message']}" for d in failed_directors
                    ),
                    "details": failed_directors
                }

        return {
            "status": "passed",
            "error_message": None,
            "details": None
        }
    
    def _validate_foreign_director_rule(self, directors_validation: Dict, conditions: Dict) -> Dict:
        """
        Validate passport or driving license for all foreign directors, with fallback to PAN.
        Passport expiry and completeness also checked.
        """
        failed_directors = []
        safe_directors = self._safe_validate_directors(directors_validation)
        passport_required = conditions.get('passport_required', True)

        for director_key, director_info in safe_directors.items():
            if not isinstance(director_info, dict):
                continue

            if director_info.get('nationality', '').lower() != 'foreign':
                continue

            documents = director_info.get('documents', {})
            passport = documents.get('passport', {})
            driving_license = documents.get('drivingLicense', {})
            pan_card = documents.get('panCard', {})

            has_license = driving_license.get('is_valid', False)
            has_passport = False

            # Check passport validity and content
            #if passport.get('is_valid', False):
            passport_data = passport.get('extracted_data', {})
            verified_data = self.extraction_service._verify_passport_data(passport_data)
            if verified_data:
                print("Verified passport data:", verified_data)
                has_passport = True
            else:
                failed_directors.append({
                    "director": director_key,
                    "status": "failed",
                    "error_message": f"Invalid or expired passport for {director_key}"
                })

            if not has_passport and not has_license:
                if pan_card.get('is_valid', False):
                    self.logger.info(f"Using PAN card as ID document for foreign director {director_key}")
                else:
                    failed_directors.append({
                        "director": director_key,
                        "status": "failed",
                        "error_message": f"Passport or Driving License is required for foreign directors (PAN fallback also missing) - {director_key}"
                    })

        if failed_directors:
            print("Failed directors:", "; ".join(f"{d['director']}: {d['error_message']}" for d in failed_directors))
            return {
                "status": "failed",
                "error_message": "; ".join(f"{d['director']}: {d['error_message']}" for d in failed_directors),
                "details": failed_directors
            }

        return {
            "status": "passed",
            "error_message": None,
            "details": None
        }

    # def _validate_foreign_director_rule(self, directors_validation: Dict, conditions: Dict) -> Dict:
    #     """
    #     Validate passport or driving license for all foreign directors.

    #     Args:
    #         directors_validation (dict): Full validation data for all directors.
    #         conditions (dict): Rule-specific conditions (if any).

    #     Returns:
    #         dict: Rule validation result with per-director error reporting.
    #     """
    #     failed_directors = []
    #     safe_directors = self._safe_validate_directors(directors_validation)
    #     passport_required = conditions.get('passport_required', True)

            
    #     for director_key, director_info in safe_directors.items():
    #         if not isinstance(director_info, dict):
    #             continue

    #         if director_info.get('nationality', '').lower() != 'foreign':
    #             continue

    #         documents = director_info.get('documents', {})
    #         passport = documents.get('passport', {})
    #         driving_license = documents.get('drivingLicense', {})

    #         has_passport = passport.get('is_valid', False)
    #         has_license = driving_license.get('is_valid', False)

    #         if not (has_passport or has_license):
    #             # Check if panCard exists and is valid as alternative ID
    #             pan_card = documents.get('panCard', {})
    #             if not pan_card or not pan_card.get('is_valid', False):
    #                 failed_directors.append({
    #                     "director": director_key,
    #                     "status": "failed",
    #                     "error_message": "Passport or Driving License is required for foreign directors"
    #                 })
    #             else:
    #                 # PanCard is being used as ID document
    #                 self.logger.info(f"Using PAN card as ID document for foreign director {director_key}")

    #     if failed_directors:
    #         return {
    #             "status": "failed",
    #             "error_message": "; ".join(
    #                 f"{d['director']}: {d['error_message']}" for d in failed_directors
    #             ),
    #             "details": failed_directors
    #         }

    #     return {
    #         "status": "passed",
    #         "error_message": None,
    #         "details": None
    #     }
    

    def _validate_company_address_proof_rule(self, company_docs_validation, conditions):
        """
        Validate company address proof with comprehensive date and address parsing
        
        Args:
            company_docs_validation (dict): Company document validation data
            conditions (dict): Rule conditions from compliance rules
            
        Returns:
            dict: Validation result with status and error message
        """
        # Get rule conditions with defaults
        max_age_days = conditions.get('max_age_days', 45)
        complete_address_required = conditions.get('complete_address_required', True)
        name_match_required = conditions.get('name_match_required', False)
        
        # Check if address proof exists and is valid
        address_proof = company_docs_validation.get('addressProof', {})
        if not address_proof or not address_proof.get('is_valid', False):
            return {
                "status": "failed",
                "error_message": "Valid company address proof required"
            }

        # Safely extract document data
        extracted_data = address_proof.get('extracted_data', {})
        fields = extracted_data.get('extracted_fields', {})
        
        # If no extracted_fields, try to use extracted_data directly
        if not fields and extracted_data:
            fields = extracted_data

        # 1. Date validation
        date_keys = [
            'date', 'bill_date', 'invoice_date', 'billing_date', 
            'generated_on', 'due_date', 'txn_date', 'value_date',
            'document_date', 'issue_date'
        ]
        
        doc_date = None
        date_found = None
        
        for key in date_keys:
            if key in fields and fields[key]:
                try:
                    doc_date = self._parse_date(fields[key])
                    if doc_date:
                        date_found = key;
                        break
                except Exception as e:
                    self.logger.debug(f"Failed to parse date from {key}: {fields[key]} - {e}")
                    continue
        
        if not doc_date:
            return {
                "status": "failed",
                "error_message": "No valid date found in company address proof document"
            }

        # Check document age
        doc_age = (datetime.now() - doc_date).days
        if doc_age > max_age_days:
            return {
                "status": "failed",
                "error_message": f"Company address proof is {doc_age} days old (exceeds {max_age_days} days limit)"
            }

        # 2. Address validation
        if complete_address_required:
            address_fields = [
                'address', 'billing_address', 'consumer_address', 
                'service_address', 'full_address', 'complete_address'
            ]
            
            address = None
            address_field_found = None
            
            for field in address_fields:
                if field in fields and fields[field]:
                    address = fields[field].strip()
                    if address and len(address) >= 10:  # Minimum viable address length
                        address_field_found = field
                        break
            
            if not address or len(address) < 10:
                self.logger.debug(f"Available fields: {list(fields.keys())}")
                self.logger.debug(f"Address found: '{address}' from field: {address_field_found}")
                return {
                    "status": "failed",
                    "error_message": "Company address proof does not contain a complete address (minimum 10 characters required)"
                }

        # 3. Name matching validation (if required)
        if name_match_required:
            # This would require company name from preconditions or input data
            # Implementation depends on your specific requirements
            company_name = self._current_preconditions.get('company_name', '')
            if company_name:
                name_fields = ['company_name', 'business_name', 'consumer_name', 'name']
                doc_name = None
                
                for field in name_fields:
                    if field in fields and fields[field]:
                        doc_name = fields[field].strip()
                        break
                
                if not doc_name or company_name.lower() not in doc_name.lower():
                    return {
                        "status": "failed",
                        "error_message": "Company name in address proof does not match registered company name"
                    }

        # All validations passed
        return {
            "status": "passed",
            "error_message": None
        }
    def _validate_aadhar_pan_name_match_rule(self, directors_validation, conditions):
        """
        Validate that names on Aadhar and PAN match
        
        Args:
            directors_validation (dict): Directors validation data
            conditions (dict): Rule conditions
        
        Returns:
            dict: Validation result
        """
        # Safely process directors validation
        safe_directors = self._safe_validate_directors(directors_validation)
        
        # Get conditions
        strict_match = conditions.get('strict_match', False)
        fuzzy_matching = conditions.get('fuzzy_matching', True)
        
        # Check each director
        for director_key, director_info in safe_directors.items():
            # Skip if not a dictionary
            if not isinstance(director_info, dict):
                continue
                
            # Skip special keys
            if director_key in ['global_errors', 'rule_validations']:
                continue
                
            documents = director_info.get('documents', {})
            
            # Get Aadhar and PAN documents
            aadhar_front = documents.get('aadharCardFront', {})
            pan_card = documents.get('panCard', {})
            
            # Skip if either document is missing
            if not aadhar_front or not pan_card:
                continue
            
            # Get extraction data
            aadhar_data = aadhar_front.get('extracted_data', {})
            pan_data = pan_card.get('extracted_data', {})
            
            # Get names
            aadhar_name = aadhar_data.get('name', '')
            pan_name = pan_data.get('name', '')
            
            # Skip if either name is missing
            if not aadhar_name or not pan_name:
                continue
            
            # Check name match
            names_match = False
            if strict_match:
                # Exact match required
                norm_aadhar = self._normalize_name(aadhar_name)
                norm_pan = self._normalize_name(pan_name)
                names_match = norm_aadhar == norm_pan
            elif fuzzy_matching:
                # Fuzzy matching
                names_match = self._names_match(aadhar_name, pan_name)
            
            if not names_match:
                return {
                    "status": "failed",
                    "error_message": f"Names on Aadhar ({aadhar_name}) and PAN ({pan_name}) do not match for director {director_key}"
                }
        
        # All directors pass the check
        return {
            "status": "passed",
            "error_message": None
        }

    def _validate_tenant_eb_name_match_rule(self, directors_validation, conditions):
        """
        Validate tenant name matches electricity bill name
        
        Args:
            directors_validation (dict): Directors validation data
            conditions (dict): Rule conditions
        
        Returns:
            dict: Validation result
        """
        # Safely process directors validation
        safe_directors = self._safe_validate_directors(directors_validation)
        
        # Get conditions
        strict_match = conditions.get('strict_match', False)
        fuzzy_matching = conditions.get('fuzzy_matching', True)
        
        # Get all tenant names from directors
        tenant_names = []
        for director_key, director_info in safe_directors.items():
            # Skip if not a dictionary or special keys
            if not isinstance(director_info, dict) or director_key in ['global_errors', 'rule_validations']:
                continue
                
            director_name = self._extract_director_name(director_info)
            if director_name:
                tenant_names.append(director_name)
        
        # If no tenant names found
        if not tenant_names:
            return {
                "status": "failed",
                "error_message": "No tenant name found for verification"
            }
        
        # Get electricity bill name (from address_proof documents)
        eb_names = []
        for director_key, director_info in safe_directors.items():
            # Skip if not a dictionary or special keys
            if not isinstance(director_info, dict) or director_key in ['global_errors', 'rule_validations']:
                continue
                
            documents = director_info.get('documents', {})
            address_proof = documents.get('address_proof', {})
            if address_proof and address_proof.get('is_valid', False):
                extracted_data = address_proof.get('extracted_data', {})
                eb_name = extracted_data.get('consumer_name') or extracted_data.get('name')
                if eb_name:
                    eb_names.append(eb_name)
        
        # Check company documents if available
        company_docs = getattr(self, '_current_company_docs', {})
        address_proof = company_docs.get('addressProof', {})
        if address_proof and address_proof.get('is_valid', False):
            extracted_data = address_proof.get('extracted_data', {})
            eb_name = extracted_data.get('consumer_name') or extracted_data.get('name')
            if eb_name:
                eb_names.append(eb_name)
        
        # If no EB names found
        if not eb_names:
            return {
                "status": "failed",
                "error_message": "No electricity bill name found for verification"
            }
        
        # Check for any match between tenant names and EB names
        match_found = False
        for tenant_name in tenant_names:
            for eb_name in eb_names:
                if strict_match:
                    norm_tenant = self._normalize_name(tenant_name)
                    norm_eb = self._normalize_name(eb_name)
                    if norm_tenant == norm_eb:
                        match_found = True
                        break
                elif fuzzy_matching:
                    if self._names_match(tenant_name, eb_name):
                        match_found = True
                        break
            if match_found:
                break
        
        if not match_found:
            return {
                "status": "failed",
                "error_message": f"Tenant name does not match electricity bill name"
            }
        
        return {
            "status": "passed",
            "error_message": None
        }

    def _validate_document_signatures_rule(self, directors_validation, conditions):
        """
        Validate that all pages have required signatures from both tenant and landlord
        
        Args:
            directors_validation (dict): Directors validation data
            conditions (dict): Rule conditions
        
        Returns:
            dict: Validation result
        """
        # Get conditions
        tenant_signature_required = conditions.get('tenant_signature_required', True)
        landlord_signature_required = conditions.get('landlord_signature_required', True)
        all_pages_required = conditions.get('all_pages_required', True)
        
        # Check for tenant signatures
        tenant_signatures_valid = True
        missing_tenant_signatures = []
        
        # Safely process directors validation
        safe_directors = self._safe_validate_directors(directors_validation)
        
        for director_key, director_info in safe_directors.items():
            # Skip if not a dictionary or special keys
            if not isinstance(director_info, dict) or director_key in ['global_errors', 'rule_validations']:
                continue
                
            documents = director_info.get('documents', {})
            signature = documents.get('signature', {})
            
            if not signature or not signature.get('is_valid', False):
                tenant_signatures_valid = False
                missing_tenant_signatures.append(director_key)
        
        if not tenant_signatures_valid and tenant_signature_required:
            return {
                "status": "failed",
                "error_message": f"Missing valid signature documents for directors: {', '.join(missing_tenant_signatures)}"
            }
        
        # For landlord signatures, we need to access company docs validation 
        # Check if we're in a context where company_docs_validation is available
        if landlord_signature_required:
            # Try to get company docs from validation context
            try:
                # Access the company docs from the current validation context
                company_docs = getattr(self, '_current_company_docs', None)
                
                if company_docs is None:
                    self.logger.warning("Cannot validate landlord signatures: company_docs not available")
                    return {
                        "status": "failed", 
                        "error_message": "Cannot validate landlord signatures: required documents not available"
                    }
                
                # Check NOC for landlord signatures
                noc = company_docs.get('noc', {})
                
                if not noc or not noc.get('is_valid', False):
                    return {
                        "status": "failed",
                        "error_message": "Valid NOC document with landlord signature required"
                    }
                
                # Get extracted data
                extracted_data = noc.get('extracted_data', {})
                
                # Check for landlord signature
                has_landlord_signature = extracted_data.get('has_signature', False)
                
                if not has_landlord_signature:
                    return {
                        "status": "failed",
                        "error_message": "NOC document missing required property owner's signature"
                    }
                
                # Check if all pages are signed
                if all_pages_required:
                    all_pages_signed = extracted_data.get('all_pages_signed', False)
                    
                    if not all_pages_signed:
                        return {
                            "status": "failed",
                            "error_message": "Not all pages of NOC document have required signatures"
                        }
            except Exception as e:
                self.logger.error(f"Error validating landlord signatures: {str(e)}", exc_info=True)
                return {
                    "status": "failed",
                    "error_message": f"Error validating landlord signatures: {str(e)}"
                }
        
        # All signature checks passed
        return {
            "status": "passed",
            "error_message": None
        }

    def _validate_noc_multiple_signatures_rule(self, company_docs_validation, conditions):
        """
        Validate NOC has multiple signatures if multiple landlords
        
        Args:
            company_docs_validation (dict): Company document validation data
            conditions (dict): Rule conditions
        
        Returns:
            dict: Validation result
        """
        # Get conditions
        verify_multiple_signatures = conditions.get('verify_multiple_signatures', True)
        
        if not verify_multiple_signatures:
            return {
                "status": "passed",
                "error_message": None
            }
        
        # Check if NOC exists
        noc = company_docs_validation.get('noc', {})
        if not noc or not noc.get('is_valid', False):
            return {
                "status": "failed",
                "error_message": "Valid NOC document required"
            }
        
        # Get extracted NOC data
        extracted_data = noc.get('extracted_data', {})
        
        # Check for multiple landlords
        owner_name = extracted_data.get('owner_name', '')
        
        # Check if multiple owners indicated by 'and' or '&' in the name
        multiple_owners_indicated = '&' in owner_name or ' and ' in owner_name.lower() or ',' in owner_name
        
        # If there are multiple signatories listed, that's another indication of multiple owners
        signatories = extracted_data.get('signatories', [])
        if isinstance(signatories, list) and len(signatories) > 1:
            multiple_owners_indicated = True
        
        # Check if multiple signatures detected
        # Use the signature_count if available, or has_multiple_signatures flag
        signature_count = extracted_data.get('signature_count', 0)
        has_multiple_signatures = extracted_data.get('has_multiple_signatures', False)
        
        multiple_signatures_detected = signature_count > 1 or has_multiple_signatures
        
        if multiple_owners_indicated and not multiple_signatures_detected:
            return {
                "status": "failed",
                "error_message": "Multiple landlords detected but not all signatures found on NOC"
            }
        
        return {
            "status": "passed",
            "error_message": None
        }

    def _validate_consent_letter_validation_rule(self, company_docs_validation, conditions):
        """
        Validate consent letter requirements for GST Family Owned Property
        
        Args:
            company_docs_validation (dict): Company document validation data
            conditions (dict): Rule conditions
        
        Returns:
            dict: Validation result
        """
        # Get conditions
        stamp_paper_required = conditions.get('stamp_paper_required', True)
        notarization_required = conditions.get('notarization_required', True)
        firm_name_required = conditions.get('firm_name_required', True)
        relation_required = conditions.get('relation_required', True)
        landlord_details_required = conditions.get('landlord_details_required', True)
        
        # Check if consent letter exists
        consent_letter = company_docs_validation.get('consent_letter', {})
        if not consent_letter or not consent_letter.get('is_valid', False):
            return {
                "status": "failed",
                "error_message": "Valid consent letter required for family owned property"
            }
        
        # Get extracted consent letter data
        extracted_data = consent_letter.get('extracted_data', {})
        
        # Initialize validation failures
        validation_failures = []
        
        # Check stamp paper requirement
        if stamp_paper_required and not extracted_data.get('on_stamp_paper', False):
            validation_failures.append("Consent letter must be executed on stamp paper")
        
        # Check notarization
        if notarization_required and not extracted_data.get('is_notarized', False):
            validation_failures.append("Consent letter must be notarized on all pages")
        
        # Check firm name
        if firm_name_required and not extracted_data.get('firm_name'):
            validation_failures.append("Firm name must be included in consent letter")
        
        # Check relation mention
        if relation_required and not extracted_data.get('relation_mentioned', False):
            validation_failures.append("Relation between landlord and applicant must be clearly mentioned")
        
        # Check landlord details
        if landlord_details_required:
            if not extracted_data.get('landlord_name'):
                validation_failures.append("Landlord's name must be included in consent letter")
            if not extracted_data.get('landlord_address'):
                validation_failures.append("Landlord's address must be included in consent letter")
        
        if validation_failures:
            return {
                "status": "failed",
                "error_message": "; ".join(validation_failures)
            }
        
        return {
            "status": "passed",
            "error_message": None
        }

    def _validate_board_resolution_validation_rule(self, company_docs_validation, conditions):
        """
        Validate board resolution requirements for GST PVT/LLP
        
        Args:
            company_docs_validation (dict): Company document validation data
            conditions (dict): Rule conditions
        
        Returns:
            dict: Validation result
        """
        # Get conditions
        company_name_required = conditions.get('company_name_required', True)
        company_address_required = conditions.get('company_address_required', True)
        date_required = conditions.get('date_required', True)
        address_verification = conditions.get('address_verification', True)
        
        # Check if board resolution exists
        board_resolution = company_docs_validation.get('board_resolution', {})
        if not board_resolution or not board_resolution.get('is_valid', False):
            return {
                "status": "failed",
                "error_message": "Valid board resolution required for PVT/LLP property"
            }
        
        # Get extracted board resolution data
        extracted_data = board_resolution.get('extracted_data', {})
        
        # Initialize validation failures
        validation_failures = []
        
        # Check company name
        if company_name_required and not extracted_data.get('company_name'):
            validation_failures.append("Board Resolution must have company name")
        
        # Check company address
        if company_address_required and not extracted_data.get('company_address'):
            validation_failures.append("Board Resolution must have company address")
        
        # Check date
        if date_required and not extracted_data.get('date'):
            validation_failures.append("Board Resolution must have date mentioned")
        
        # Check address verification if required
        if address_verification:
            company_address = extracted_data.get('company_address', '')
            address_proof = company_docs_validation.get('addressProof', {})
            
            if address_proof and address_proof.get('is_valid', False):
                proof_address = address_proof.get('extracted_data', {}).get('address', '')
                
                if not self._names_match(company_address, proof_address):
                    validation_failures.append("Board Resolution address does not match address proof")
        
        if validation_failures:
            return {
                "status": "failed",
                "error_message": "; ".join(validation_failures)
            }
        
        return {
            "status": "passed",
            "error_message": None
        }
    
    def _validate_noc_rule(self, company_docs_validation, conditions):
        """
        Validate No Objection Certificate (NOC) with more flexible validation
        
        Args:
            company_docs_validation (dict): Company document validation data
            conditions (dict): Rule conditions
        
        Returns:
            dict: Validation result
        """
        # Get conditions
        noc_required = conditions.get('noc_required', True)
        signature_required = conditions.get('signature_required', True)
        
        # If NOC is not required, pass automatically
        # if not noc_required:
        #     return {
        #         "status": "passed",
        #         "error_message": None
        #     }
        
        # Check if NOC exists in company documents
        noc = company_docs_validation.get('noc', {})
        
        # If no NOC document found and it's required
        if not noc:
            return {
                "status": "failed",
                "error_message": "No Objection Certificate (NOC) is required but not provided"
            }
        
        # Get extracted NOC data
        extracted_data = noc.get('extracted_data', {})
        
        # Comprehensive NOC validation checks
        validation_checks = []
        
        # Check for mandatory fields
        mandatory_fields = ['owner_name', 'property_address', 'applicant_name', 'date']
        missing_fields = [field for field in mandatory_fields if not extracted_data.get(field)]
        
        if missing_fields:
            validation_checks.append(f"Missing mandatory NOC fields: {', '.join(missing_fields)}")
        
        # Date validation - should be recent (within last 90 days)
        if 'date' in extracted_data:
            try:
                noc_date = self._parse_date(extracted_data['date'])
                if noc_date:
                    today = datetime.now()
                    noc_age = (today - noc_date).days
                    
                    # Allow NOC up to 90 days old
                    if noc_age > 90:
                        validation_checks.append(f"NOC is {noc_age} days old (exceeds 90 days limit)")
            except Exception as e:
                validation_checks.append(f"Invalid NOC date: {str(e)}")
        
        # Signature validation
        if signature_required:
            # Check for signature presence
            has_signature = extracted_data.get('has_signature', False)
            
            # Additional checks for signature validity
            if not has_signature:
                validation_checks.append("NOC lacks required property owner's signature")
        
        # Property address check
        if 'property_address' in extracted_data:
            address = extracted_data['property_address']
            if not address or len(address.strip()) < 10:
                validation_checks.append("Incomplete or invalid property address")
        
        # Purpose validation
        purpose = extracted_data.get('purpose', '')
        if not purpose or len(purpose.strip()) < 5:
            validation_checks.append("Invalid or missing purpose in NOC")
        
        # Comprehensive owner name and applicant name validation
        owner_name = extracted_data.get('owner_name', '').strip()
        applicant_name = extracted_data.get('applicant_name', '').strip()
        
        if not owner_name or not applicant_name:
            validation_checks.append("Missing owner or applicant name")
        
        # If any validation checks failed, return failure
        if validation_checks:
            return {
                "status": "failed",
                "error_message": "; ".join(validation_checks)
            }
        
        # Clarity score check
        clarity_score = extracted_data.get('clarity_score', 0)
        if clarity_score < 0.7:  # Minimum clarity threshold
            return {
                "status": "failed",
                "error_message": f"Low document clarity: {clarity_score}"
            }
        
        # Validate NOC is marked as valid
        is_valid_noc = extracted_data.get('is_valid_noc', False)
        if not is_valid_noc:
            return {
                "status": "failed",
                "error_message": "Document does not appear to be a valid NOC"
            }
        
        # All checks passed
        return {
            "status": "passed",
            "error_message": None
        }
    
    def _validate_aadhar_pan_linkage_rule(self, directors_validation, conditions):
        """
        Validate Aadhar PAN linkage with strict error handling
        
        Args:
            directors_validation (dict): Directors validation data
            conditions (dict): Rule conditions
        
        Returns:
            dict: Validation result
        """
        # Safely process directors validation
        safe_directors = self._safe_validate_directors(directors_validation)
        failed_directors = []
        # Check if linkage check is required
        linkage_api_check_required = conditions.get('linkage_api_check_required', True)
        if not linkage_api_check_required:
            return {
                "status": "passed",
                "error_message": None
            }
        
        # Check each director
        for director_key, director_info in safe_directors.items():
            # Only validate Indian directors
            if director_info.get('nationality', '').lower() != 'indian':
                continue
                
            documents = director_info.get('documents', {})
            
            # Get Aadhar and PAN documents
            aadhar_front = documents.get('aadharCardFront', {})
            aadhar_back = documents.get('aadharCardBack', {})
            pan_card = documents.get('panCard', {})
            
            # Check if both documents exist
            if not aadhar_front and not aadhar_back:
                failed_directors.append({
                    "director": director_key,
                    "status": "failed",
                    "error_message": f"No Aadhar card found for {director_key}"
                })
                continue
                     
            if not pan_card:
                failed_directors.append({
                    "director": director_key,
                    "status": "failed",
                    "error_message": f"No PAN card found for {director_key}"
                })
                #self.logger.warning(f"No PAN card found for {director_key}")
                continue
            
            # Get extraction data
            aadhar_data = aadhar_front.get('extracted_data', {})
            aadhar_back_data = aadhar_back.get('extracted_data', {}) if aadhar_back else {}
            pan_data = pan_card.get('extracted_data', {})
            
            # Get Aadhar number (try from both front and back)
            aadhar_number = aadhar_data.get('aadhar_number', '')
            
            # If front is masked, try to get from back
            if aadhar_data.get('is_masked', False) and aadhar_back_data:
                aadhar_number = aadhar_back_data.get('aadhar_number', aadhar_number)
            
            pan_number = pan_data.get('pan_number', '')
            
            # Check if both numbers are available and valid
            if not aadhar_number or 'XXXX' in aadhar_number:
                failed_directors.append({
                    "director": director_key,
                    "status": "failed",
                    "error_message": f"Masked or missing Aadhar number for {director_key}"
                })
                #self.logger.warning(f"Masked or missing Aadhar number for {director_key}")
                continue
                
            if not pan_number:
                failed_directors.append({
                    "director": director_key,
                    "status": "failed",
                    "error_message": f"Missing PAN number for {director_key}"
                })
                #self.logger.warning(f"Missing PAN number for {director_key}")
                continue
            
            # Remove spaces and any other non-numeric characters from Aadhar number
            formatted_aadhar = re.sub(r'\D', '', aadhar_number)
            
            # Verify linkage
            try:
                self.logger.info(f"Verifying Aadhar-PAN linkage for {director_key}: Aadhar={formatted_aadhar}, PAN={pan_number}")
                
                linkage_result = self.aadhar_pan_linkage_service.verify_linkage(
                    formatted_aadhar,
                    pan_number
                )
                
                # Log the result for debugging
                self.logger.info(f"Linkage result: {linkage_result}")
                
                # Strictly check for linkage - fail on any error or non-linked status
                if not linkage_result.get('is_linked', False):
                    error_message = linkage_result.get('message', 'Unknown error')
                    failed_directors.append({
                        "director": director_key,
                        "status": "failed",
                        "error_message": f"Aadhar and PAN not linked for {director_key}: {error_message}"
                    })
                    self.logger.warning(f"Aadhar and PAN not linked for {director_key}: {error_message}")
                    continue
               
                
                # Successful linkage for at least one Indian director
                failed_directors.append({
                    "director": director_key,
                    "status": "passed",
                    "error_message": None
                })
                self.logger.info(f"Aadhar and PAN successfully linked for {director_key}")
    
            except Exception as e:
                self.logger.error(f"Error verifying Aadhar-PAN linkage for {director_key}: {str(e)}", exc_info=True)
                failed_directors.append({
                    "director": director_key,
                    "status": "failed",
                    "error_message": f"Error during Aadhar-PAN linkage verification for {director_key}: {str(e)}"
                })
                self.logger.error(f"Comprehensive linkage verification error: {e}", exc_info=True)
                
        if failed_directors:
            return {
                "status": "failed",
                "error_message": "; ".join(
                    f"{d['director']}: {d['error_message']}" for d in failed_directors
                ),
                "details": failed_directors
            }
        # No Indian directors found for linkage check
        return {
            "status": "passed",
            "error_message": None,
            "details": None
        }
    
    def _extract_director_name(self, director_info):
        """
        Extract director name from documents
        
        Args:
            director_info (dict): Director information
        
        Returns:
            str: Director name
        """
        # Priority order for documents to get name from
        priority_docs = ['panCard', 'aadharCardFront', 'passport', 'drivingLicense']
        
        documents = director_info.get('documents', {})
        
        # Try to get name from documents in priority order
        for doc_key in priority_docs:
            if doc_key in documents:
                doc = documents[doc_key]
                extracted_data = doc.get('extracted_data', {})
                name = extracted_data.get('name')
                if name:
                    return name
        
        # If no name found, try any other document
        for doc_key, doc in documents.items():
            extracted_data = doc.get('extracted_data', {})
            name = extracted_data.get('name')
            if name:
                return name
        
        # No name found
        return None
    
    def _get_director_names(self, directors):
        """
        Get all director names
        
        Args:
            directors (dict): Directors data
        
        Returns:
            list: Director names
        """
        director_names = []
        
        # Check if directors is a dict
        if not isinstance(directors, dict):
            self.logger.warning(f"Invalid directors input in _get_director_names. Expected dict, got {type(directors)}")
            return director_names
        
        for director_key, director_info in directors.items():
            if not isinstance(director_info, dict):
                continue
                
            name = self._extract_director_name(director_info)
            if name:
                director_names.append(name)
        
        return director_names

    def _extract_director_name(self, director_info):
        """
        Extract director name from documents
        
        Args:
            director_info (dict): Director information
        
        Returns:
            str: Director name
        """
        # Check if director_info is a dict
        if not isinstance(director_info, dict):
            self.logger.warning(f"Invalid director_info in _extract_director_name. Expected dict, got {type(director_info)}")
            return None
            
        # Priority order for documents to get name from
        priority_docs = ['panCard', 'aadharCardFront', 'passport', 'drivingLicense']
        
        documents = director_info.get('documents', {})
        
        # Try to get name from documents in priority order
        for doc_key in priority_docs:
            if doc_key in documents:
                doc = documents[doc_key]
                if not isinstance(doc, dict):
                    continue
                    
                extracted_data = doc.get('extracted_data', {})
                name = extracted_data.get('name')
                if name:
                    return name
        
        # If no name found, try any other document
        for doc_key, doc in documents.items():
            if not isinstance(doc, dict):
                continue
                
            extracted_data = doc.get('extracted_data', {})
            name = extracted_data.get('name')
            if name:
                return name
        
        # No name found
        return None
    
    def _parse_date(self, date_str):
        """
        Parse date string in multiple formats with better detection
        
        Args:
            date_str (str): Date string
        
        Returns:
            datetime: Parsed date or None
        """
        if not date_str:
            return None
        
        # Pre-process the date string
        date_str = date_str.strip()
        
        # Try multiple date formats in order of preference
        formats = [
            '%Y-%m-%d',           # 2024-01-15
            '%d-%m-%Y',           # 15-01-2024
            '%d/%m/%Y',           # 15/01/2024
            '%m/%d/%Y',           # 01/15/2024
            '%Y/%m/%d',           # 2024/01/15
            '%d.%m.%Y',           # 15.01.2024
            '%Y.%m.%d',           # 2024.01.15
            '%d %b %Y',           # 15 Jan 2024
            '%d %B %Y',           # 15 January 2024
            '%b %d, %Y',          # Jan 15, 2024
            '%B %d, %Y',          # January 15, 2024
            '%Y-%m-%d %H:%M:%S',  # 2024-01-15 10:30:00
            '%d-%m-%Y %H:%M:%S',  # 15-01-2024 10:30:00
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        # Try with dateutil parser as fallback
        try:
            # Force day first parsing since most dates in India are DD/MM/YYYY
            parsed_date = parser.parse(date_str, dayfirst=True)
            
            # Extra validation - reject future dates
            if parsed_date > datetime.now() + timedelta(days=3):  # Allow 3 days for timezone differences
                self.logger.warning(f"Rejecting future date: {parsed_date}")
                return None
                
            return parsed_date
        except Exception:
            return None
    
    def _names_match(self, name1, name2):
        """
        Check if names match with fuzzy logic
        
        Args:
            name1 (str): First name
            name2 (str): Second name
        
        Returns:
            bool: Whether names match
        """
        # Handle None values
        if not name1 or not name2:
            return False
        
        # Normalize names
        def normalize_name(name):
            # Convert to lowercase and remove punctuation
            name = re.sub(r'[^\w\s]', '', name.lower())
            # Remove multiple spaces
            name = re.sub(r'\s+', ' ', name).strip()
            return name
        
        norm1 = normalize_name(name1)
        norm2 = normalize_name(name2)
        
        # Check for exact match
        if norm1 == norm2:
            return True
        
        # Check if one is substring of another
        if norm1 in norm2 or norm2 in norm1:
            return True
        
        # Split names into parts
        parts1 = set(norm1.split())
        parts2 = set(norm2.split())
        
        # Check for common words
        common_words = parts1.intersection(parts2)
        
        # If at least 50% words match
        return len(common_words) >= min(len(parts1), len(parts2)) / 2

    def _validate_tm_documents(self, service_id, request_id, input_data, compliance_rules, start_time):
        """
        Validate documents for TM services
        """
        try:
            # Extract rules
            rules = self._extract_rules_from_compliance_data(compliance_rules)
            
            # Initialize validation results
            validation_results = {
                "applicant_validation": {},
                "trademark_validation": {},
                "director_validation": {},
                "company_documents_validation": {},
                "rule_validations": {}
            }
            
            # 1. Validate applicant information
            applicant_validation = self._validate_tm_applicant(input_data.get('applicant', {}), rules)
            validation_results["applicant_validation"] = applicant_validation
            
            # 2. Validate trademarks
            trademark_validation = self._validate_tm_trademarks(
                input_data.get('Trademarks', {}),
                input_data.get('applicant', {}),
                rules
            )
            validation_results["trademark_validation"] = trademark_validation
            
            # 3. Standard validations if needed
            if input_data.get('directors'):
                directors_validation = self._validate_directors(
                    input_data.get('directors', {}), 
                    rules
                )
                validation_results["director_validation"] = directors_validation
            
            if input_data.get('companyDocuments'):
                company_docs_validation = self._validate_company_documents(
                    input_data.get('companyDocuments', {}),
                    input_data.get('directors', {}),
                    rules
                )
                validation_results["company_documents_validation"] = company_docs_validation
            
            # 4. Apply all relevant rules
            rule_validations = self._apply_tm_rules(
                validation_results,
                rules
            )
            validation_results["rule_validations"] = rule_validations
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Determine overall compliance
            is_compliant = all(
                rule.get("status") == "passed" 
                for rule in rule_validations.values()
            )
            
            # Format standard result
            standard_result = self._format_tm_api_response(validation_results, rule_validations)
            
            # Format detailed result
            detailed_result = {
                "validation_results": validation_results,
                "rule_validations": rule_validations,
                "metadata": {
                    "service_id": service_id,
                    "request_id": request_id,
                    "timestamp": datetime.now().isoformat(),
                    "processing_time": processing_time,
                    "is_compliant": is_compliant
                }
            }
            
            return standard_result, detailed_result
        
        except Exception as e:
            self.logger.error(f"TM validation error: {str(e)}", exc_info=True)
            raise
    def _validate_tm_applicant(self, applicant_data, rules):
        """
        Validate TM applicant information
        """
        validation_result = {
            "is_valid": True,
            "validation_errors": []
        }
        
        # Validate applicant type
        applicant_type = applicant_data.get('applicant_type')
        if not applicant_type:
            validation_result["is_valid"] = False
            validation_result["validation_errors"].append("Missing applicant type")
            return validation_result
        company_name = applicant_data.get("company_name", "")
        # if not company_name:
        #     validation_result["is_valid"] = False
        #     validation_result["validation_errors"].append("Company name is required")
        # Get TM_APPLICANT_TYPE rule
        applicant_type_rule = next(
            (rule for rule in rules if rule.get('rule_id') == 'TM_APPLICANT_TYPE'),
            None
        )
        aadhaar_front = applicant_data.get("aadhaar_front")
        aadhaar_back = applicant_data.get("aadhaar_back")

        extracted_front = self.extraction_service.extract_document_data(aadhaar_front, "aadhar_front") if aadhaar_front else None
        extracted_back = self.extraction_service.extract_document_data(aadhaar_back, "aadhar_back") if aadhaar_back else None

        # Aadhaar presence checks
        if not extracted_front:
            validation_result["is_valid"] = False
            validation_result["validation_errors"].append("Applicant Aadhaar Front is invalid or missing required data.")
        if not extracted_back:
            validation_result["is_valid"] = False
            validation_result["validation_errors"].append("Applicant Aadhaar Back is invalid or missing required data.")

        # Only proceed if both are present
        if extracted_front and extracted_back:
            # Masked check: at least one side must be unmasked
            if extracted_front.get('is_masked', False) and extracted_back.get('is_masked', False):
                validation_result["is_valid"] = False
                validation_result["validation_errors"].append("Both Aadhaar front and back are masked, need at least one unmasked.")

            # Name visible in at least one side
            name_front = extracted_front.get('name')
            name_back = extracted_back.get('name')
            if not (name_front and name_front.strip()) and not (name_back and name_back.strip()):
                validation_result["is_valid"] = False
                validation_result["validation_errors"].append("Applicant name not visible in either Aadhaar front or back.")

        if applicant_type_rule:
            conditions = applicant_type_rule.get('conditions', {})
            valid_types = conditions.get('valid_types', ['Individual', 'Company'])
            
            if applicant_type not in valid_types:
                validation_result["is_valid"] = False
                validation_result["validation_errors"].append(
                    f"Invalid applicant type: {applicant_type}. Must be one of {', '.join(valid_types)}"
                )
        
        # If Company, validate certificate requirements
        if applicant_type == "Company":
            certificate_validation = self._validate_tm_company_certificates(
                applicant_data,
                rules
            )
            # Merge validation results
            if not certificate_validation["is_valid"]:
                validation_result["is_valid"] = False
                validation_result["validation_errors"].extend(
                    certificate_validation["validation_errors"]
                )
            
            # Add certificate validation details
            validation_result["certificate_validation"] = certificate_validation
        
        return validation_result
    # def _validate_tm_applicant(self, applicant_data, rules):
    #     """
    #     Validate TM applicant information
    #     """
    #     validation_result = {
    #         "is_valid": True,
    #         "validation_errors": []
    #     }
        
    #     # Validate applicant type
    #     applicant_type = applicant_data.get('applicant_type')
    #     if not applicant_type:
    #         validation_result["is_valid"] = False
    #         validation_result["validation_errors"].append("Missing applicant type")
    #         return validation_result
        
    #     # Get TM_APPLICANT_TYPE rule
    #     applicant_type_rule = next(
    #         (rule for rule in rules if rule.get('rule_id') == 'TM_APPLICANT_TYPE'),
    #         None
    #     )
    #     aadhaar_front = applicant_data.get("aadhaar_front")
    #     aadhaar_back = applicant_data.get("aadhaar_back")

    #     if aadhaar_front:
    #         extracted_front = self.extraction_service.extract_document_data(aadhaar_front, "aadhar_front")
    #         if not extracted_front :
    #             validation_result["is_valid"] = False
    #             validation_result["validation_errors"].append("Applicant Aadhaar Front is invalid or missing required data.")

    #     if aadhaar_back:
    #         extracted_back = self.extraction_service.extract_document_data(aadhaar_back, "aadhar_back")
    #         if not extracted_back :
    #             validation_result["is_valid"] = False
    #             validation_result["validation_errors"].append("Applicant Aadhaar Back is invalid or missing required data.")
    #     # aadhaar_validation = self._validate_applicant_aadhaar(extracted_front, extracted_back)
    #     # if not aadhaar_validation["is_valid"]:
    #     #     validation_result["is_valid"] = False
    #     #     validation_result["validation_errors"].extend(aadhaar_validation["validation_errors"])
        
    #     if applicant_type_rule:
    #         conditions = applicant_type_rule.get('conditions', {})
    #         valid_types = conditions.get('valid_types', ['Individual', 'Company'])
            
    #         if applicant_type not in valid_types:
    #             validation_result["is_valid"] = False
    #             validation_result["validation_errors"].append(
    #                 f"Invalid applicant type: {applicant_type}. Must be one of {', '.join(valid_types)}"
    #             )
        
    #     # If Company, validate certificate requirements
    #     if applicant_type == "Company":
    #         certificate_validation = self._validate_tm_company_certificates(
    #             applicant_data,
    #             rules
    #         )
    #         # Merge validation results
    #         if not certificate_validation["is_valid"]:
    #             validation_result["is_valid"] = False
    #             validation_result["validation_errors"].extend(
    #                 certificate_validation["validation_errors"]
    #             )
            
    #         # Add certificate validation details
    #         validation_result["certificate_validation"] = certificate_validation
        
    #     return validation_result

    def _validate_tm_company_certificates(self, applicant_data, rules):
        """
        Validate TM company certificates (MSME or DIPP)
        """
        validation_result = {
            "is_valid": True,
            "validation_errors": [],
            "msme_validation": {},
            "dipp_validation": {}
        }
        company_name = applicant_data.get("company_name", "")
        if not company_name:
            validation_result["is_valid"] = False
            validation_result["validation_errors"].append("Company name is required for TM company certificate validation")
        # Get certificate requirements
        documents = applicant_data.get('documents', {})
        compliance = applicant_data.get('compliance', {})
        
        # Get TM_COMPANY_CERTIFICATE rule
        certificate_rule = next(
            (rule for rule in rules if rule.get('rule_id') == 'TM_COMPANY_CERTIFICATE'),
            None
        )
        
        if not certificate_rule:
            return validation_result
        
        conditions = certificate_rule.get('conditions', {})
        
        # Check if certificates are required
        msme_or_dipp_required = conditions.get('msme_or_dipp_required', True)
        company_name_visible_required = conditions.get('company_name_visible_required', True)
        certificate_legible_required = conditions.get('certificate_legible_required', True)
        
        # Get certificate URLs
        msme_cert_url = documents.get('msme_certificate')
        dipp_cert_url = documents.get('dipp_certificate')
        
        # Check if at least one certificate is provided when required
        if msme_or_dipp_required and not (msme_cert_url or dipp_cert_url):
            validation_result["is_valid"] = False
            validation_result["validation_errors"].append(
                "Either MSME or DIPP certificate is required for company applicants"
            )
            return validation_result
        
        # Process MSME certificate if provided
        if msme_cert_url:
            msme_validation = self._validate_certificate(
                msme_cert_url,
                'msme_certificate',
                company_name_visible_required,
                certificate_legible_required,
                expected_company_name=applicant_data.get('company_name')
            )
            
            validation_result["msme_validation"] = msme_validation
            
            if not msme_validation["is_valid"]:
                validation_result["is_valid"] = False
                validation_result["validation_errors"].extend(
                    msme_validation["validation_errors"]
                )
        
        # Process DIPP certificate if provided
        if dipp_cert_url:
            dipp_validation = self._validate_certificate(
                dipp_cert_url,
                'dipp_certificate',
                company_name_visible_required,
                certificate_legible_required,
                expected_company_name=applicant_data.get('company_name')
            )
            
            validation_result["dipp_validation"] = dipp_validation
            
            if not dipp_validation["is_valid"]:
                validation_result["is_valid"] = False
                validation_result["validation_errors"].extend(
                    dipp_validation["validation_errors"]
                )
        
        return validation_result

    def _validate_certificate(self, cert_url, cert_type, company_name_visible_required, certificate_legible_required, expected_company_name=None):
        """
        Validate a specific certificate
        """
        validation_result = {
            "is_valid": True,
            "validation_errors": [],
            "url": cert_url,
            "extracted_data": {}
        }
        
        try:
            # Save base64 to temp file if needed
            if isinstance(cert_url, str) and not (cert_url.startswith("http://") or cert_url.startswith("https://")):
                source = self._save_base64_to_tempfile(cert_url, "pdf")
            else:
                source = cert_url

            # Extract data from certificate
            extracted_data = self.extraction_service.extract_document_data(
                source,
                cert_type
            )
            
            # Store extracted data
            validation_result["extracted_data"] = extracted_data
            
            if not extracted_data:
                validation_result["is_valid"] = False
                validation_result["validation_errors"].append(f"Failed to extract data from {cert_type}")
                return validation_result
            # Check company name visibility
            if company_name_visible_required:
                company_name_visible = extracted_data.get('company_name_visible', False)
                extracted_company_name = (extracted_data.get('company_name') or '').strip().lower()
                expected_company_name = (expected_company_name or '').strip().lower()
                if not company_name_visible:
                    validation_result["is_valid"] = False
                    validation_result["validation_errors"].append(
                        f"Company name not visible in {cert_type}"
                    )
                elif expected_company_name and extracted_company_name and extracted_company_name != expected_company_name:
                    validation_result["is_valid"] = False
                    validation_result["validation_errors"].append(
                        f"Company name mismatch in {cert_type}: expected '{expected_company_name}', found '{extracted_company_name}'"
                    )
            # # Check company name visibility
            # if company_name_visible_required:
            #     company_name_visible = extracted_data.get('company_name_visible', False)
            #     if not company_name_visible:
            #         validation_result["is_valid"] = False
            #         validation_result["validation_errors"].append(
            #             f"Company name not visible in {cert_type}"
            #         )
            
            # Check certificate legibility
            if certificate_legible_required:
                clarity_score = float(extracted_data.get('clarity_score', 0))
                is_legible = extracted_data.get('is_legible', False)
                
                if not is_legible or clarity_score < 0.7:
                    validation_result["is_valid"] = False
                    validation_result["validation_errors"].append(
                        f"{cert_type} is not sufficiently legible"
                    )
        
        except Exception as e:
            self.logger.error(f"Certificate validation error: {str(e)}", exc_info=True)
            validation_result["is_valid"] = False
            validation_result["validation_errors"].append(f"Error validating {cert_type}: {str(e)}")
        
        return validation_result

    def _validate_tm_trademarks(self, trademarks_data, applicant_data, rules):
        """
        Validate trademark information
        """
        validation_result = {
            "is_valid": True,
            "validation_errors": [],
            "trademark_validations": {}
        }
        
        # Get number of trademarks
        trademark_nos = trademarks_data.get('TrademarkNos', 0)
        
        if trademark_nos <= 0:
            validation_result["is_valid"] = False
            validation_result["validation_errors"].append("No trademarks specified")
            return validation_result
        
        # Process each trademark
        for i in range(1, trademark_nos + 1):
            trademark_key = f"Trademark{i}"
            
            if trademark_key not in trademarks_data:
                validation_result["is_valid"] = False
                validation_result["validation_errors"].append(f"Missing {trademark_key} information")
                continue
            
            # Validate individual trademark
            trademark_validation = self._validate_single_trademark(
                trademarks_data[trademark_key],
                applicant_data,
                rules
            )
            
            # Store validation results
            validation_result["trademark_validations"][trademark_key] = trademark_validation
            
            # Update overall validity
            if not trademark_validation["is_valid"]:
                validation_result["is_valid"] = False
                validation_result["validation_errors"].extend([
                    f"{trademark_key}: {error}" for error in trademark_validation["validation_errors"]
                ])
        
        return validation_result

    def _validate_single_trademark(self, trademark_data, applicant_data, rules):
        """
        Validate a single trademark
        """
        validation_result = {
            "is_valid": True,
            "validation_errors": [],
            "verification_docs_validation": {}
        }
        logo_file = trademark_data.get("LogoFile")
        # Validate required fields
        required_fields = ['BrandName', 'Logo', 'AlreadyInUse']
        for field in required_fields:
            if field not in trademark_data:
                validation_result["is_valid"] = False
                validation_result["validation_errors"].append(f"Missing required field: {field}")
        
        if not validation_result["is_valid"]:
            return validation_result
        
        # Check if verification documents are required
        already_in_use = trademark_data.get('AlreadyInUse') == "Yes"
        has_logo = trademark_data.get('Logo') == "Yes"
        brand_name = trademark_data.get('BrandName', '')
        # print("--------------Brand Name:", brand_name)
        # Brand name must not be empty
        # if not brand_name or not brand_name.strip():
        #     validation_result["is_valid"] = False
        #     validation_result["validation_errors"].append("Brand name is required.")

        if has_logo and not logo_file:
            validation_result["is_valid"] = False
            validation_result["validation_errors"].append("Logo is marked as 'Yes' but no logo file was uploaded.")
        if already_in_use:
            # Get TM_TRADEMARK_VERIFICATION rule
            verification_rule = next(
                (rule for rule in rules if rule.get('rule_id') == 'TM_TRADEMARK_VERIFICATION'),
                None
            )
            
            if verification_rule:
                conditions = verification_rule.get('conditions', {})
                min_verification_docs = conditions.get('min_verification_docs', 1)
                
                # Get verification documents
                verification_docs = trademark_data.get('VerificationDocs', {})
                
                # Check if enough verification docs provided
                if not verification_docs or len(verification_docs) < min_verification_docs:
                    validation_result["is_valid"] = False
                    validation_result["validation_errors"].append(
                        f"At least {min_verification_docs} verification document(s) required for trademark already in use"
                    )
                else:
                    # Validate each verification document
                    verification_docs_validation = self._validate_verification_documents(
                        verification_docs,
                        brand_name,
                        has_logo,
                        already_in_use,
                        applicant_data
                    )
                    
                    validation_result["verification_docs_validation"] = verification_docs_validation
                    
                    if not verification_docs_validation["is_valid"]:
                        validation_result["is_valid"] = False
                        validation_result["validation_errors"].extend(
                            verification_docs_validation["validation_errors"]
                        )
                    
        # Check brand name in logo if needed
        if has_logo and logo_file:
            brand_name_in_logo_validation = self._validate_brand_name_in_logo(
                trademark_data,
                applicant_data,
                rules
            )
            # if brand_name_in_logo_validation["status"] == "failed":
            #     validation_result["is_valid"] = False
            #     error_msg = brand_name_in_logo_validation["error_message"]
            #     if error_msg not in validation_result["validation_errors"]:
            #         validation_result["validation_errors"].append(error_msg)
            if brand_name_in_logo_validation["status"] == "failed":
                validation_result["is_valid"] = False
                validation_result["validation_errors"].append(
                    brand_name_in_logo_validation["error_message"]
                )
        
        return validation_result
    def _validate_verification_documents(self, verification_docs, brand_name, has_logo, already_in_use, applicant_data):
        """
        Validate trademark verification documents.
        1. If any doc has a brand name matching the entered brand name, pass.
        2. Else, if any doc has a logo and it matches the uploaded logo, pass.
        3. Else, fail with error.
        """
        validation_result = {
            "is_valid": True,
            "validation_errors": [],
            "document_validations": {}
        }

        # Extract logo features from the uploaded logo file (if present)
        logo_file = applicant_data.get("LogoFile") or applicant_data.get("logo_file")
        logo_features = None
        if has_logo and logo_file:
            try:
                if isinstance(logo_file, str) and not (logo_file.startswith("http://") or logo_file.startswith("https://")):
                    logo_source = self._save_base64_to_tempfile(logo_file, "png")
                else:
                    logo_source = logo_file
                logo_extracted = self.extraction_service.extract_document_data(
                    logo_source,
                    'trademark_verification'
                )
                logo_features = logo_extracted.get('logo_features')  # This should be a hash, embedding, or similar
            except Exception as e:
                self.logger.error(f"Error extracting features from uploaded logo: {str(e)}")

        brand_name_found = False
        logo_match_found = False
        date_found = False
        for doc_key, doc_info in verification_docs.items():
            doc_url = doc_info.get('url', '')
            if not doc_url:
                validation_result["is_valid"] = False
                validation_result["validation_errors"].append(f"Missing URL for {doc_key}")
                continue

            try:
                # Save base64 to temp file if needed
                if isinstance(doc_url, str) and not (doc_url.startswith("http://") or doc_url.startswith("https://")):
                    source = self._save_base64_to_tempfile(doc_url, "pdf")
                else:
                    source = doc_url

                extracted_data = self.extraction_service.extract_document_data(
                    source,
                    'trademark_verification'
                )

                doc_validation = {
                    "is_valid": True,
                    "validation_errors": [],
                    "url": doc_url,
                    "extracted_data": extracted_data
                }

                if not extracted_data:
                    doc_validation["is_valid"] = False
                    doc_validation["validation_errors"].append(f"Failed to extract data from {doc_key}")
                else:
                    document_date = extracted_data.get('document_date')
                    if document_date:
                       date_found = True 
                    # 1. Check for brand name match
                    brand_names_found = extracted_data.get('brand_names_found') or []
                    if any(self._names_match(brand_name, b) for b in brand_names_found if b):
                        brand_name_found = True

                    # 2. If no brand name match, check for logo match
                    if not brand_name_found and has_logo and logo_features:
                        doc_logo_features = extracted_data.get('logo_features')
                        # You need to define how to compare logo_features (hash, embedding, etc.)
                        if doc_logo_features and self._logo_features_match(logo_features, doc_logo_features):
                            logo_match_found = True

                validation_result["document_validations"][doc_key] = doc_validation

            except Exception as e:
                self.logger.error(f"Verification document validation error: {str(e)}", exc_info=True)
                validation_result["document_validations"][doc_key] = {
                    "is_valid": False,
                    "validation_errors": [f"Error validating document: {str(e)}"],
                    "url": doc_url
                }

        # Final check

        if brand_name_found and date_found:
            return validation_result  
        elif logo_match_found and date_found:
            return validation_result  
        else:
            if not date_found:
                validation_result["is_valid"] = False
                validation_result["validation_errors"].append("No document date found in any verification document")
            validation_result["is_valid"] = False
            validation_result["validation_errors"].append(
                f"Neither brand name '{brand_name}' nor matching logo found in any verification document"
            )
            return validation_result

    # Add this helper for logo comparison
    def _logo_features_match(self, features1, features2):
        """
        Compare two logo features (hash, embedding, etc.)
        For hash: return features1 == features2
        For embedding: use cosine similarity or similar metric
        """
        # Example for hash:
        return features1 == features2
    # def _validate_verification_documents(self, verification_docs, brand_name, has_logo, already_in_use, applicant_data):
    #     """
    #     Validate trademark verification documents.
    #     If logo is present and already in use, at least one verification doc must have either the brand name or logo.
    #     """
    #     validation_result = {
    #         "is_valid": True,
    #         "validation_errors": [],
    #         "document_validations": {}
    #     }

    #     # Track if any doc has logo or brand name
    #     logo_or_brand_name_found = False

    #     for doc_key, doc_info in verification_docs.items():
    #         doc_url = doc_info.get('url', '')
    #         if not doc_url:
    #             validation_result["is_valid"] = False
    #             validation_result["validation_errors"].append(f"Missing URL for {doc_key}")
    #             continue

    #         try:
    #             # Save base64 to temp file if needed
    #             if isinstance(doc_url, str) and not (doc_url.startswith("http://") or doc_url.startswith("https://")):
    #                 source = self._save_base64_to_tempfile(doc_url, "pdf")
    #             else:
    #                 source = doc_url

    #             extracted_data = self.extraction_service.extract_document_data(
    #                 source,
    #                 'trademark_verification'
    #             )

    #             doc_validation = {
    #                 "is_valid": True,
    #                 "validation_errors": [],
    #                 "url": doc_url,
    #                 "extracted_data": extracted_data
    #             }

    #             if not extracted_data:
    #                 doc_validation["is_valid"] = False
    #                 doc_validation["validation_errors"].append(f"Failed to extract data from {doc_key}")
    #             else:
    #                 # Check for logo or brand name (only if has_logo is True)
    #                 if has_logo and already_in_use:
    #                     logo_visible = extracted_data.get('logo_visible', False)
    #                     brand_names_found = extracted_data.get('brand_names_found') or []
    #                     brand_name_match = any(
    #                         self._names_match(brand_name, b) for b in brand_names_found if b
    #                     )
    #                     # If either logo or brand name is found, set flag
    #                     if logo_visible or brand_name_match:
    #                         logo_or_brand_name_found = True

    #             validation_result["document_validations"][doc_key] = doc_validation

    #         except Exception as e:
    #             self.logger.error(f"Verification document validation error: {str(e)}", exc_info=True)
    #             validation_result["document_validations"][doc_key] = {
    #                 "is_valid": False,
    #                 "validation_errors": [f"Error validating document: {str(e)}"],
    #                 "url": doc_url
    #             }

    #     # Final check: if logo is required, at least one doc must have logo or brand name
    #     if has_logo and not logo_or_brand_name_found:
    #         validation_result["is_valid"] = False
    #         validation_result["validation_errors"].append(
    #             f"Neither logo nor brand name '{brand_name}' found in any verification document"
    #         )

    #     return validation_result
    # def _validate_verification_documents(self, verification_docs, brand_name, has_logo, applicant_data):
    #     """
    #     Validate trademark verification documents
    #     """
    #     validation_result = {
    #         "is_valid": True,
    #         "validation_errors": [],
    #         "document_validations": {}
    #     }
        
    #     # Get company name
    #     company_name = applicant_data.get('company_name', '')
        
    #     # Variables to track requirements
    #     company_name_found = False
    #     logo_or_brand_name_found = False
        
    #     # Process each verification document
    #     for doc_key, doc_info in verification_docs.items():
    #         doc_url = doc_info.get('url', '')
            
    #         if not doc_url:
    #             validation_result["is_valid"] = False
    #             validation_result["validation_errors"].append(f"Missing URL for {doc_key}")
    #             continue
            
    #         # Extract data from document
    #         try:
    #             # Save base64 to temp file if needed
    #             if isinstance(doc_url, str) and not (doc_url.startswith("http://") or doc_url.startswith("https://")):
    #                 source = self._save_base64_to_tempfile(doc_url, "pdf")
    #             else:
    #                 source = doc_url
    #             extracted_data = self.extraction_service.extract_document_data(
    #                 source,
    #                 'trademark_verification'
    #             )
                
    #             # Validate document
    #             doc_validation = {
    #                 "is_valid": True,
    #                 "validation_errors": [],
    #                 "url": doc_url,
    #                 "extracted_data": extracted_data
    #             }
                
    #             if not extracted_data:
    #                 doc_validation["is_valid"] = False
    #                 doc_validation["validation_errors"].append(f"Failed to extract data from {doc_key}")
    #             else:
    #                 # Check for company name
    #                 if company_name:
    #                     company_name_visible = extracted_data.get('company_name_visible', False)
                        
    #                     if company_name_visible:
    #                         company_name_found = True
                    
    #                 # Check for logo or brand name
    #                 if has_logo:
    #                     logo_visible = extracted_data.get('logo_visible', False)
    #                     brand_names_found = extracted_data.get('brand_names_found') or []
    #                     brand_name_match = any(
    #                         self._names_match(brand_name, b) for b in brand_names_found if b
    #                     )
    #                     extracted_text = extracted_data.get('extracted_text', '').lower()
    #                     brand_name_in_text = self._names_match(brand_name, extracted_text) if extracted_text else False

    #                     brand_name_visible = brand_name_match or brand_name_in_text

    #                     if logo_visible or brand_name_visible:
    #                         logo_or_brand_name_found = True
    #                         extracted_data['brand_name_visible'] = True
    #                     else:
    #                         extracted_data['brand_name_visible'] = False
                       

    #                 if company_name and company_name.lower() in (extracted_data.get('extracted_text', '').lower() or ''):
    #                     company_name_found = True
    #                 if has_logo and (brand_name.lower() in (extracted_data.get('extracted_text', '').lower() or '') or extracted_data.get('logo_visible', False)):
    #                     logo_or_brand_name_found = True
    #             # Store document validation
    #             validation_result["document_validations"][doc_key] = doc_validation
            
    #         except Exception as e:
    #             self.logger.error(f"Verification document validation error: {str(e)}", exc_info=True)
    #             validation_result["document_validations"][doc_key] = {
    #                 "is_valid": False,
    #                 "validation_errors": [f"Error validating document: {str(e)}"],
    #                 "url": doc_url
    #             }
        
    #     # Check overall requirements
    #     if not company_name_found and company_name:
    #         validation_result["is_valid"] = False
    #         validation_result["validation_errors"].append(
    #             "Company name not found in any verification document"
    #         )
        
    #     if has_logo and not logo_or_brand_name_found:
    #         validation_result["is_valid"] = False
    #         validation_result["validation_errors"].append(
    #             f"Neither logo nor brand name '{brand_name}' found in any verification document"
    #         )
        
    #     return validation_result

    def _validate_brand_name_in_logo(self, trademark_data, applicant_data, rules):
        """
        Validate that the brand name is present in the logo file itself
        """
        validation_result = {
            "status": "passed",
            "error_message": None
        }

        has_logo = trademark_data.get('Logo') == "Yes"
        logo_file = trademark_data.get("LogoFile")
        brand_name = trademark_data.get('BrandName', '')

        # if not has_logo:
        #     return validation_result

        # if not logo_file:
        #     validation_result["status"] = "failed"
        #     validation_result["error_message"] = "Logo file not provided for logo validation"
        #     return validation_result

        if not brand_name or not brand_name.strip():
            validation_result["status"] = "failed"
            validation_result["error_message"] = "Brand name is required for logo validation"
            return validation_result

        try:
            # Save base64 to temp file if needed
            if isinstance(logo_file, str) and not (logo_file.startswith("http://") or logo_file.startswith("https://")):
                source = self._save_base64_to_tempfile(logo_file, "png")
            else:
                source = logo_file

            extracted_data = self.extraction_service.extract_document_data(
                source,
                'trademark_verification'
            )
            # print("-----------------------Extracted Data:")
            logo_visible = extracted_data.get('logo_visible', False)
            brand_in_logo = extracted_data.get('brand_name_in_logo', False)
            brand_names_found = extracted_data.get('brand_names_found') or []
            # Require exact (normalized) match with any extracted brand name
            def normalize(name):
                import re
                return re.sub(r'[^\w\s]', '', name.lower()).strip()

            normalized_input = normalize(brand_name)
            matches = [normalize(b) for b in brand_names_found if b]
            if not (logo_visible and normalized_input in matches):
                validation_result["status"] = "failed"
                validation_result["error_message"] = f"Brand name '{brand_name}' not found as exact match in logo file"
        except Exception as e:
            self.logger.error(f"Error checking brand name in logo file: {str(e)}", exc_info=True)
            validation_result["status"] = "failed"
            validation_result["error_message"] = f"Error validating logo file: {str(e)}"

        return validation_result
        #     brand_name_match = any(
        #         self._names_match(brand_name, b) for b in brand_names_found if b
        #     )
        #     # if logo_visible:
        #     #     validation_result["error_message"] = "Logo is visible in the logo file"
        #     if not (logo_visible and (brand_in_logo and brand_name_match)):
        #         validation_result["status"] = "failed"
        #         validation_result["error_message"] = f"Brand name '{brand_name}' not found in logo file"
        # except Exception as e:
        #     self.logger.error(f"Error checking brand name in logo file: {str(e)}", exc_info=True)
        #     validation_result["status"] = "failed"
        #     validation_result["error_message"] = f"Error validating logo file: {str(e)}"

        # return validation_result

    # def _validate_brand_name_in_logo(self, trademark_data, applicant_data, rules):
    #     """
    #     Validate that the brand name is present in the logo itself
    #     """
    #     validation_result = {
    #         "status": "passed",
    #         "error_message": None
    #     }
        
    #     # Skip if logo is not used
    #     has_logo = trademark_data.get('Logo') == "Yes"
    #     if not has_logo:
    #         return validation_result
        
    #     # Get brand name
    #     brand_name = trademark_data.get('BrandName', '')
    #     if not brand_name:
    #         validation_result["status"] = "failed"
    #         validation_result["error_message"] = "Brand name is required for logo validation"
    #         return validation_result
        
    #     # Get verification documents if already in use
    #     already_in_use = trademark_data.get('AlreadyInUse') == "Yes"
    #     if already_in_use:
    #         verification_docs = trademark_data.get('VerificationDocs', {})
            
    #         # Check if any verification document has logo with brand name
    #         brand_name_in_logo_found = False
    #         for doc_key, doc_info in verification_docs.items():
    #             doc_url = doc_info.get('url', '')
                
    #             if not doc_url:
    #                 continue
                
    #             # Extract data from document
    #             try:
    #                 extracted_data = self.extraction_service.extract_document_data(
    #                     doc_url,
    #                     'trademark_verification'
    #                 )
                    
    #                 if extracted_data:
    #                     # Check if logo has brand name
    #                     logo_visible = extracted_data.get('logo_visible', False)
    #                     brand_in_logo = extracted_data.get('brand_name_in_logo', False)
                        
    #                     if logo_visible and brand_in_logo:
    #                         brand_name_in_logo_found = True
    #                         break
                
    #             except Exception as e:
    #                 self.logger.error(f"Error checking brand name in logo: {str(e)}", exc_info=True)
            
    #         if not brand_name_in_logo_found:
    #             validation_result["status"] = "failed"
    #             validation_result["error_message"] = f"Brand name '{brand_name}' not found in any logo"
        
    #     return validation_result

    def _apply_tm_rules(self, validation_results, rules):
        """
        Apply all TM-specific rules
        """
        rule_validations = {}
        
        # TM_APPLICANT_TYPE rule
        applicant_type_rule = next(
            (rule for rule in rules if rule.get('rule_id') == 'TM_APPLICANT_TYPE'),
            None
        )
        
        if applicant_type_rule:
            applicant_validation = validation_results.get('applicant_validation', {})
            rule_validations['tm_applicant_type'] = {
                "status": "passed" if applicant_validation.get('is_valid', False) else "failed",
                "error_message": "; ".join(applicant_validation.get('validation_errors', []))
            }
        
        # TM_COMPANY_CERTIFICATE rule
        company_cert_rule = next(
            (rule for rule in rules if rule.get('rule_id') == 'TM_COMPANY_CERTIFICATE'),
            None
        )
        
        if company_cert_rule:
            applicant_validation = validation_results.get('applicant_validation', {})
            certificate_validation = applicant_validation.get('certificate_validation', {})
            
            rule_validations['tm_company_certificate'] = {
                "status": "passed" if certificate_validation.get('is_valid', True) else "failed",
                "error_message": "; ".join(certificate_validation.get('validation_errors', []))
            }
        
        # TM_TRADEMARK_VERIFICATION rule
        trademark_rule = next(
            (rule for rule in rules if rule.get('rule_id') == 'TM_TRADEMARK_VERIFICATION'),
            None
        )
        
        if trademark_rule:
            trademark_validation = validation_results.get('trademark_validation', {})
            
            rule_validations['tm_trademark_verification'] = {
                "status": "passed" if trademark_validation.get('is_valid', False) else "failed",
                "error_message": "; ".join(trademark_validation.get('validation_errors', []))
            }
        
        # TM_LOGO_BRANDNAME_VALIDATION rule
        logo_brand_rule = next(
            (rule for rule in rules if rule.get('rule_id') == 'TM_LOGO_BRANDNAME_VALIDATION'),
            None
        )
        
        if logo_brand_rule:
            trademark_validation = validation_results.get('trademark_validation', {})
            trademark_validations = trademark_validation.get('trademark_validations', {})
            
            # Check all trademarks with logo
            logo_brand_errors = []
            for tm_key, tm_validation in trademark_validations.items():
                if not tm_validation.get('is_valid', False):
                    for error in tm_validation.get('validation_errors', []):
                        if "logo nor brand name" in error:
                            logo_brand_errors.append(f"{tm_key}: {error}")
            
            rule_validations['tm_logo_brandname_validation'] = {
                "status": "passed" if not logo_brand_errors else "failed",
                "error_message": "; ".join(logo_brand_errors)
            }
        
        # TM_BRAND_NAME_IN_LOGO rule
        brand_name_rule = next(
            (rule for rule in rules if rule.get('rule_id') == 'TM_BRAND_NAME_IN_LOGO'),
            None
        )
        
        if brand_name_rule:
            trademark_validation = validation_results.get('trademark_validation', {})
            trademark_validations = trademark_validation.get('trademark_validations', {})
            
            # Check all trademarks with logo
            brand_name_errors = []
            for tm_key, tm_validation in trademark_validations.items():
                if not tm_validation.get('is_valid', False):
                    for error in tm_validation.get('validation_errors', []):
                        if ("brand name not found in any logo" in error.lower() or "brand name is required" in error.lower()):
                            brand_name_errors.append(f"{tm_key}: {error}")
            
            rule_validations['tm_brand_name_in_logo'] = {
                "status": "passed" if not brand_name_errors else "failed",
                "error_message": "; ".join(brand_name_errors)
            }
        
        # TM_DOCUMENT_LEGIBILITY rule
        legibility_rule = next(
            (rule for rule in rules if rule.get('rule_id') == 'TM_DOCUMENT_LEGIBILITY'),
            None
        )
        
        if legibility_rule:
            # Check all document clarity scores
            legibility_errors = []
            
            # Check applicant documents
            applicant_validation = validation_results.get('applicant_validation', {})
            certificate_validation = applicant_validation.get('certificate_validation', {})
            
            for cert_type in ['msme_validation', 'dipp_validation']:
                cert_data = certificate_validation.get(cert_type, {})
                extracted_data = cert_data.get('extracted_data', {})
                
                if extracted_data:
                    clarity_score = float(extracted_data.get('clarity_score', 0))
                    if clarity_score < 0.7:
                        legibility_errors.append(f"{cert_type} has low clarity: {clarity_score:.2f}")
            
            # Check verification documents
            trademark_validation = validation_results.get('trademark_validation', {})
            trademark_validations = trademark_validation.get('trademark_validations', {})
            
            for tm_key, tm_validation in trademark_validations.items():
                verification_docs = tm_validation.get('verification_docs_validation', {})
                doc_validations = verification_docs.get('document_validations', {})
                
                for doc_key, doc_validation in doc_validations.items():
                    extracted_data = doc_validation.get('extracted_data', {})
                    
                    if extracted_data:
                        clarity_score = float(extracted_data.get('clarity_score', 0))
                        if clarity_score < 0.7:
                            legibility_errors.append(
                                f"{tm_key} - {doc_key} has low clarity: {clarity_score:.2f}"
                            )
            
            rule_validations['tm_document_legibility'] = {
                "status": "passed" if not legibility_errors else "failed",
                "error_message": "; ".join(legibility_errors)
            }
        
        # Add INDIAN_DIRECTOR_AADHAR and INDIAN_DIRECTOR_PAN validation if applicable
        if validation_results.get('director_validation'):
            # These rules reuse your existing methods
            pass
        
        return rule_validations

    def _format_tm_api_response(self, validation_results, rule_validations):
        """
        Format API response for TM validation
        """
        api_response = {
            "validation_rules": rule_validations,
            "document_validation": {
                "applicant": self._format_applicant_validation(validation_results.get('applicant_validation', {})),
                "trademarks": self._format_trademark_validation(validation_results.get('trademark_validation', {}))
            }
        }
        
        # Add standard document validation if available
        if validation_results.get('director_validation'):
            api_response["document_validation"]["directors"] = validation_results.get('director_validation', {})
        
        if validation_results.get('company_documents_validation'):
            api_response["document_validation"]["companyDocuments"] = validation_results.get('company_documents_validation', {})
        
        return api_response

    def _format_applicant_validation(self, applicant_validation):
        """
        Format applicant validation results
        """
        formatted_result = {
            "status": "Valid" if applicant_validation.get('is_valid', False) else "Not Valid",
            "error_messages": applicant_validation.get('validation_errors', [])
        }
        
        # Add certificate validation if available
        certificate_validation = applicant_validation.get('certificate_validation', {})
        
        if certificate_validation:
            formatted_result["certificates"] = {}
            
            for cert_type in ['msme_validation', 'dipp_validation']:
                cert_data = certificate_validation.get(cert_type, {})
                
                if cert_data:
                    formatted_result["certificates"][cert_type.replace('_validation', '')] = {
                        "status": "Valid" if cert_data.get('is_valid', False) else "Not Valid",
                        "error_messages": cert_data.get('validation_errors', [])
                    }
        
        return formatted_result

    def _format_trademark_validation(self, trademark_validation):
        """
        Format trademark validation results
        """
        formatted_result = {
            "status": "Valid" if trademark_validation.get('is_valid', False) else "Not Valid",
            "error_messages": trademark_validation.get('validation_errors', []),
            "trademarks": {}
        }
        
        # Format individual trademark validations
        trademark_validations = trademark_validation.get('trademark_validations', {})
        
        for tm_key, tm_validation in trademark_validations.items():
            formatted_result["trademarks"][tm_key] = {
                "status": "Valid" if tm_validation.get('is_valid', False) else "Not Valid",
                "error_messages": tm_validation.get('validation_errors', []),
                "verification_documents": {}
            }
            
            # Format verification documents
            verification_docs = tm_validation.get('verification_docs_validation', {})
            doc_validations = verification_docs.get('document_validations', {})
            
            for doc_key, doc_validation in doc_validations.items():
                formatted_result["trademarks"][tm_key]["verification_documents"][doc_key] = {
                    "status": "Valid" if doc_validation.get('is_valid', False) else "Not Valid",
                    "error_messages": doc_validation.get('validation_errors', [])
                }
        
        return formatted_result