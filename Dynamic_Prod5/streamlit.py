import streamlit as st
import base64
import json
from services.validation_service import DocumentValidationService
from api.document_validation_api import DocumentValidationAPI

import os

# Load from secrets (Streamlit deployment)
if "ELASTICSEARCH_PASSWORD" in st.secrets:
    os.environ["ELASTICSEARCH_HOST"] = st.secrets["ELASTICSEARCH_HOST"]
    os.environ["ELASTICSEARCH_USERNAME"] = st.secrets["ELASTICSEARCH_USERNAME"]
    os.environ["ELASTICSEARCH_PASSWORD"] = st.secrets["ELASTICSEARCH_PASSWORD"]
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.title("Document Validation UI")

validation_api = DocumentValidationAPI()

def encode_file(file):
    if file is None:
        return None
    try:
        return base64.b64encode(file.read()).decode("utf-8")
    except Exception as e:
        return None

def display_results(response_data, _):
    st.subheader("📊 Validation Overview")
    col1, col2, col3 = st.columns(3)

    total_rules = len(response_data.get('validation_rules', {}))
    failed_rules = sum(1 for rule in response_data.get('validation_rules', {}).values()
                      if rule.get('status') == 'failed')

    with col1:
        st.metric("Total Validation Rules", total_rules)
    with col2:
        st.metric("Failed Rules", failed_rules, delta_color="inverse")
    with col3:
        overall_status = "Passed ✅" if failed_rules == 0 else "Failed ❌"
        st.metric("Overall Status", overall_status)

    st.subheader("🔍 Validation Rules Check")
    passed_rules = []
    failed_rules_list = []

    for rule_name, details in response_data.get('validation_rules', {}).items():
        if details.get('status') == 'passed':
            passed_rules.append(f"✅ {rule_name.replace('_', ' ').title()}")
        else:
            failed_rules_list.append(f"❌ {rule_name.replace('_', ' ').title()}: {details.get('error_message', '')}")

    if failed_rules_list:
        with st.expander("❌ Failed Validation Rules", expanded=True):
            for item in failed_rules_list:
                st.write(item)

    if passed_rules:
        with st.expander("✅ Passed Validation Rules", expanded=True):
            for item in passed_rules:
                st.write(item)

    # TM Applicant and Trademarks Section
    if "applicant" in response_data.get("document_validation", {}):
        st.subheader("👤 TM Applicant Status")
        applicant = response_data["document_validation"]["applicant"]
        st.write(f"Status: {applicant.get('status', 'Unknown')}")
        if applicant.get("error_messages"):
            for err in applicant["error_messages"]:
                st.error(f"• {err}")

    if "trademarks" in response_data.get("document_validation", {}):
        st.subheader("™️ Trademark(s) Status")
        trademarks = response_data["document_validation"]["trademarks"]
        if isinstance(trademarks, dict):
            st.write(f"Status: {trademarks.get('status', 'Unknown')}")
            if trademarks.get("error_messages"):
                for err in trademarks["error_messages"]:
                    st.error(f"• {err}")
            # Show each trademark's validation if present
            if "trademarks" in trademarks:
                for tm_key, tm_val in trademarks["trademarks"].items():
                    with st.expander(f"{tm_key} Validation", expanded=False):
                        st.write(f"Status: {tm_val.get('status', 'Unknown')}")
                        if tm_val.get("error_messages"):
                            for err in tm_val["error_messages"]:
                                st.error(f"• {err}")

    st.subheader("👤 Director Document Status")
    directors = response_data.get('document_validation', {}).get('directors', {})
    for director_name, details in directors.items():
        with st.expander(f"{director_name.replace('_', ' ').title()} Documents", expanded=True):
            cols = st.columns(2)
            doc_statuses = []
            expected_docs = [
                "aadharCardFront", "aadharCardBack", "panCard", "passportPhoto",
                "address_proof", "signature", "passport", "drivingLicense"
            ]
            actual_docs = details.get('documents', {})
            if details.get("nationality", "").lower() == "foreign":
                st.info("Passport or Driving License must be provided for foreign directors. Aadhaar is not applicable.")
            else:
                st.info("Aadhaar is required for Indian directors. Passport/Driving License is not mandatory.")

            for doc_type in expected_docs:
                display_name = doc_type.replace('_', ' ').replace('Card', ' Card').title()
                doc_details = actual_docs.get(doc_type, {})
                status = doc_details.get('status', 'Not Uploaded')
                reason = doc_details.get('reason')
                errors = doc_details.get('error_messages', [])

                if status.lower() == 'valid':
                    doc_statuses.append(f"✅ {display_name}")
                else:
                    if reason:
                        doc_statuses.append(f"❌ {display_name}\n• {reason}")
                    elif errors:
                        error_list = "\n".join([f"\u2022 {err}" for err in errors])
                        doc_statuses.append(f"❌ {display_name}\n{error_list}")
                    elif status == 'Not Uploaded':
                        doc_statuses.append(f"❌ {display_name}\n• Document not uploaded")
                    else:
                        doc_statuses.append(f"❌ {display_name}\n• Validation failed")

            for i, status in enumerate(doc_statuses):
                cols[i % 2].write(status)

    st.subheader("🏢 Company Documents Status")
    company_docs = _.get('document_validation', {}).get('companyDocuments', {})
    doc_display_names = {
        "addressProof": "Address Proof",
        "noc": "NOC (No Objection Certificate)"
    }
    for doc_type in ["addressProof", "noc"]:
        if doc_type in company_docs:
            details = company_docs.get(doc_type, {})
            status = details.get('status', 'Unknown')
            errors = details.get('error_messages', [])
            reason = details.get('reason', None)
            display_name = doc_display_names.get(doc_type, doc_type.replace('_', ' ').title())
            if status.lower() == 'valid':
                st.success(f"✅ {display_name}")
            else:
                if reason:
                    st.error(f"❌ {display_name}\n• {reason}")
                elif errors:
                    error_list = "\n".join([f"• {err}" for err in errors])
                    st.error(f"❌ {display_name}\n{error_list}")
                else:
                    st.error(f"❌ {display_name}\nValidation failed")

    st.subheader("📥 Download Results")
    st.download_button(
        label="📁 Download JSON",
        data=json.dumps(response_data, indent=2),
        file_name="validation_results.json",
        mime="application/json"
    )

# --- UI Input Section ---
service_id = st.text_input("Service ID", value="1")
request_id = st.text_input("Request ID", value="req-12345")

# TM Service UI
if service_id == "8":
    st.header("Trademark (TM) Service Input")
    request_id = st.text_input("TM Request ID", value="tm-req-001")
    applicant_type = st.selectbox("Applicant Type", options=["Individual", "Company"])
    applicant_name = st.text_input("Applicant Name")
    company_name = st.text_input("Company Name")
    aadhaar_front = encode_file(st.file_uploader("Applicant Aadhaar Front"))
    aadhaar_back = encode_file(st.file_uploader("Applicant Aadhaar Back"))
    msme_certificate = None
    dipp_certificate = None
    if applicant_type == "Company":
        msme_certificate = encode_file(st.file_uploader("MSME Certificate"))
        dipp_certificate = encode_file(st.file_uploader("DIPP Certificate"))

    applicant = {
        "applicant_type": applicant_type,
        "applicant_name": applicant_name,
        "company_name": company_name,
        "aadhaar_front": aadhaar_front,
        "aadhaar_back": aadhaar_back,
        "documents": {
            "msme_certificate": msme_certificate,
            "dipp_certificate": dipp_certificate
        }
    }

    trademark_nos = st.number_input("Number of Trademarks", min_value=1, max_value=10, value=1)
    trademarks = {"TrademarkNos": trademark_nos}
    for i in range(1, trademark_nos + 1):
        st.subheader(f"Trademark {i}")
        brand_name = st.text_input(f"Brand Name for Trademark {i}", key=f"brand_{i}")
        logo = st.selectbox(f"Logo Provided for Trademark {i}", options=["Yes", "No"], key=f"logo_{i}")
        already_in_use = st.selectbox(f"Already In Use for Trademark {i}", options=["Yes", "No"], key=f"aiu_{i}")
        verification_docs = {}
        logo_file = None
        # Always allow uploading a logo as a verification doc if logo is "Yes"
        if logo == "Yes":
            logo_file = encode_file(st.file_uploader(f"Upload Logo File as Verification Doc for Trademark {i}", key=f"logo_file_{i}"))

        if already_in_use == "Yes":
            num_ver_docs = st.number_input(f"Number of Additional Verification Docs for Trademark {i}", min_value=0, max_value=5, value=0, key=f"vernum_{i}")
            for j in range(1, num_ver_docs + 1):
                ver_doc_url = encode_file(st.file_uploader(f"Verification Doc {j} for Trademark {i}", key=f"verdoc_{i}_{j}"))
                if ver_doc_url:
                    verification_docs[f"ver_doc_{j}"] = {"url": ver_doc_url}

        trademarks[f"Trademark{i}"] = {
            "BrandName": brand_name,
            "Logo": logo,
            "LogoFile": logo_file,  # <-- separate field
            "AlreadyInUse": already_in_use,
            "VerificationDocs": verification_docs if (already_in_use == "Yes" or logo == "Yes") else {}
        }

    payload = {
        "service_id": service_id,
        "request_id": request_id,
        "applicant": applicant,
        "Trademarks": trademarks
    }

    if st.button("Validate TM Documents"):
        try:
            api_response, _ = validation_api.validate_document(payload)
            st.success("✅ TM Validation Completed Successfully!")
            display_results(api_response, _)
            with st.expander("Show Raw Validation Response"):
                st.json(api_response)
        except Exception as e:
            st.error(f"🚨 Validation Error: {str(e)}")

else:
    num_directors = st.slider("Number of Directors", min_value=2, max_value=5, value=2)
    directors = {}
    for i in range(num_directors):
        st.subheader(f"Director {i+1}")
        nationality = st.selectbox(f"Nationality for Director {i+1}", options=["Indian", "Foreign"], key=f"nat_{i}")
        authorised = st.selectbox(f"Authorised for Director {i+1}", options=["Yes", "No"], key=f"auth_{i}")
        st.write("Upload documents for this director:")
        if nationality == "Foreign":
            st.info("Note: For foreign directors, Passport or Driving License is mandatory. Aadhaar is not required.")
        else:
            st.info("Note: For Indian directors, Aadhaar is required. Passport/Driving License is optional.")

        directors[f"director_{i+1}"] = {
            "nationality": nationality,
            "authorised": authorised,
            "documents": {
                "aadharCardFront": encode_file(st.file_uploader("Aadhar Front", key=f"aadharFront_{i}")),
                "aadharCardBack": encode_file(st.file_uploader("Aadhar Back", key=f"aadharBack_{i}")),
                "panCard": encode_file(st.file_uploader("PAN Card", key=f"pan_{i}")),
                "passportPhoto": encode_file(st.file_uploader("Passport Photo", key=f"passportPhoto_{i}")),
                "address_proof": encode_file(st.file_uploader("Address Proof", key=f"addressProof_{i}")),
                "signature": encode_file(st.file_uploader("Signature", key=f"signature_{i}")),
                "passport": encode_file(st.file_uploader("Passport (Foreign)", key=f"passport_{i}")),
                "drivingLicense": encode_file(st.file_uploader("Driving License", key=f"drivingLicense_{i}"))
            }
        }

    st.subheader("Company Documents")
    address_proof_type = st.selectbox("Select Address Proof Type", options=["Electricity Bill", "NOC", "Gas Bill"])
    addressProof = encode_file(st.file_uploader("Company Address Proof"))
    noc = encode_file(st.file_uploader("NOC Document"))

    if st.button("Validate Documents"):
        payload = {
            "service_id": service_id,
            "request_id": request_id,
            "directors": directors,
            "companyDocuments": {
                "address_proof_type": address_proof_type,
                "addressProof": addressProof,
                "noc": noc
            }
        }
        try:
            api_response, _ = validation_api.validate_document(payload)
            st.success("✅ Validation Completed Successfully!")
            display_results(api_response, _)
            with st.expander("Show Raw Validation Response"):
                st.json(api_response)
        except Exception as e:
            st.error(f"🚨 Validation Error: {str(e)}")