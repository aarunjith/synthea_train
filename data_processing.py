import json
import pandas as pd
from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime
from collections import Counter


def calculate_age(birth_date: str) -> int:
    """Calculate age from birth date."""
    if not birth_date:
        return None
    try:
        dob = datetime.strptime(birth_date, "%Y-%m-%d")
        today = datetime.now()
        age = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
        return age
    except:
        return None


def extract_demographics(resource: Dict) -> Dict:
    """Extract comprehensive demographic information from a Patient resource."""
    demographics = {
        "patient_id": resource.get("id"),
        "gender": resource.get("gender"),
        "birth_date": resource.get("birthDate"),
        "age": calculate_age(resource.get("birthDate")),
        "race": None,
        "ethnicity": None,
        "marital_status": resource.get("maritalStatus", {}).get("text"),
        "language": None,
        "deceased": resource.get("deceasedBoolean", False),
        "multiple_birth": resource.get("multipleBirth", {}).get("boolean"),
        "birth_place": None,
    }

    # Extract race
    race_extension = next(
        (
            ext
            for ext in resource.get("extension", [])
            if ext.get("url")
            == "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race"
        ),
        None,
    )
    if race_extension:
        race_coding = next(
            (
                ext.get("valueCoding", {})
                for ext in race_extension.get("extension", [])
                if ext.get("url") == "ombCategory"
            ),
            {},
        )
        demographics["race"] = race_coding.get("display")
        demographics["race_code"] = race_coding.get("code")

    # Extract ethnicity
    ethnicity_extension = next(
        (
            ext
            for ext in resource.get("extension", [])
            if ext.get("url")
            == "http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity"
        ),
        None,
    )
    if ethnicity_extension:
        ethnicity_coding = next(
            (
                ext.get("valueCoding", {})
                for ext in ethnicity_extension.get("extension", [])
                if ext.get("url") == "ombCategory"
            ),
            {},
        )
        demographics["ethnicity"] = ethnicity_coding.get("display")
        demographics["ethnicity_code"] = ethnicity_coding.get("code")

    # Extract communication language
    communication = resource.get("communication", [{}])[0]
    if communication:
        language = communication.get("language", {})
        if isinstance(language, dict):
            demographics["language"] = language.get("text")

    # Extract birth place if available
    birthplace_extension = next(
        (
            ext
            for ext in resource.get("extension", [])
            if ext.get("url")
            == "http://hl7.org/fhir/StructureDefinition/patient-birthPlace"
        ),
        None,
    )
    if birthplace_extension:
        demographics["birth_place"] = birthplace_extension.get("valueString")

    return demographics


def extract_medication_info(resource: Dict) -> Dict:
    """Extract comprehensive medication information from a MedicationRequest resource."""
    med_info = {
        "patient_id": resource.get("subject", {})
        .get("reference", "")
        .replace("Patient/", ""),
        "medication_request_id": resource.get("id"),
        "status": resource.get("status"),
        "intent": resource.get("intent"),
        "priority": resource.get("priority"),
        "authorized_on": resource.get("authoredOn"),
        "medication_code": None,
        "medication_system": None,
        "medication_display": None,
        "dosage_text": None,
        "dosage_timing": None,
        "route": None,
        "method": None,
    }

    # Extract medication coding
    medication_concept = resource.get("medicationCodeableConcept", {})
    if medication_concept:
        coding = medication_concept.get("coding", [{}])[0]
        med_info["medication_code"] = coding.get("code")
        med_info["medication_system"] = coding.get("system")
        med_info["medication_display"] = coding.get("display")

    # Extract dosage information
    dosage = resource.get("dosageInstruction", [{}])[0]
    if dosage:
        med_info["dosage_text"] = dosage.get("text")
        med_info["dosage_timing"] = str(dosage.get("timing", {}))
        route = dosage.get("route", {})
        if isinstance(route, dict):
            med_info["route"] = route.get("text")
        method = dosage.get("method", {})
        if isinstance(method, dict):
            med_info["method"] = method.get("text")

    return med_info


def extract_encounter_info(resource: Dict) -> Dict:
    """Extract comprehensive encounter information from an Encounter resource."""
    encounter_info = {
        "patient_id": resource.get("subject", {})
        .get("reference", "")
        .replace("Patient/", ""),
        "encounter_id": resource.get("id"),
        "status": resource.get("status"),
        "class": resource.get("class", {}).get("code"),
        "type": None,
        "service_type": None,
        "priority": None,
        "start_date": resource.get("period", {}).get("start"),
        "end_date": resource.get("period", {}).get("end"),
        "length_of_stay": None,
        "reason_code": None,
        "reason_display": None,
        "icd_code": None,
        "icd_display": None,
        "cpt_code": None,
        "cpt_display": None,
    }

    # Calculate length of stay if both start and end dates are available
    if encounter_info["start_date"] and encounter_info["end_date"]:
        try:
            start = datetime.strptime(encounter_info["start_date"], "%Y-%m-%d")
            end = datetime.strptime(encounter_info["end_date"], "%Y-%m-%d")
            encounter_info["length_of_stay"] = (end - start).days
        except:
            pass

    # Extract encounter type
    types = resource.get("type", [])
    if types and isinstance(types, list):
        type_coding = types[0].get("coding", [{}])[0]
        encounter_info["cpt_code"] = type_coding.get("code")
        encounter_info["cpt_display"] = type_coding.get("display")

    # Extract service type
    service_type = resource.get("serviceType", {})
    if service_type:
        encounter_info["service_type"] = service_type.get("text")

    # Extract priority
    priority = resource.get("priority", {})
    if priority:
        encounter_info["priority"] = priority.get("text")

    # Extract diagnoses
    diagnoses = resource.get("diagnosis", [])
    if diagnoses and isinstance(diagnoses, list):
        diagnosis = diagnoses[0]
        condition = diagnosis.get("condition", {})
        if isinstance(condition, dict):
            coding = condition.get("coding", [{}])[0]
            encounter_info["icd_code"] = coding.get("code")
            encounter_info["icd_display"] = coding.get("display")

    # Extract reason
    reason = resource.get("reasonCode", [{}])[0]
    if reason:
        coding = reason.get("coding", [{}])[0]
        encounter_info["reason_code"] = coding.get("code")
        encounter_info["reason_display"] = coding.get("display")

    return encounter_info


def extract_claim_info(resource: Dict) -> Dict:
    """Extract specific claim information from a Claim resource."""
    # Extract service dates for calculating days of service
    billable_period = resource.get("billablePeriod", {})
    start_date = billable_period.get("start")
    end_date = billable_period.get("end")

    # Calculate days of service
    days_of_service = 1  # Default minimum of 1 day
    if start_date and end_date:
        try:
            start = datetime.strptime(start_date.split("T")[0], "%Y-%m-%d")
            end = datetime.strptime(end_date.split("T")[0], "%Y-%m-%d")
            days = (end - start).days
            days_of_service = max(
                1, days + 1
            )  # Add 1 to include both start and end days
        except:
            pass

    claim_info = {
        "patient_id": None,
        "claim_id": resource.get("id"),
        "status": resource.get("status"),
        "claim_type": None,
        "priority": None,
        "days_of_service": days_of_service,
        "product_service_display": None,
        "claim_amount": None,
        "encounter_reference": None,
        "created_date": resource.get("created"),  # Add created date
    }

    # Extract patient ID from either subject or patient reference
    patient_ref = resource.get("subject", {}) or resource.get("patient", {})
    if isinstance(patient_ref, dict):
        ref = patient_ref.get("reference", "")
        # Handle both Patient/ and urn:uuid: prefixes
        claim_info["patient_id"] = ref.replace("Patient/", "").replace("urn:uuid:", "")

    # Extract claim type
    claim_type = resource.get("type", {})
    if isinstance(claim_type, dict):
        coding = claim_type.get("coding", [{}])[0]
        claim_info["claim_type"] = coding.get("display") or coding.get("code")

    # Extract priority
    priority = resource.get("priority", {})
    if isinstance(priority, dict):
        coding = priority.get("coding", [{}])[0]
        claim_info["priority"] = coding.get("display") or coding.get("code")

    # Extract product or service information from items
    items = resource.get("item", [])
    if items and isinstance(items, list):
        item = items[0]  # Get first item
        product_service = item.get("productOrService", {})
        if isinstance(product_service, dict):
            coding = product_service.get("coding", [{}])[0]
            claim_info["product_service_display"] = coding.get("display")

        # Extract encounter reference
        encounters = item.get("encounter", [])
        if encounters and isinstance(encounters, list):
            encounter_ref = encounters[0].get("reference", "")
            claim_info["encounter_reference"] = encounter_ref.replace("urn:uuid:", "")

    # Extract claim amount
    total = resource.get("total", {})
    if isinstance(total, dict):
        claim_info["claim_amount"] = total.get("value")

    return claim_info


def extract_eob_info(resource: Dict) -> Dict:
    """Extract specific EOB information from an ExplanationOfBenefit resource."""
    eob_info = {
        "claim_reference": resource.get("claim", {})
        .get("reference", "")
        .replace("urn:uuid:", ""),
        "eob_outcome": resource.get("outcome"),
        "submitted_amount": None,
    }

    # Extract submitted amount
    totals = resource.get("total", [])
    for total in totals:
        category = total.get("category", {}).get("coding", [{}])[0]
        if category.get("code") == "submitted":
            eob_info["submitted_amount"] = total.get("amount", {}).get("value")

    return eob_info


def format_date(date_str: str) -> str:
    """Format date string to YYYY-MM-DD."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str.split("T")[0], "%Y-%m-%d").strftime(
            "%Y-%m-%d"
        )
    except:
        return None


def parse_fhir_bundle(file_path: str) -> Dict[str, pd.DataFrame]:
    """
    Parse a FHIR Bundle JSON file into separate DataFrames for each resource type.

    Args:
        file_path (str): Path to the FHIR JSON file

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing DataFrames for each resource type
    """
    print(f"Reading file: {file_path}")
    with open(file_path, "r") as f:
        data = json.load(f)

    print("Checking if it's a FHIR Bundle...")
    if data.get("resourceType") != "Bundle":
        raise ValueError("The provided JSON is not a FHIR Bundle")

    entries = data.get("entry", [])
    print(f"Found {len(entries)} entries in the bundle")

    # Initialize lists to store different types of resources
    patients = []
    medications = []
    encounters = []
    claims = []
    eobs = []  # New list for EOBs

    # Process each resource
    for entry in entries:
        resource = entry.get("resource", {})
        resource_type = resource.get("resourceType")

        if resource_type == "Patient":
            patients.append(extract_demographics(resource))
        elif resource_type == "MedicationRequest":
            medications.append(extract_medication_info(resource))
        elif resource_type == "Encounter":
            encounters.append(extract_encounter_info(resource))
        elif resource_type == "Claim":
            claims.append(extract_claim_info(resource))
        elif resource_type == "ExplanationOfBenefit":
            eobs.append(extract_eob_info(resource))

    # Create DataFrames
    dfs = {}

    # Process patients DataFrame
    if patients:
        dfs["patients"] = pd.DataFrame(patients)
        # Format birth date
        if "birth_date" in dfs["patients"].columns:
            dfs["patients"]["birth_date"] = dfs["patients"]["birth_date"].apply(
                format_date
            )

    # Process medications DataFrame
    if medications:
        dfs["medications"] = pd.DataFrame(medications)
        if "authorized_on" in dfs["medications"].columns:
            dfs["medications"]["authorized_on"] = dfs["medications"][
                "authorized_on"
            ].apply(format_date)
            dfs["medications"] = dfs["medications"].sort_values("authorized_on")

    # Process encounters DataFrame
    if encounters:
        dfs["encounters"] = pd.DataFrame(encounters)
        if "start_date" in dfs["encounters"].columns:
            dfs["encounters"]["start_date"] = dfs["encounters"]["start_date"].apply(
                format_date
            )
            dfs["encounters"]["end_date"] = dfs["encounters"]["end_date"].apply(
                format_date
            )
            dfs["encounters"] = dfs["encounters"].sort_values("start_date")

    # Process claims DataFrame and merge with encounters, EOBs, and patient info
    if claims:
        claims_df = pd.DataFrame(claims)
        encounters_df = pd.DataFrame(encounters)
        eobs_df = pd.DataFrame(eobs) if eobs else pd.DataFrame()
        patients_df = pd.DataFrame(patients)

        # Format created date
        if "created_date" in claims_df.columns:
            claims_df["created_date"] = claims_df["created_date"].apply(format_date)

        # Prepare encounters DataFrame for merging
        encounter_details = encounters_df[
            ["encounter_id", "class", "reason_display"]
        ].copy()
        encounter_details.columns = [
            "encounter_reference",
            "encounter_class",
            "encounter_reason",
        ]

        # Prepare patient demographics for merging
        patient_demographics = patients_df[
            [
                "patient_id",
                "gender",
                "birth_date",
                "age",
                "race",
                "ethnicity",
                "marital_status",
                "language",
                "deceased",
            ]
        ].copy()

        # Merge claims with encounter details
        claims_df = claims_df.merge(
            encounter_details, on="encounter_reference", how="left"
        )

        # Merge with EOB information if available
        if not eobs_df.empty:
            claims_df = claims_df.merge(
                eobs_df, left_on="claim_id", right_on="claim_reference", how="left"
            )

        # Merge with patient demographics
        claims_df = claims_df.merge(patient_demographics, on="patient_id", how="left")

        # Remove ID columns and clean up
        columns_to_drop = [
            "patient_id",
            "claim_id",
            "encounter_reference",
            "claim_reference",
            "medication_request_id",
            "race_code",
            "ethnicity_code",
        ]
        for col in columns_to_drop:
            if col in claims_df.columns:
                claims_df = claims_df.drop(col, axis=1)

        # Sort by created date
        if "created_date" in claims_df.columns:
            claims_df = claims_df.sort_values("created_date")

        dfs["claims"] = claims_df

    # Print summary
    print("\nDataFrames created:")
    for key, df in dfs.items():
        print(f"{key}: {df.shape[0]} rows, {df.shape[1]} columns")
        if key != "patients":  # Show date range for time-based DataFrames
            date_col = {
                "medications": "authorized_on",
                "encounters": "start_date",
                "claims": "created_date",
            }[key]
            if not df.empty and date_col in df.columns:
                print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")

    return dfs


if __name__ == "__main__":
    sample_file = "sample.json"
    if Path(sample_file).exists():
        print(f"Processing {sample_file}...")
        dfs = parse_fhir_bundle(sample_file)

        # Save claims DataFrame to a tab-separated file
        if "claims" in dfs:
            output_file = "claims_final.tsv"
            dfs["claims"].to_csv(output_file, sep="\t", index=False)
            print(f"\nSaved claims to {output_file}")
            print(f"Columns: {dfs['claims'].columns.tolist()}")
            print("\nFirst few rows:")
            print(dfs["claims"].head())
    else:
        print(f"Error: {sample_file} not found")
