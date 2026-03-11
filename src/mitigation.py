def apply_mitigation(record):

    required_fields = ["id", "gender", "race", "prediction"]
    for field in required_fields:
        if field not in record:
            raise ValueError(f"Missing field: {field}")

    gender = record["gender"].lower()
    race = record["race"].lower()
    prediction = record["prediction"]

    # store original prediction
    record["original_prediction"] = prediction

    mitigation_applied = False

    # mitigation rule
    if (gender == "female" or race == "minority") and prediction == 0:
        record["prediction"] = 1
        mitigation_applied = True
        record["mitigation_reason"] = "post_processing_correction"

    record["mitigation_applied"] = mitigation_applied

    return record