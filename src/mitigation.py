# def apply_mitigation(record):

#     required_fields = ["id", "gender", "race", "prediction"]
#     for field in required_fields:
#         if field not in record:
#             raise ValueError(f"Missing field: {field}")

#     gender = record["gender"].lower()
#     race = record["race"].lower()
#     prediction = record["prediction"]

#     # store original prediction
#     record["original_prediction"] = prediction

#     mitigation_applied = False

#     # mitigation rule
#     if (gender == "female" or race == "minority") and prediction == 0:
#         record["prediction"] = 1
#         mitigation_applied = True
#         record["mitigation_reason"] = "post_processing_correction"

#     record["mitigation_applied"] = mitigation_applied

#     return record


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

    # define groups
    disadvantaged = False
    majority = False

    if gender == "female" or race == "minority":
        disadvantaged = True

    if gender == "male" and race == "majority":
        majority = True

    # ---------- mitigation logic ----------

    # case 1: disadvantaged group with negative prediction
    if disadvantaged and prediction == 0:
        record["prediction"] = 1
        mitigation_applied = True
        record["mitigation_reason"] = "increase_positive_for_disadvantaged"

    # case 2: majority group with positive prediction (optional balancing)
    elif majority and prediction == 1:
        # reduce advantage slightly
        record["prediction"] = 1   # keep same or change to 0 if needed
        record["mitigation_reason"] = "checked_majority_bias"

    # store group type
    if disadvantaged:
        record["group_type"] = "disadvantaged"
    elif majority:
        record["group_type"] = "majority"
    else:
        record["group_type"] = "other"

    record["mitigation_applied"] = mitigation_applied

    return record