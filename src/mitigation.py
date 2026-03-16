
# def apply_mitigation(record, stats):

#     required_fields = ["id", "gender", "race", "prediction"]
#     for field in required_fields:
#         if field not in record:
#             raise ValueError(f"Missing field: {field}")

#     gender = record["gender"].lower()
#     race = record["race"].lower()
#     prediction = record["prediction"]

#     record["original_prediction"] = prediction

#     mitigation_applied = False

#     minority_rate = stats["minority_positive_rate"]
#     majority_rate = stats["majority_positive_rate"]

#     disadvantaged = (gender == "female" or race == "minority")
#     majority = (gender == "male" and race == "majority")

#     # ---------- FAIRNESS CORRECTION ----------

#     # minority getting fewer positives → increase
#     if disadvantaged and prediction == 0:
#         if minority_rate < majority_rate:
#             record["prediction"] = 1
#             mitigation_applied = True
#             record["mitigation_reason"] = "increase_minority_positive"

#     # majority getting too many positives → reduce
#     elif majority and prediction == 1:
#         if majority_rate > minority_rate:
#             record["prediction"] = 0
#             mitigation_applied = True
#             record["mitigation_reason"] = "reduce_majority_positive"

#     record["mitigation_applied"] = mitigation_applied

#     return record








import random


def apply_mitigation(records):
    """
    records: list of dict
    each record must contain:
    id, gender, race, prediction (0/1)
    """

    # -----------------------------
    # 1. Compute statistics
    # -----------------------------

    gender_stats = {}
    race_stats = {}

    for r in records:

        g = r["gender"].lower()
        rc = r["race"].lower()
        p = r["prediction"]

        # gender stats
        if g not in gender_stats:
            gender_stats[g] = {"total": 0, "positive": 0}

        gender_stats[g]["total"] += 1
        if p == 1:
            gender_stats[g]["positive"] += 1

        # race stats
        if rc not in race_stats:
            race_stats[rc] = {"total": 0, "positive": 0}

        race_stats[rc]["total"] += 1
        if p == 1:
            race_stats[rc]["positive"] += 1

    # -----------------------------
    # 2. Compute positive rates
    # -----------------------------

    gender_rates = {}
    race_rates = {}

    for g, s in gender_stats.items():
        gender_rates[g] = s["positive"] / s["total"]

    for rc, s in race_stats.items():
        race_rates[rc] = s["positive"] / s["total"]

    # -----------------------------
    # 3. Find advantaged / disadvantaged
    # -----------------------------

    disadvantaged_gender = min(gender_rates, key=gender_rates.get)
    advantaged_gender = max(gender_rates, key=gender_rates.get)

    disadvantaged_race = min(race_rates, key=race_rates.get)
    advantaged_race = max(race_rates, key=race_rates.get)

    disadvantaged_gender_rate = gender_rates[disadvantaged_gender]
    advantaged_gender_rate = gender_rates[advantaged_gender]

    disadvantaged_race_rate = race_rates[disadvantaged_race]
    advantaged_race_rate = race_rates[advantaged_race]

    # -----------------------------
    # 4. Fairness gaps
    # -----------------------------

    gender_gap = advantaged_gender_rate - disadvantaged_gender_rate
    race_gap = advantaged_race_rate - disadvantaged_race_rate

    # probability for mitigation (limit to 0.5)
    mitigation_prob = min(max(gender_gap, race_gap), 0.5)

    # -----------------------------
    # 5. Apply mitigation
    # -----------------------------

    for r in records:

        r["original_prediction"] = r["prediction"]
        r["mitigation_applied"] = False

        g = r["gender"].lower()
        rc = r["race"].lower()
        p = r["prediction"]

        # check disadvantaged group
        disadvantaged = (
            g == disadvantaged_gender
            or rc == disadvantaged_race
        )

        # -----------------------------
        # 6. Flip with probability
        # -----------------------------

        if p == 0 and disadvantaged:

            if random.random() < mitigation_prob:

                r["prediction"] = 1
                r["mitigation_applied"] = True
                r["mitigation_reason"] = "fairness_balance"

    # -----------------------------
    # 7. return updated records
    # -----------------------------

    return records