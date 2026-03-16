# from fastapi import FastAPI
# from src.mitigation import apply_mitigation
# from src.db_config import fetch_latest_records, insert_final_record

# app = FastAPI()

# @app.post("/mitigate")
# def mitigation_pipeline():

#     records = fetch_latest_records(100)

#     if not records:
#         return {"status": "no_data"}

#     processed = 0

#     for record in records:

#         raw_id = record["id"]
#         stats = {
#     "minority_positive_rate": 0.3,
#     "majority_positive_rate": 0.7
# }
#         corrected = apply_mitigation(record,stats)

#         insert_final_record(
#             raw_id=raw_id,
#             gender=corrected["gender"],
#             race=corrected["race"],
#             features=corrected["features"],
#             prediction=corrected["prediction"],
#             mitigation_applied=corrected["mitigation_applied"]
#         )

#         processed += 1

#     return {
#         "status": "mitigation_complete",
#         "processed_records": processed
#     }







from fastapi import FastAPI
from src.mitigation import apply_mitigation
from src.db_config import fetch_latest_records, insert_final_record

app = FastAPI()


@app.post("/mitigate")
def mitigation_pipeline():

    # 1. fetch records from RAW table
    records = fetch_latest_records(100)

    if not records:
        return {"status": "no_data"}

    # 2. apply mitigation on full batch
    corrected_records = apply_mitigation(records)

    processed = 0

    # 3. insert corrected records into FINAL table
    for rec in corrected_records:

        insert_final_record(
            raw_id=rec["id"],
            gender=rec["gender"],
            race=rec["race"],
            features=rec.get("features"),
            prediction=rec["prediction"],
            mitigation_applied=rec["mitigation_applied"]
        )

        processed += 1

    return {
        "status": "mitigation_complete",
        "processed_records": processed
    }