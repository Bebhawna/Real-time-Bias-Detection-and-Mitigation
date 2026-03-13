from fastapi import FastAPI
from mitigation import apply_mitigation
from db_config import fetch_latest_records, insert_final_record

app = FastAPI()

@app.post("/mitigate")
def mitigation_pipeline():

    records = fetch_latest_records(100)

    if not records:
        return {"status": "no_data"}

    processed = 0

    for record in records:

        raw_id = record["id"]

        corrected = apply_mitigation(record)

        insert_final_record(
            raw_id=raw_id,
            gender=corrected["gender"],
            race=corrected["race"],
            features=corrected["features"],
            prediction=corrected["prediction"],
            mitigation_applied=corrected["mitigation_applied"]
        )

        processed += 1

    return {
        "status": "mitigation_complete",
        "processed_records": processed
    }
