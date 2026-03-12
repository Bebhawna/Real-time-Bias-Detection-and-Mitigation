from mitigation import apply_mitigation
from db_config import fetch_latest_records, insert_final_record


def mitigation_pipeline():

    # fetch RAW records
    records = fetch_latest_records(100)

    if not records:
        print("No records found in predictions_log")
        return

    for record in records:

        # print("\nProcessing RAW record:")
        # print(record)

        raw_id = record["id"]

        # Apply mitigation logic
        corrected = apply_mitigation(record)

        print("Corrected Record:\n")
        print(corrected)

        # Insert into FINAL table
        insert_final_record(
            raw_id=raw_id,
            gender=corrected["gender"],
            race=corrected["race"],
            features=corrected["features"],
            prediction=corrected["prediction"],
            mitigation_applied=corrected["mitigation_applied"]
        )

        print(f"Inserted FINAL record for raw_id={raw_id}")


if __name__ == "__main__":
    mitigation_pipeline()