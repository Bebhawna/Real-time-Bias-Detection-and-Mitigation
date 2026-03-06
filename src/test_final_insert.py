from db_config import insert_record, insert_final_record

features = {"income":50000,"age":32}

raw_id = insert_record(
    gender="Male",
    race="Asian",
    features=features,
    prediction=1
)

insert_final_record(
    raw_id=raw_id,
    gender="Male",
    race="Asian",
    features=features,
    prediction=1,
    mitigation=False
)

print("RAW + FINAL insertion successful")