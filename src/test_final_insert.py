from db_config import insert_record, insert_final_record,fetch_latest_records,fetch_final_records

# # features = {"income":50000,"age":32}
# # record = {
# #     "gender": "Male",
# #     "race": "Asian",
# #     "features": features,
# #     "prediction": 1
# # }

# # raw_id = record
# # # raw_id = insert_record(
# # #     gender="Male",
# # #     race="Asian",
# # #     features=features,
# # #     prediction=1
# # # )
# # # features = {"income": 50000, "age": 32}
# # insert_final_record(
# #     raw_id=raw_id,
# #     gender="Male",
# #     race="Asian",
# #     features=features,
# #     prediction=1,
# #     mitigation_applied=False
# # )

# # print("RAW + FINAL insertion successful")




# # from db_config import insert_final_record

features = {"age": 30, "income": 50000}

insert_final_record(
    raw_id=1,
    gender="Male",
    race="Asian",
    features=features,
    prediction=1,
    mitigation_applied=False
)

# fetch_final_records(3)

# fetch_latest_records(10)
print("RAW + FINAL fetch successful")
