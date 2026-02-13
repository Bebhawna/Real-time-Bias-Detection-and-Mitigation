import pandas as pd
import time
import joblib
#from db_config import insert_record   

def decode_gender(row):
    return "Male" if row["sex_Male"] == 1 else "Female"

def decode_race(row):
    race_cols = ["race_Asian-Pac-Islander", "race_Black", "race_Other", "race_White"]
    for col in race_cols:
        if row[col] == 1:
            return col.replace("race_", "")
    return "Unknown"

def main():
    model = joblib.load("../models/income_model.pkl")
    df = pd.read_csv("../data/stream.csv")

    print("Streaming started...")

    for idx, row in df.iterrows():
        x = row.drop("income", errors="ignore")

        # FIX: Keep feature names
        pred = int(model.predict(pd.DataFrame([x]))[0])

        gender = decode_gender(row)
        race = decode_race(row)

        record = {
            "gender": gender,
            "race": race,
            "features": x.to_dict(),
            "prediction": pred
        }

        try:
            #insert_record(record)
            print(f"{idx}: PRED={pred} | {gender} | {race}")
        except Exception as e:
            print(f"DB Insert Error at row {idx}: {e}")

        time.sleep(1)

    print("Streaming finished.")

if __name__ == "__main__":
    main()
