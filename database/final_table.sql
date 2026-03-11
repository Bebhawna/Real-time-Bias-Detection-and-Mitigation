SET search_path TO fairness_system;

CREATE TABLE IF NOT EXISTS final_predictions_log (
    id BIGSERIAL PRIMARY KEY,
    raw_id INTEGER NOT NULL REFERENCES predictions_log(id),
    gender VARCHAR(10) CHECK (gender IN ('Male','Female','Other')),
    race VARCHAR(100),
    features JSONB NOT NULL,
    prediction INTEGER CHECK (prediction IN (0,1)),
    mitigation_applied BOOLEAN DEFAULT FALSE,
    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(raw_id)
);

CREATE INDEX IF NOT EXISTS idx_final_timestamp
ON final_predictions_log(timestamp);

CREATE INDEX IF NOT EXISTS idx_final_gender
ON final_predictions_log(gender);

CREATE INDEX IF NOT EXISTS idx_final_race
ON final_predictions_log(race);