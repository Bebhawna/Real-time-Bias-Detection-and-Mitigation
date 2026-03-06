SET search_path TO fairness_system;

CREATE TABLE IF NOT EXISTS predictions_log (
    id SERIAL PRIMARY KEY,
    gender VARCHAR(10) CHECK (gender IN ('Male','Female','Other')),
    race VARCHAR(100),
    features JSONB NOT NULL,
    prediction INTEGER CHECK (prediction IN (0,1)),
    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_raw_timestamp
ON predictions_log(timestamp);