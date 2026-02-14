CREATE TABLE predictions_log (
    id SERIAL PRIMARY KEY,
    gender VARCHAR(10) CHECK (gender IN ('Male', 'Female', 'Other')),
    race VARCHAR(100),
    features JSONB NOT NULL,
    prediction INTEGER CHECK (prediction IN (0, 1)),
    timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

SELECT * FROM predictions_log;
WHERE prediction=1;

SELECT * FROM predictions_log;

