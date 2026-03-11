SET search_path TO fairness_system;

TRUNCATE TABLE final_predictions_log RESTART IDENTITY CASCADE;
TRUNCATE TABLE predictions_log RESTART IDENTITY CASCADE;