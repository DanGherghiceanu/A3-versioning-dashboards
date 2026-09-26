-- Read-only login for Grafana. Safe to run repeatedly.
-- Grafana can SELECT from MLflow's tables and the app's prediction log, but never change them.
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'grafana') THEN
    CREATE ROLE grafana LOGIN PASSWORD 'grafana';
  END IF;
END
$$;
GRANT CONNECT ON DATABASE mlflow TO grafana;
GRANT USAGE ON SCHEMA public TO grafana;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO grafana;
-- Tables created later by the mlflow user (e.g. app_predictions on a fresh stack) are readable too.
ALTER DEFAULT PRIVILEGES FOR ROLE mlflow IN SCHEMA public GRANT SELECT ON TABLES TO grafana;
