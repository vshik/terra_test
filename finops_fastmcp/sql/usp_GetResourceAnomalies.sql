CREATE OR ALTER PROCEDURE usp_GetResourceAnomalies
AS
BEGIN
    SELECT r.resource_id, r.resource_name, p.date, p.cpu_utilization
    FROM turbo_performance_metrics p
    JOIN turbo_resources r ON r.resource_id = p.resource_id
    WHERE p.cpu_utilization > 95
    ORDER BY p.date DESC;
END;
