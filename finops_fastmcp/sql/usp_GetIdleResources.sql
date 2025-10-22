CREATE OR ALTER PROCEDURE usp_GetIdleResources
AS
BEGIN
    SELECT r.resource_id, r.resource_name, r.resource_type, AVG(p.cpu_utilization) AS avg_cpu
    FROM turbo_resources r
    JOIN turbo_performance_metrics p ON r.resource_id = p.resource_id
    WHERE p.cpu_utilization < 5
    GROUP BY r.resource_id, r.resource_name, r.resource_type
    HAVING COUNT(DISTINCT p.date) > 7;
END;
