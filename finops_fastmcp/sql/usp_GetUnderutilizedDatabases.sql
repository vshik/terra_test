CREATE OR ALTER PROCEDURE usp_GetUnderutilizedDatabases
AS
BEGIN
    SELECT r.resource_id, r.resource_name, AVG(p.cpu_utilization) AS avg_cpu,
           AVG(p.memory_utilization) AS avg_mem
    FROM turbo_resources r
    JOIN turbo_performance_metrics p ON r.resource_id = p.resource_id
    WHERE r.resource_type LIKE '%database%'
    GROUP BY r.resource_id, r.resource_name
    HAVING AVG(p.cpu_utilization) < 10 AND AVG(p.memory_utilization) < 10;
END;
