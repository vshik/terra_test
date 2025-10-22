CREATE OR ALTER PROCEDURE usp_GetResourcePerformanceSummary
    @ResourceID NVARCHAR(100)
AS
BEGIN
    SELECT resource_id,
           AVG(cpu_utilization) AS avg_cpu,
           AVG(memory_utilization) AS avg_mem,
           AVG(network_utilization) AS avg_net
    FROM turbo_performance_metrics
    WHERE resource_id = @ResourceID
    GROUP BY resource_id;
END;
