CREATE OR ALTER PROCEDURE usp_GetHighCostResources
AS
BEGIN
    SELECT TOP 50 resource_id, resource_name, region, environment, SUM(cost) AS total_monthly_cost
    FROM turbo_costs
    WHERE cost_date >= DATEADD(MONTH, -1, GETDATE())
    GROUP BY resource_id, resource_name, region, environment
    ORDER BY total_monthly_cost DESC;
END;
