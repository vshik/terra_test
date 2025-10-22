CREATE OR ALTER PROCEDURE usp_GetRegionCostBreakdown
AS
BEGIN
    SELECT region, SUM(cost) AS total_cost
    FROM turbo_costs
    WHERE cost_date >= DATEADD(MONTH, -1, GETDATE())
    GROUP BY region
    ORDER BY total_cost DESC;
END;
