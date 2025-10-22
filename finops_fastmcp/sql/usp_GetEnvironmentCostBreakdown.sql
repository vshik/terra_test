CREATE OR ALTER PROCEDURE usp_GetEnvironmentCostBreakdown
AS
BEGIN
    SELECT environment, SUM(cost) AS total_cost
    FROM turbo_costs
    WHERE cost_date >= DATEADD(MONTH, -1, GETDATE())
    GROUP BY environment
    ORDER BY total_cost DESC;
END;
