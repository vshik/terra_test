CREATE OR ALTER PROCEDURE usp_GetCostTrends
    @ResourceID NVARCHAR(100)
AS
BEGIN
    SELECT cost_date, SUM(cost) AS daily_cost
    FROM turbo_costs
    WHERE resource_id = @ResourceID
    GROUP BY cost_date
    ORDER BY cost_date;
END;
