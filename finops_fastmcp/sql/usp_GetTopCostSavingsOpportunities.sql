CREATE OR ALTER PROCEDURE usp_GetTopCostSavingsOpportunities
AS
BEGIN
    SELECT TOP 20 resource_id, resource_name, cost_saving, action_type, recommended_size
    FROM turbo_recommendations
    WHERE cost_saving > 0
    ORDER BY cost_saving DESC;
END;
