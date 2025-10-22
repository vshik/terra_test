CREATE OR ALTER PROCEDURE usp_GetRightSizingRecommendations
AS
BEGIN
    SELECT resource_id, resource_name, resource_type, current_size, recommended_size,
           cost_saving, performance_impact, timestamp
    FROM turbo_recommendations
    WHERE action_type = 'resize'
    ORDER BY cost_saving DESC;
END;
