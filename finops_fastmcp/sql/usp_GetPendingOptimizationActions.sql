CREATE OR ALTER PROCEDURE usp_GetPendingOptimizationActions
AS
BEGIN
    SELECT resource_id, resource_name, action_type, recommended_size, cost_saving, created_at
    FROM turbo_actions_log
    WHERE status = 'pending'
    ORDER BY created_at DESC;
END;
