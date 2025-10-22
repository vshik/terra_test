CREATE OR ALTER PROCEDURE usp_GetExecutedActionsSummary
AS
BEGIN
    SELECT action_type, COUNT(*) AS total, SUM(cost_saving) AS total_saving
    FROM turbo_actions_log
    WHERE status = 'completed'
    GROUP BY action_type
    ORDER BY total_saving DESC;
END;
