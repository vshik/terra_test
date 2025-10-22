CREATE OR ALTER PROCEDURE usp_GetDailySavingsSummary
AS
BEGIN
    SELECT CAST(timestamp AS DATE) AS action_date,
           SUM(cost_saving) AS daily_savings
    FROM turbo_recommendations
    WHERE action_type = 'resize'
    GROUP BY CAST(timestamp AS DATE)
    ORDER BY action_date DESC;
END;
