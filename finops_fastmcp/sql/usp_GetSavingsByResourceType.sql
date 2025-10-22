CREATE OR ALTER PROCEDURE usp_GetSavingsByResourceType
AS
BEGIN
    SELECT resource_type, SUM(cost_saving) AS total_saving
    FROM turbo_recommendations
    GROUP BY resource_type
    ORDER BY total_saving DESC;
END;
