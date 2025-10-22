CREATE OR ALTER PROCEDURE usp_GetLongRunningResources
AS
BEGIN
    SELECT r.resource_id, r.resource_name, DATEDIFF(DAY, r.created_at, GETDATE()) AS running_days
    FROM turbo_resources r
    WHERE DATEDIFF(DAY, r.created_at, GETDATE()) > 90;
END;
