-- 3.sql
-- Please estimate the time spent on this problem.
-- Answer: __10__ minutes

-----------------------------------------------------------------------
SELECT c.name
FROM company c
WHERE c.company_id IN (
    SELECT publisher_id
    FROM app
    WHERE release_date >= DATE '2005-01-01' 
      AND release_date < DATE '2015-01-01'
    GROUP BY publisher_id
    HAVING COUNT(app_id) > 5
);