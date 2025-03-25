-- 4.sql
-- Please estimate the time spent on this problem.
-- Answer: __10__ minutes

-----------------------------------------------------------------------
SELECT c.name
FROM company c
WHERE NOT EXISTS (
    SELECT 1
    FROM app a
    WHERE a.developer_id = c.company_id
      AND a.price > 50
);