-- 1.sql
-- Please estimate the time spent on this problem.
-- Answer: __1__ minutes

-----------------------------------------------------------------------
SELECT name  FROM app
WHERE price = 0
ORDER BY  positive_ratings DESC
LIMIT 5;