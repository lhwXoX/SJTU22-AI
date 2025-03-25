-- 5.sql
-- Please estimate the time spent on this problem.
-- Answer: __5__ minutes

-----------------------------------------------------------------------
SELECT DISTINCT a.name
FROM app a
JOIN description d ON a.app_id = d.app_id
WHERE LOWER(d.about_the_game) LIKE '%game of the year%'
ORDER BY a.name DESC;