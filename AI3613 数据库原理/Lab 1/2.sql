-- 2.sql
-- Please estimate the time spent on this problem.
-- Answer: __10__ minutes

-----------------------------------------------------------------------
SELECT DISTINCT support.website
FROM support 
JOIN requirements ON support.app_id = requirements.app_id
WHERE (requirements.mac_requirements IS NOT NULL OR requirements.linux_requirements IS NOT NULL)
AND support.website IS NOT NULL;