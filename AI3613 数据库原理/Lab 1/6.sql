-- 6.sql
-- Please estimate the time spent on this problem.
-- Answer: __5__ minutes
-- 6~10 questions are modified and checked by AI.

-----------------------------------------------------------------------
SELECT a2.accountId as id, ROUND(CAST(SUM(t.amount) as numeric), 2) as value 
FROM Account AS a1
JOIN AccountTransferAccount AS t ON a1.accountId = t.fromId
JOIN Account AS a2 ON t.toId = a2.accountId
GROUP BY a2.accountId
ORDER BY SUM(t.amount) DESC 
LIMIT 10;