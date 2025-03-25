-- 8.sql
-- Please estimate the time spent on this problem.
-- Answer: __20__ minutes

-----------------------------------------------------------------------
SELECT 
  a.mid AS id,
  ROUND((a.in_flow / b.out_flow)::numeric, 2) AS value
FROM (
  SELECT toid AS mid, SUM(amount) AS in_flow
  FROM accounttransferaccount
  GROUP BY toid
) a
JOIN (
  SELECT fromid AS mid, SUM(amount) AS out_flow
  FROM accounttransferaccount
  GROUP BY fromid
) b
  ON a.mid = b.mid
WHERE b.out_flow != 0
ORDER BY id;