-- 7.sql
-- Please estimate the time spent on this problem.
-- Answer: __15__ minutes

-----------------------------------------------------------------------
SELECT e1.fromid AS id, COUNT(*) AS value
FROM public.accounttransferaccount e1
JOIN public.accounttransferaccount e2 ON e1.toid = e2.fromid
JOIN public.accounttransferaccount e3 ON e2.toid = e3.fromid AND e3.toid = e1.fromid
WHERE e1.fromid != e1.toid
  AND e2.fromid != e2.toid
  AND e3.fromid != e3.toid
GROUP BY e1.fromid
ORDER BY id ASC;