-- 10.sql
-- Please estimate the time spent on this problem.
-- Answer: __30__ minutes

-----------------------------------------------------------------------
WITH RECURSIVE GuaranteeChain AS (
    SELECT 
        fromid AS p1_id, 
        toid AS guaranteed_person_id, 
        1 AS level
    FROM personguaranteeperson
    UNION ALL
    SELECT 
        g.p1_id, 
        p.toid, 
        g.level + 1 AS level
    FROM GuaranteeChain g
    JOIN personguaranteeperson p ON g.guaranteed_person_id = p.fromid
    WHERE g.level < 3
),
DistinctGuaranteed AS (
    SELECT DISTINCT p1_id, guaranteed_person_id
    FROM GuaranteeChain
)
SELECT 
    dg.p1_id AS id,
    ROUND(COALESCE(SUM(l.loanamount), 0)::numeric, 2) AS value
FROM 
    DistinctGuaranteed dg
    LEFT JOIN personapplyloan pal ON dg.guaranteed_person_id = pal.personid
    LEFT JOIN loan l ON pal.loanid = l.loanid
GROUP BY 
    dg.p1_id
ORDER BY 
    id ASC;