-- 9.sql
-- Please estimate the time spent on this problem.
-- Answer: __25__ minutes

-----------------------------------------------------------------------
SELECT 
    sub.id AS id,
    ROUND(SUM(l.loanamount)::NUMERIC, 2) AS value
FROM (
    SELECT DISTINCT 
        p.personid AS id,
        lda.loanid AS loanid
    FROM 
        public.person p
    JOIN 
        public.personownaccount poa ON p.personid = poa.personid
    JOIN 
        public.account a_owned ON poa.accountid = a_owned.accountid
    JOIN 
        public.accounttransferaccount ata ON a_owned.accountid = ata.toid
    JOIN 
        public.loandepositaccount lda ON ata.fromid = lda.accountid
) AS sub
JOIN 
    public.loan l ON sub.loanid = l.loanid
GROUP BY 
    sub.id
ORDER BY 
    sub.id;