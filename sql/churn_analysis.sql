-- Customer Churn Business Analysis

-- 1. Overall churn rate
SELECT
    exit,
    COUNT(*) AS customer_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS percentage
FROM churn_data
GROUP BY exit;


-- 2. Churn rate by customer segment
SELECT
    customer_segment,
    COUNT(*) AS customer_count,
    ROUND(AVG(CAST(exit AS INTEGER)) * 100, 2) AS churn_rate
FROM churn_data
GROUP BY customer_segment
ORDER BY churn_rate DESC;


-- 3. Churn rate by number of services
SELECT
    nums_service,
    COUNT(*) AS customer_count,
    ROUND(AVG(CAST(exit AS INTEGER)) * 100, 2) AS churn_rate
FROM churn_data
GROUP BY nums_service
ORDER BY nums_service;


-- 4. Churn rate by active membership
SELECT
    active_member,
    COUNT(*) AS customer_count,
    ROUND(AVG(CAST(exit AS INTEGER)) * 100, 2) AS churn_rate
FROM churn_data
GROUP BY active_member
ORDER BY churn_rate DESC;

-- 5. Churn rate by loyalty level
SELECT
    loyalty_level,
    COUNT(*) AS customer_count,
    ROUND(AVG(CAST(exit AS INTEGER)) * 100, 2) AS churn_rate
FROM churn_data
GROUP BY loyalty_level
ORDER BY churn_rate DESC;

-- 6. Churn rate by digital behaviour
SELECT
    digital_behavior,
    COUNT(*) AS customer_count,
    ROUND(AVG(CAST(exit AS INTEGER)) * 100, 2) AS churn_rate
FROM churn_data
GROUP BY digital_behavior
ORDER BY churn_rate DESC;

-- 7. Average customer financial characteristics by churn status
SELECT
    exit,
    COUNT(*) AS customer_count,
    ROUND(AVG(balance), 2) AS average_balance,
    ROUND(AVG(monthly_ir), 2) AS average_monthly_income
FROM churn_data
GROUP BY exit
ORDER BY exit;

-- 8. Identify high-risk customer groups
SELECT
    customer_segment,
    loyalty_level,
    active_member,
    digital_behavior,
    COUNT(*) AS customer_count,
    ROUND(AVG(CAST(exit AS INTEGER)) * 100, 2) AS churn_rate
FROM churn_data
GROUP BY
    customer_segment,
    loyalty_level,
    active_member,
    digital_behavior
HAVING COUNT(*) >= 100
ORDER BY churn_rate DESC;

-- 9. Highest-churn customer groups with meaningful sample sizes
SELECT
    customer_segment,
    loyalty_level,
    nums_service,
    COUNT(*) AS customer_count,
    ROUND(AVG(CAST(exit AS INTEGER)) * 100, 2) AS churn_rate
FROM churn_data
GROUP BY
    customer_segment,
    loyalty_level,
    nums_service
HAVING COUNT(*) >= 100
ORDER BY churn_rate DESC
LIMIT 10;

-- 10. Customer value potentially exposed to churn by segment
SELECT
    customer_segment,
    COUNT(*) AS customer_count,
    SUM(CASE WHEN exit = 1 THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(
        SUM(CASE WHEN exit = 1 THEN balance ELSE 0 END),
        2
    ) AS balance_held_by_churned_customers
FROM churn_data
GROUP BY customer_segment
ORDER BY balance_held_by_churned_customers DESC;

