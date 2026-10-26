
-- ============================================================
-- PART A: SEVEN REQUIRED SQL QUERIES
-- Database: data/instamart.db
-- ============================================================

-- Q1. WHERE + GROUP BY + HAVING
-- Find customers whose completed orders above INR 500
-- have a combined value greater than INR 8,000.

SELECT
    customer_id,
    ROUND(SUM(order_amount), 2) AS qualifying_spend
FROM orders
WHERE status = 'completed'
  AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY qualifying_spend DESC;


-- Q2. THREE-TABLE JOIN
-- Completed revenue by customer city and product category.

SELECT
    c.city,
    p.category,
    ROUND(SUM(o.order_amount), 2) AS completed_revenue
FROM orders AS o
JOIN customers AS c
    ON o.customer_id = c.customer_id
JOIN products AS p
    ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;


-- Q3. SCALAR SUBQUERY
-- Products priced above the average catalogue price.

SELECT
    product_id,
    product_name,
    category,
    price_inr
FROM products
WHERE price_inr > (
    SELECT AVG(price_inr)
    FROM products
)
ORDER BY price_inr DESC;


-- Q4. LEFT JOIN
-- Find customers who have never placed a completed order.

SELECT
    c.customer_id,
    c.city,
    c.gender
FROM customers AS c
LEFT JOIN orders AS o
    ON c.customer_id = o.customer_id
   AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;


-- Q5. COMMON TABLE EXPRESSION (CTE)
-- Find customers whose completed spend exceeds the average
-- completed spend among customers with completed orders.

WITH customer_spend AS (
    SELECT
        customer_id,
        SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
average_spend AS (
    SELECT AVG(total_spend) AS avg_spend
    FROM customer_spend
)
SELECT
    cs.customer_id,
    ROUND(cs.total_spend, 2) AS total_spend,
    ROUND(a.avg_spend, 2) AS average_spend
FROM customer_spend AS cs
CROSS JOIN average_spend AS a
WHERE cs.total_spend > a.avg_spend
ORDER BY cs.total_spend DESC;


-- Q6. WINDOW FUNCTIONS
-- Rank customers by completed spend within each city.
-- RANK and DENSE_RANK preserve ties; ROW_NUMBER assigns
-- a distinct sequential number to every row.

WITH customer_spend AS (
    SELECT
        c.customer_id,
        c.city,
        SUM(o.order_amount) AS total_spend
    FROM customers AS c
    JOIN orders AS o
        ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id, c.city
)
SELECT
    customer_id,
    city,
    ROUND(total_spend, 2) AS total_spend,
    RANK() OVER (
        PARTITION BY city
        ORDER BY total_spend DESC
    ) AS spend_rank,
    DENSE_RANK() OVER (
        PARTITION BY city
        ORDER BY total_spend DESC
    ) AS dense_spend_rank,
    ROW_NUMBER() OVER (
        PARTITION BY city
        ORDER BY total_spend DESC, customer_id
    ) AS spend_row_number
FROM customer_spend
ORDER BY city, spend_rank, customer_id;


-- Q7. DAILY REVENUE + CUMULATIVE SUM
-- Calculate daily completed revenue and running total.

WITH daily_revenue AS (
    SELECT
        SUBSTR(order_ts, 1, 10) AS order_day,
        SUM(order_amount) AS daily_revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT
    order_day,
    ROUND(daily_revenue, 2) AS daily_revenue,
    ROUND(
        SUM(daily_revenue) OVER (
            ORDER BY order_day
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ),
        2
    ) AS cumulative_revenue
FROM daily_revenue
ORDER BY order_day;
