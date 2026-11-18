> Synthetic demo document for the fictional company ShopCo. Definitions are for demo purposes.

# ShopCo KPI Definitions

Effective date: 2025-01-01. Owner: Analytics team. Last reviewed: 2025-08-20.

These are the official definitions for ShopCo's core KPIs. Every dashboard, report and analysis must use them. Table and column names refer to the BigQuery dataset described in data_dictionary.md.

## Revenue

Revenue is the sum of sale_price from order_items where status is not 'Cancelled' and not 'Returned'.

In SQL terms: SUM(order_items.sale_price) over rows with order_items.status NOT IN ('Cancelled', 'Returned'). Revenue is reported in US dollars, before tax and shipping. Items with status 'Processing', 'Shipped' or 'Complete' all count toward revenue.

## Return rate

Return rate per category is the count of order_items with status 'Returned' divided by the count of all order_items for that category.

The category comes from products.category, joined on order_items.product_id = products.id. The denominator includes every order_items row for the category, whatever its status, including 'Cancelled'. Return rate is reported as a percentage with one decimal place. Thresholds per category are in returns_policy.md.

## Average order value (AOV)

Average order value is revenue divided by the number of distinct orders that contain at least one item counted in revenue. Count distinct order_items.order_id over the same rows used for revenue.

## Repeat customer rate

Repeat customer rate is the number of customers with 2 or more distinct orders in the period divided by the number of customers with at least 1 order in the period. Customers are identified by orders.user_id. Cancelled orders (orders.status = 'Cancelled') are excluded from both counts.

## Reporting periods

KPIs are reported by calendar month and calendar quarter, using order_items.created_at in UTC. Quarter 1 is January to March.
