> Synthetic demo document for the fictional company ShopCo. It describes the public BigQuery dataset bigquery-public-data.thelook_ecommerce.

# ShopCo Data Dictionary

Owner: Analytics team. Last reviewed: 2025-10-10.

ShopCo's sales data lives in the BigQuery dataset bigquery-public-data.thelook_ecommerce. All data is synthetic and is refreshed continuously, so counts change over time and some timestamps may be in the future. Always use fully qualified table names such as `bigquery-public-data.thelook_ecommerce.orders`.

## order_items

One row per item sold. This is the main table for revenue and return rate.
Columns: id, order_id, user_id, product_id, inventory_item_id, status, created_at, shipped_at, delivered_at, returned_at, sale_price.
status is one of 'Processing', 'Shipped', 'Complete', 'Cancelled', 'Returned'. sale_price is in US dollars.

## orders

One row per customer order. An order contains one or more order_items.
Columns: order_id, user_id, status, gender, created_at, returned_at, shipped_at, delivered_at, num_of_item.
Join to order_items on orders.order_id = order_items.order_id.

## products

One row per product.
Columns: id, cost, category, name, brand, retail_price, department, sku, distribution_center_id.
products.category holds the product category names used in returns_policy.md and category_owners.md, for example 'Jeans' or 'Outerwear & Coats'. department is 'Men' or 'Women'.
Join to order_items on order_items.product_id = products.id.

## users

One row per customer.
Columns: id, first_name, last_name, email, age, gender, state, street_address, postal_code, city, country, latitude, longitude, traffic_source, created_at, user_geom.

## inventory_items

One row per physical stock unit.
Columns: id, product_id, created_at, sold_at, cost, product_category, product_name, product_brand, product_retail_price, product_department, product_sku, product_distribution_center_id.

## distribution_centers

One row per warehouse.
Columns: id, name, latitude, longitude, distribution_center_geom.

## events

One row per website event (page view, cart, purchase).
Columns: id, user_id, sequence_number, session_id, created_at, ip_address, city, state, postal_code, browser, traffic_source, uri, event_type.

## Official metric definitions

Revenue is the sum of sale_price from order_items where status is not 'Cancelled' and not 'Returned'.

Return rate per category is the count of order_items with status 'Returned' divided by the count of all order_items for that category.

See kpi_definitions.md for average order value and repeat customer rate.
