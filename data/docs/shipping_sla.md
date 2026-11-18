> Synthetic demo document for the fictional company ShopCo. Not a real service level agreement.

# ShopCo Shipping Service Level Agreement (SLA)

Effective date: 2025-02-01. Owner: Fulfillment Operations. Last reviewed: 2025-07-30.

## Processing time

Orders placed before 2:00 PM US Eastern time on a business day must ship the same day. Orders placed after 2:00 PM, or on weekends and US public holidays, must ship by the end of the next business day.

Processing time is measured from order_items.created_at to order_items.shipped_at. The target is that 95% of items ship within 1 business day.

## Delivery time by shipping method

Standard shipping: delivered within 5 business days of shipping. Free on orders of $50 or more, otherwise $5.95.
Expedited shipping: delivered within 2 business days of shipping. Costs $14.95.
Overnight shipping: delivered the next business day. Costs $29.95. Available only in the contiguous United States.

International orders ship by standard international shipping and are delivered within 7 to 14 business days. Customs fees are paid by the customer.

## SLA breach

An item is late if delivered_at is more than the promised number of business days after shipped_at. If more than 3% of items in a calendar week are late, Fulfillment Operations must file an incident report within 2 business days. Customers whose expedited or overnight order arrives late receive an automatic refund of the shipping fee.

## Distribution centers

Orders ship from the distribution center closest to the customer that has stock (see distribution_centers in data_dictionary.md). Split shipments from two centers count as one order for SLA purposes; the SLA applies to the last item delivered.
