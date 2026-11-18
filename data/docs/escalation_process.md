> Synthetic demo document for the fictional company ShopCo. All names and emails are fictional.

# ShopCo Escalation Process

Effective date: 2025-03-01. Owner: Operations Excellence. Last reviewed: 2025-09-30.

## When to escalate

Escalate when any of these happen:
- A product category's monthly return rate is above its maximum acceptable return rate in returns_policy.md.
- More than 3% of items in a week are delivered late, as defined in shipping_sla.md.
- A single customer complaint involves a safety issue or a legal threat.
- Revenue for any category drops more than 20% compared with the same month last year.

## Return rate breach escalation levels

Level 1: The category is above its maximum acceptable return rate by up to 1.0 percentage point. The Analytics team notifies the category owner (see category_owners.md) within 2 business days of month end. The category owner must reply with a root-cause note within 5 business days.

Level 2: The category is above its maximum acceptable return rate by more than 1.0 percentage point, or has been at Level 1 for 2 months in a row. Escalate to the Head of Merchandising, Priya Nair (priya.nair@shopco.example), within 2 business days. A corrective action plan is due within 10 business days.

Level 3: The category is above its maximum acceptable return rate by more than 2.0 percentage points, or has been at Level 2 for 2 months in a row. Escalate to the Chief Operating Officer, Daniel Okafor (daniel.okafor@shopco.example). The category may be paused from all promotions until the return rate is back under its threshold.

## How to escalate

Open a ticket in the Operations queue with the title "[ESCALATION L<level>] <topic>". Include the metric, its value, the threshold, the time period, and the SQL or report used. Send the ticket link by email to the person named for that level.

## Response times

Safety and legal issues: acknowledge within 4 business hours.
All other escalations: acknowledge within 1 business day.
