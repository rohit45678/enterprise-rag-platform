---
title: Cloud Cost Optimization Guide
type: guide
source: engineering
date: 2025-08-19
---

This guide summarizes the engineering practices that keep our cloud bill under control. Teams review their cloud spend monthly in the cost dashboard and tag every resource with an owner and a project code; untagged resources are flagged for cleanup.

Reserved instances and savings plans should cover steady-state compute. Purchase one-year reservations for baseline workloads and review utilization quarterly; reservations running below 80 percent utilization should be exchanged or flagged to the platform team.

Rightsizing is the second lever. Instances averaging below 20 percent CPU for two weeks are candidates for a smaller instance type. Storage costs drop when infrequently accessed data moves to archive tiers after 90 days.

Development environments must be shut down outside working hours using the automated scheduler. Orphaned volumes, unattached IP addresses, and snapshots older than 90 days are deleted by the weekly cleanup automation after owner notification.
