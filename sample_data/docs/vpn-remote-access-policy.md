---
title: Remote Access and VPN Policy
type: policy
source: it-security
date: 2025-01-15
---

All remote access to company systems must go through the corporate VPN. Employees connect to the VPN using the GlobalProtect client with their network credentials, and multi-factor authentication (MFA) through the authenticator app is mandatory for every VPN session.

To connect to the VPN, install the GlobalProtect client from the IT software portal, enter the gateway address vpn.example-corp.com, sign in with your network username and password, and approve the MFA push notification. Split tunneling is disabled, so all traffic routes through the corporate network while the VPN is connected.

VPN accounts are provisioned automatically on your start date. If you cannot connect to the VPN, first check that your MFA device is registered, then restart the GlobalProtect service. Persistent VPN connection failures should be reported to the IT help desk, which can reset your VPN profile.

Remote access from personal devices is permitted only when the device is enrolled in mobile device management and reports a compliant posture. VPN sessions time out after 12 hours and require re-authentication with MFA.
