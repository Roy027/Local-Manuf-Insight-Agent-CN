KNOWLEDGE_BASE_SOPS = """
SOP-101: Temperature Control in Curing Process
- Optimal range: 145degC - 155degC.
- Deviation > 5degC for > 10 mins results in brittle fracture risk.
- Action: Check thermocouple calibration and PID controller gains.

SOP-204: Pressure Regulation
- Target pressure: 4500 PSI.
- Fluctuations > 10% indicate hydraulic leak or pump cavitation.
- Correlated with surface pitting defects.

SOP-305: Shift Handover Protocol
- Operators must log all yield excursions manually.
- Sudden drops in yield often correlate with Shift A to Shift B transition due to machine recalibration habits.
- Action: Standardize calibration offset to 0.0 before shift end.

Historical Case #882:
- Issue: Batch 45-50 showed high rejection rate.
- Cause: Raw material lot #992 was contaminated with moisture.
- Signature: Spike in humidity sensor coupled with lower viscosity.
""".strip()
