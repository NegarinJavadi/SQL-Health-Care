Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records:
This project uses 10 years of real hospital data (101,766 patient visits) to do two
things: answer clinical questions with SQL, and build a model that predicts whether
a patient will be readmitted within 30 days.

The 9 features built in SQL:

- `elderly_flag` — 1 if the patient is 70 or older
- `total_prior_visits` — outpatient + emergency + inpatient visits added up
- `insulin_adjusted` — 1 if the insulin dose was changed during the stay
- `a1c_abnormal` — 1 if the A1C result was high
- ...and five more

https://doi.org/10.1155/2014/781670