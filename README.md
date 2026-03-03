# 🛡️ Sentinel-Detect
### AI-Powered Employee Behaviour Anomaly Detection System

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic data & seed the database
python data_generator.py

# 3. Train the model & score all logs
python model_engine.py

# 4. Launch the Security Audit Dashboard
streamlit run dashboard.py
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Sentinel-Detect                        │
│                                                             │
│  ┌──────────────────┐    ┌──────────────────────────────┐  │
│  │  data_generator  │───▶│      SQLite Database         │  │
│  │                  │    │  • activity_logs             │  │
│  │ • Synthetic logs │    │  • audit_tasks               │  │
│  │ • Anomaly inject │    └──────────┬───────────────────┘  │
│  │ • Feature eng.   │               │                       │
│  └──────────────────┘    ┌──────────▼───────────────────┐  │
│                           │      model_engine            │  │
│  ┌──────────────────┐    │                              │  │
│  │   dashboard.py   │◀───│ • Isolation Forest           │  │
│  │                  │    │ • XAI Explainer              │  │
│  │ • Integrity Score│    │ • Audit Task trigger         │  │
│  │ • Alert Feed     │    │ • Real-time scoring          │  │
│  │ • Visualisations │    └──────────────────────────────┘  │
│  │ • Audit Tasks    │                                       │
│  └──────────────────┘                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Feature Engineering

The Isolation Forest model is trained on **6 engineered features** derived from raw activity logs. Each was chosen because it captures a distinct fraud or integrity-risk signal.

### Feature 1 — `freq_score` (Event Frequency)
**What it measures:** Number of events the same employee performed in the **rolling 60-minute window** prior to the current event.

**Why it matters:** A normal employee might access 3–5 documents per hour. An insider threat or compromised credential will exhibit a burst pattern — 20–50 accesses within minutes — that normal work patterns cannot explain.

**How it's computed:**
```python
for each event, count events by same employee_id
where timestamp >= (event.timestamp − 60 min)
```

### Feature 2 — `duration_minutes` (Task Duration)
**What it measures:** Wall-clock minutes between task start and completion.

**Why it matters:** Each action type has an expected duration profile (e.g. "Submit Task" ≈ 30 min ± 8 min). A task completing in 0.02 minutes suggests automation/scripted exfiltration. A task taking 1,000+ minutes suggests the session was left open, a sign of credential sharing or abandonment.

**How it's computed:** Directly from the raw log field. Z-score outliers are surfaced by the XAI explainer.

### Feature 3 — `security_level` (Document Sensitivity)
**What it measures:** Encoded numeric sensitivity of the accessed document: Public=1, Internal=2, Confidential=3, Restricted=4.

**Why it matters:** When combined with dept clearance, high security-level access by a low-clearance employee (e.g. Marketing accessing Restricted HR files) is a strong indicator of privilege escalation or misconfigured ACL.

### Feature 4 — `transaction_amount` (Financial Claim Value)
**What it measures:** Dollar value of a Financial Claim event (0 for non-financial events).

**Why it matters:** Fraudulent expense claims or payment diversion attacks manifest as outlier amounts — either single large claims or many small claims that aggregate above threshold. IsolationForest isolates these as low-density regions.

### Feature 5 — `hour_of_day` (Temporal Context)
**What it measures:** Integer hour (0–23) at which the event occurred.

**Why it matters:** After-hours activity (22:00–05:00) combined with high security-level access is a known exfiltration pattern. Legitimate employees rarely access Restricted documents at 3 AM.

### Feature 6 — `action_encoded` (Action Type)
**What it measures:** Ordinal encoding of the action type (Access Doc=0, Submit Task=1, Financial Claim=2).

**Why it matters:** The distribution of action types per employee follows stable priors. An employee who suddenly shifts from 90% "Submit Task" to 90% "Access Doc" (especially sensitive docs) departs from their behavioural baseline.

---

## ML Model: Isolation Forest

### Why Isolation Forest?
- **Unsupervised**: No labelled anomaly data required — critical in security where novel attacks are unknown.
- **Scalable**: O(n log n) complexity; handles real-time event streams.
- **Interpretable**: Anomaly score is a continuous value, enabling risk triage rather than binary flags.

### How It Works
Isolation Forest builds an ensemble of random decision trees. Anomalous points require **fewer splits** to isolate because they occupy sparse, low-density regions of the feature space. The `score_samples()` output is normalised: values near 0 indicate normal behaviour; values approaching −0.5 indicate strong anomalies.

### Threshold
`ANOMALY_THRESHOLD = −0.10` — events below this score trigger an Audit Task. This can be tuned via `model_engine.py` depending on the organisation's risk tolerance.

### Contamination
`CONTAMINATION = 0.08` — tells the model to expect ~8% anomalies during training, aligning with the synthetic injection rate. In production, calibrate from historical confirmed-incident rate.

---

## XAI Explainability

For every flagged anomaly, Sentinel-Detect computes a **z-score** for each feature against the training distribution:

```
z = (feature_value − training_mean) / training_std
```

Features with |z| ≥ 1.5 are reported in descending order of deviation:

```
'Task Duration (minutes)' = 0.02 min (18.3σ below mean);
'Event Frequency (last 60 min)' = 43 events (9.1σ above mean)
```

This gives security analysts an immediate, actionable reason for each alert rather than a black-box score.

---

## Audit Task Hand-off (Orchestra-Agent Integration)

When `anomaly_score < ANOMALY_THRESHOLD`, a record is written to the `audit_tasks` table:

```sql
INSERT INTO audit_tasks
  (log_id, employee_id, department, action_type,
   timestamp, anomaly_score, reason, status)
VALUES (…, 'OPEN');
```

An orchestration agent (Orchestra-Agent / Project 2) polls this table for `status = 'OPEN'` records and can:
- Notify the Security Operations Centre via Slack/email.
- Trigger an automated account lock after N open tasks for the same employee.
- Escalate to HR/Legal for high-severity scores.
- Mark tasks `CLOSED` after review, feeding back to the model as confirmed labels.

---

## Fraud Prevention & Compliance

### Internal Fraud Prevention
| Threat Vector | How Sentinel-Detect Catches It |
|---|---|
| Credential theft / account takeover | Frequency burst + after-hours hour_of_day |
| Privilege escalation | Security level > department clearance |
| Financial fraud (ghost employees, inflated claims) | Transaction amount outlier |
| Data exfiltration via bulk download | High freq_score on Restricted docs |
| Insider threat (slow-burn data theft) | Gradual drift in action_encoded baseline |
| Automation / scripted attacks | Near-zero duration_minutes |

### Compliance Alignment
- **SOX (Sarbanes-Oxley)**: Financial claim outlier detection supports internal controls over financial reporting.
- **GDPR / Data Protection**: Restricted-document access monitoring limits unauthorised PII access.
- **ISO 27001 / NIST CSF**: Continuous monitoring and audit trail support Annex A.12 (Operations Security) and NIST DE.CM-7 (Monitoring for unauthorized activity).
- **Audit Trail**: Every scored event is immutably stored in SQLite with timestamps, making it court-admissible evidence.

---

## File Structure

```
sentinel-detect/
├── data_generator.py     # Synthetic log generation, anomaly injection, feature engineering
├── model_engine.py       # Isolation Forest training, scoring, XAI, audit task creation
├── dashboard.py          # Streamlit Security Audit Dashboard
├── requirements.txt      # Python dependencies
├── sentinel.db           # SQLite database (auto-created)
├── sentinel_model.joblib # Trained Isolation Forest (auto-created)
├── sentinel_scaler.joblib# Feature scaler (auto-created)
└── sentinel_feature_stats.joblib  # Feature stats for XAI (auto-created)
```

---

*Sentinel-Detect v1.0 — Built with scikit-learn, Pandas, SQLite, Streamlit & Plotly*
