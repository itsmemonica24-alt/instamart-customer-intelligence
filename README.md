# Instamart Customer Intelligence & Support Copilot

## Overview

This project analyses synthetic Instamart customer, product, order, and support-ticket data. It builds a foundation for customer analytics, customer lapse prediction, support-ticket classification, and an AI support copilot.

The project is divided into four parts:

- **Part A:** Data cleaning, SQLite storage, SQL queries, and exploratory data analysis.
- **Part B:** RFM customer segmentation, lapse prediction, model evaluation, and retention targeting.
- **Part C:** Support-ticket classification and triage.
- **Part D:** Policy-grounded support and a refund-agent workflow.

All results are based on synthetic assignment data and should not be interpreted as actual Instamart customer behaviour.

## Reproduce

Run these commands from the project root in Windows PowerShell.

1. Create a virtual environment:

   ```powershell
   python -m venv .venv
   ```

2. Activate it:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Generate the raw data and policy files:

   ```powershell
   python generate_data.py
   ```

5. Audit the raw data:

   ```powershell
   python audit_data.py
   ```

6. Clean the data and create the SQLite database:

   ```powershell
   python clean_data.py
   ```

7. Validate the cleaned data:

   ```powershell
   python validate_data.py
   ```

8. Run the SQL queries:

   ```powershell
   python run_sql_queries.py
   ```

9. Run the Part A analysis:

   ```powershell
   python part_a_analysis.py
   ```

10. Run the Part B analysis:

    ```powershell
    python part_b_analysis.py
    ```

11. Run the Part C and Part D scripts after their implementation. Their commands and outputs will be documented here when those parts are complete.

The original `generate_data.py` and `mock_llm.py` files are retained at the project root.

## Part A

### Data cleaning and storage

The raw dataset contained 13,084 order rows, including duplicate records and data-quality issues. Cleaning produced:

- 1,200 customers
- 40 products
- 12,309 distinct orders
- 11,110 completed orders
- 1,199 cancelled orders
- 0 pending orders

The cleaning process handled duplicate records, invalid and missing customer ages, inconsistent city and gender values, missing order amounts, and invalid delivery times. Cancelled orders retain missing delivery times because those orders were not completed.

Cleaned CSV files are stored in `data/clean/`, and the SQLite database is stored at `data/instamart.db`.

### SQL and exploratory analysis

Seven SQL queries cover filtering and aggregation, joins, subqueries, common table expressions, window functions, and cumulative revenue.

Key results:

- Completed-order revenue: **INR 2,998,668.80**
- Customers with more than INR 8,000 in completed-order spend and more than 500 completed orders: 30
- Customers with no completed orders: 25
- Customers above average completed-order spend: 550
- Daily completed revenue records: 183

The pandas revenue cross-check matched the SQL total.

EDA files:

- `outputs/city_category_revenue_pivot.csv`
- `outputs/city_order_summary.csv`
- `outputs/completed_order_correlations.csv`
- `outputs/part_a_eda_report.txt`

Completed-order amounts are right-skewed, with a mean of INR 269.91 and median of INR 135. The correlation analysis shows only weak linear relationships among order amount, discount, quantity, and delivery time.

## Part B

### RFM segmentation

RFM features were calculated using completed orders as of `2025-10-01 00:00:00`.

- Customers with completed orders: 1,175
- Recency median: 19 days
- Recency range: 0–180 days
- Mean purchase frequency: 9.46
- Maximum purchase frequency: 33
- Mean monetary value: INR 2,552.06
- Maximum monetary value: INR 14,766.55

K-means was evaluated for K values from 2 through 8 using standardized RFM features. **K = 4** was selected because it had the strongest silhouette score among the evaluated choices (0.526) and offered four actionable customer groups. Inertia decreases as K increases, so the lowest inertia alone is not sufficient to choose the number of clusters.

The four profiles include a Premium segment, High-Frequency Regulars, and lower-spend or less-recently-active customer groups. Segment IDs are arbitrary cluster labels, not customer rankings. The Premium segment is defined as the cluster with the highest mean monetary value.

PCA explained approximately 85.2% of the standardized RFM variance in two components.

### Lapse prediction

The lapse cohort contained 1,156 customers who had completed an order before `2025-09-01`. A customer was labelled lapsed if they had no completed order from September 1 through September 30, 2025.

- Lapsed customers: 443 (38.32%)
- Not lapsed: 713
- Training rows: 924
- Test rows: 232

Predictive features use orders strictly before the cutoff. Segment labels are not used as model features. Preprocessing was fitted on training data only.

The tuned Random Forest was selected as the primary model:

- Best parameters: `max_depth=8`, `min_samples_leaf=10`
- Cross-validation F1: 0.6811
- Test accuracy: 0.737
- Test precision for lapsed customers: 0.659
- Test recall for lapsed customers: 0.652
- Test F1 for lapsed customers: 0.6554

The tuned Logistic Regression achieved a test F1 of 0.648 for the lapsed class.

### Oversampling experiment

The training data was oversampled by duplicating minority-class lapsed customers with replacement. The test set was left unchanged.

For Logistic Regression, oversampling changed the results as follows:

- Lapsed-class recall increased from 0.652 to 0.854.
- Lapsed-class precision changed from 0.644 to 0.589.
- Lapsed-class F1 increased from 0.648 to 0.697.

This trade-off means the model identifies more lapsing customers but also flags more customers who do not lapse. Campaign thresholds should reflect available budget and incentive margins.

The five leading Random Forest features by importance were frequency, monetary value, recency, average delivery time, and cancellation rate.

### Part B outputs and limitations

Generated files:

- `outputs/customer_segments.csv`
- `outputs/lapse_model.joblib`
- `outputs/retention_list.csv`
- `outputs/part_b_report.txt`

The saved model was reloaded and produced predictions identical to the original model's predictions.

The retention list ranks ten Premium customers in the test split by predicted lapse probability. Only one of these ten customers was labelled as actually lapsed.

**Important limitation:** The Premium flag is derived from whole-window RFM data, including the September label window. Therefore, the retention list is illustrative and is not a leakage-free evaluation of a prospective Premium retention campaign. The predictive features themselves use only pre-cutoff order data.

## Part C

Implementation and evaluation results will be documented after the support-ticket classification and triage tasks are completed.

## Part D

Implementation and evaluation results will be documented after the policy-grounded support and refund-agent workflow tasks are completed.

## Limitations and Next Steps

- The data is synthetic and may not represent real customer behaviour.
- Lapse predictions are estimates, not guarantees of future customer behaviour.
- Retention campaigns should account for campaign capacity, incentive cost, precision, and recall.
- The Premium retention list has a post-hoc segment-label limitation and should be treated as illustrative.
- Complete Parts C and D, document their evaluation results, and update the reproduction commands.
- Validate the entire workflow from a fresh environment before final submission.

## Part C — Support Ticket Triage with Guardrails

### Privacy and input guards

Processed 164 support tickets. Before calling the model, ticket text was redacted using the required patterns for email addresses, phone numbers, and order IDs.

| Privacy audit | Result |
|---|---:|
| Email substitutions | 55 |
| Phone substitutions | 91 |
| Order ID substitutions | 132 |
| Total substitutions | 278 |
| PII matches remaining after redaction | 0 |

Empty tickets were skipped (4), and overly long tickets were flagged (4). Neither group was sent to the model.

### Classification results

Both prompts were tested on the eligible tickets using the offline `mock_llm.py`.

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Classified tickets | 147 | 147 |
| Correct classifications | 126 | 147 |
| Accuracy on classified tickets | 85.714% | 100.000% |
| Total response tokens | 13,612 | 29,212 |
| Tokens per classified ticket | 92.60 | 198.72 |
| Transient retries recovered | 19 | 19 |

Few-shot prompting improved observed accuracy by about 14.29 percentage points, using 15,600 additional tokens across the run.

The zero-shot confusion table showed 11 fraud tickets and 10 technical tickets incorrectly classified as billing. The few-shot run classified all 147 classified tickets correctly.

### Output validation and budget demonstration

The final run produced:

| Status | Tickets |
|---|---:|
| `classified_json` | 135 |
| `classified_keyword` | 12 |
| `flagged_invalid_output` | 9 |
| `skipped_empty` | 4 |
| `flagged_long` | 4 |
| **Total** | **164** |

A zero-shot run with a token budget of 6,000 used 6,018 response tokens and marked 87 subsequent tickets `budget_exhausted`. The budget is checked before each model call, so a call already in progress can take the total slightly over the limit.

### Final routing

The few-shot run used a token budget of 100,000 and wrote `outputs/triage_results.csv`.

| Route | Tickets |
|---|---:|
| `support_queue` | 102 |
| `security_team` | 34 |
| `human_review` | 17 |
| `premium_desk` | 11 |
| **Total** | **164** |

The final output contains 164 rows. Fraud tickets are routed to the security team with high priority; premium customers with other classifiable tickets are routed to the premium desk with high priority. Empty, long, invalid-output, and budget-exhausted tickets are routed to human review.

### Output files

- `outputs/triage_results.csv`
- `outputs/triage_budget_demo.csv`

The model is simulated locally using `mock_llm.py`; token counts are estimates from the supplied mock implementation, not charges from a hosted service.

## Part D — Policy RAG Assistant and Refund Agent

### 1. Policy document chunking

The five policy documents were split into 40-word chunks with a 10-word overlap.

| Policy document | Number of chunks |
|---|---:|
| refund_policy.txt | 4 |
| delivery_policy.txt | 3 |
| escalation_policy.txt | 3 |
| payment_policy.txt | 3 |
| privacy_policy.txt | 2 |
| **Total** | **15** |

With zero overlap, the documents produced 13 chunks.

### 2. TF-IDF retrieval

The policy assistant uses scikit-learn's `TfidfVectorizer()` to convert policy chunks into numerical vectors and cosine similarity to rank the chunks relevant to a question.

Term Frequency (TF) measures how often a term occurs in a chunk. Inverse Document Frequency (IDF) gives less weight to common terms and more weight to distinctive terms. A limitation of TF-IDF is that it relies on word overlap, so semantically similar questions using different words may not retrieve the best passage.

For the query `refund policy late delivery`, the top three retrieved chunks all came from `refund_policy.txt`.

### 3. Policy RAG evaluation

| Evaluation | 10-word overlap | Zero overlap |
|---|---:|---:|
| Policy chunks | 15 | 13 |
| Expected document retrieved for answerable questions | 12/12 | 12/12 |
| Expected fact found in answer | 12/12 | 10/12 |
| Correct fallback for unsupported questions | 2/2 | 2/2 |

With zero overlap, the question “What delivery time makes a completed order eligible for a full refund?” failed the expected-fact check because the retrieved answer did not contain “more than 45 minutes.” The question about money returned after a failed but deducted payment also failed the expected-fact check because the answer did not contain “48 hours.”

This demonstrates that overlapping chunks can preserve facts that would otherwise be split across chunk boundaries.

Detailed results are saved in `outputs/policy_rag_eval.csv` and `outputs/policy_rag_eval_zero_overlap.csv`.

### 4. Refund tools and safety rules

The agent implements five tools: `get_order`, `search_policy`, `calculate_refund`, `issue_refund`, and `escalate_to_human`.

The agent looks up orders in the Part A SQLite database, retrieves relevant policy chunks, and calculates refunds using the business rules:

- Only completed orders qualify for refunds.
- Completed orders delivered in more than 45 minutes qualify for a full refund of the order amount.
- Orders delivered in 45 minutes or less receive no late-delivery refund.
- Refunds up to INR 1,000 can be issued automatically.
- Refunds above INR 1,000 are escalated to a human and are not issued automatically.
- Missing orders, cancelled orders, and ineligible orders are not refunded.

The refund ledger is stored in `outputs/refund_ledger.jsonl`. Agent runs are stored in `outputs/agent_runs.jsonl`, and checkpoint files are stored in `state/`.

### 5. Seven-scenario agent evaluation

| Scenario | Status | Refund amount (INR) | Model calls | Tokens used |
|---|---|---:|---:|---:|
| S1 | REFUND | 128.25 | 5 | 1,417 |
| S2 | REFUND | 135.00 | 5 | 1,417 |
| S3 | ESCALATE | 1,999.00 | 5 | 1,404 |
| S4 | DENIED | — | 4 | 1,021 |
| S5 | DENIED | — | 3 | 649 |
| S6 | NOT_FOUND | — | 2 | 280 |
| S7 | FALLBACK_MAX_ITERATIONS | — | 6 | 3,624 |

S1 and S2 received the expected refunds. S3 was escalated because its refund exceeded INR 1,000. S4 and S5 were denied because the orders were ineligible. S6 returned `NOT_FOUND`, and S7 stopped safely after six model calls.

### 6. Safety-limit demonstrations

| Demonstration | Result |
|---|---|
| S1 with `token_limit=200` | `FALLBACK_TOKEN_LIMIT` |
| S1 with `time_limit_seconds=0` | `FALLBACK_TIMEOUT` |
| S7 with the default token limit of 6,000 | `FALLBACK_MAX_ITERATIONS` |

The token-limit fallback stopped execution when the accumulated model tokens exceeded the limit. The timeout demonstration stopped before the first model call. The default S7 run completed all six model calls without exceeding the token budget.

### 7. Crash and resume

The S1 crash-and-resume demonstration simulated a crash immediately after the second tool execution, after its checkpoint had been saved.

The tools executed across the interrupted and resumed run were:

1. `get_order`
2. `search_policy`
3. `calculate_refund`
4. `issue_refund`

The resumed run completed with status `REFUND` and amount INR 128.25. No tool was repeated across the crash and resume.

### 8. ReAct versus plan-and-execute

The ReAct agent makes decisions step by step, using each tool observation to decide what to do next. In the measured runs, S1 required five model calls, S5 required three, and S6 required two. This approach adapts to unexpected observations, such as a missing order or a cancelled order, before proceeding to the next action.

A plan-and-execute agent could ask the model to generate a sequence of steps once and replan only when an error occurs. This could reduce model calls and token use for predictable cases, although generating a complete plan may require a longer initial response. Actual token savings would need to be measured with a plan-and-execute implementation.

For this refund workflow, I would choose ReAct because refund decisions depend on database observations and strict financial safety rules. Checking each observation before proceeding makes it easier to prevent an invalid refund and to stop safely when a tool reports an error.
