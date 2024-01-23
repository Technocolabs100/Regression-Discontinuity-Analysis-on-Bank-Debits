# Regression Discontinuity Analysis on Bank Debits

## Project Overview
This project explores the impact of recovery strategies implemented by a bank after declaring a debt uncollectable. The primary focus is on assessing whether assigning customers to a higher recovery plan, especially around the expected recovery amount of USD1000, results in a significant increase in actual recovery amounts. The analysis involves statistical tests, graphical exploratory data analysis, and regression modeling to gain insights into the effectiveness of recovery strategies.

## Objectives
• Examine how the bank assigns recovery plans at thresholds, particularly around USD1000.
• Ensure age and sex variables don't significantly differ above and below USD1000.
• Analyze actual recovery amounts to identify any discontinuity around the USD1000 threshold.
• Utilize statistical tests to check for discontinuities in recovery amounts across different expected recovery windows.
• Build regression models to estimate the impact of recovery programs.
• Validate the impact by confirming consistent results across different analysis windows.

## Project Structure
  **• Data:** The dataset includes columns such as expected_recovery_amount, actual_recovery_amount, recovery_strategy, age, and sex.

 **• Analysis Steps:**

**1. Data Transformation:** Apply Box-Cox transformation and standard scaling to numerical columns.
**2. Exploratory Data Analysis**: Visualize age as a function of expected recovery amount, focusing on the transition between Level 0 and Level 1.
**3. Statistical Tests:** Conduct t-tests for age and chi-square tests for sex to ensure similarity above and below the USD1000 threshold.
**4. Graphical Analysis:** Create scatter plots to explore the relationship between expected and actual recovery amounts.
**5. Regression Modeling:** Build regression models to quantify the impact of recovery strategies.

## Results
**Data Transformation**
• Applied Box-Cox transformation and standard scaling to numerical columns.

**Exploratory Data Analysis**
• Examined age as a function of expected recovery amount around the USD1000 threshold.

**Statistical Tests**

**Age vs. Expected Recovery Amount:**
•Conducted a t-test to compare average age above and below USD1000.
•Result: Significant difference in average age.

**Sex vs. Expected Recovery Amount:**
•Conducted a chi-square test to compare the percentage of male customers above and below USD1000.
•Result: No significant difference in gender distribution.

**Graphical Analysis**
• Created scatter plots to visualize the relationship between expected and actual recovery amounts.

**Regression Modeling**
• Built regression models to quantify the impact of recovery strategies.

## Conclusion
•Identified a significant impact of the recovery strategy around USD1000.
•Established a positive relationship between expected and actual recovery amounts.
•Emphasized the importance of weighing observed impact against associated costs for optimizing recovery strategies.

