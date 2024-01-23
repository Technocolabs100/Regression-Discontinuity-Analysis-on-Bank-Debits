# Regression-Discontinuity-Analysis-on-Bank-Debits

**Report on the Effectiveness of Recovery Strategies in Financial Recovery**

**Introduction**

This report presents a detailed analysis of our bank's recovery strategies, focusing on their effectiveness at different expected recovery amount thresholds. The primary objective was to assess whether the additional investment in more intensive recovery strategies is justified by an increase in actual recovery amounts, particularly around the key $1000 threshold.

**Data Overview**

The dataset comprises records of recovery attempts, including the expected recovery amount, actual recovery amount, recovery strategy employed, and demographic details (age and sex) of the customers. The recovery strategies are tiered based on the expected recovery amount, with more resources allocated as the amount increases.

**Analysis and Findings**

1. **Relationship Between Expected and Actual Recovery Amounts**:
   - Linear regression analysis revealed a strong positive relationship between expected and actual recovery amounts.
   - The increase in actual recovery amount was consistently higher than the expected amount, supporting the effectiveness of the recovery strategies.

2. **Demographic Analysis Near the $1000 Threshold**:
   - Statistical tests showed no significant difference in the average age or sex distribution of customers just below and above the $1000 threshold.
   - This suggests that the differences in recovery amounts are due to the strategy rather than demographic factors.

3. **Impact of the $1000 Threshold**:
   - Regression models with a threshold indicator showed a significant change in the actual recovery amount at the $1000 mark.
   - The model indicated a discontinuity at this threshold, suggesting an increased effectiveness of the recovery strategy above $1000.

4. **Consistency Across Different Ranges**:
   - The significant impact of the $1000 threshold was consistent across different expected recovery amount ranges, including $900-$1100 and $950-$1050.
   - This consistency strengthens the conclusion that the recovery strategy becomes markedly more effective at the $1000 threshold.

5. **Residual Analysis**:
   - Residual plots and Q-Q plots indicated no major violations of regression assumptions.
   - Residuals did not show patterns that would suggest issues with the model's fit.

**Conclusion**

The analysis provides strong evidence that the bank's tiered recovery strategy is effective, particularly at the $1000 threshold. The additional investment in the recovery strategy for amounts above $1000 is justified by a corresponding increase in the actual recovery amount. It is recommended to continue with the current tiered approach and consider similar detailed analyses for other thresholds to optimize resource allocation across different recovery strategies.

**Recommendations for Further Analysis**:
- Explore machine learning models for predictive insights.
- Conduct a segmented analysis by customer demographics for more targeted strategies.
- Perform a detailed cost-benefit analysis of the recovery strategies.


