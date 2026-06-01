# Version 1.3 Extension: Threshold Sensitivity Analysis

This extension adds threshold sensitivity analysis to the Text Classification to QCA Analysis Tool.

## Purpose

QCA results can be sensitive to calibration thresholds. This module tests multiple crisp-set thresholds and reports whether truth-table configurations and sufficient solutions remain stable.

## Main outputs

- threshold_sensitivity_summary.csv
- threshold_sensitivity_details.csv
- threshold_sensitivity_solutions.csv
- configuration_stability_table.csv

## Interpretation

A robust configuration should remain sufficient across multiple nearby thresholds. If a configuration appears only under one threshold, it should be interpreted more cautiously.

## Recommended UI placement

Add the results to a new tab named:

```text
Threshold sensitivity
```

under the QCA Results section.
