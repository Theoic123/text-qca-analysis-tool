# Supervised Classifier Extension

This extension adds a supervised text classifier to the Text Classification to QCA Analysis Tool.

## Method

The implementation uses:

- TF-IDF character n-gram features
- one-vs-rest binary Logistic Regression
- one classifier per QCA condition

This design is lightweight and deployment-friendly. It does not require GPU or BERT fine-tuning.

## Labeled Data Format

The manually labeled training file should contain one binary 0/1 column for each condition:

```csv
case_id,text,dissatisfaction,policy_demand,coproduction_request
1,居民对停车收费不透明非常不满。,1,0,0
2,请说明申请补贴需要哪些材料。,0,1,0
3,我们愿意参加社区志愿服务。,0,0,1
```

## Output

The classifier outputs probability scores in the same format as the prototype-based scorer:

```csv
case_id,text,outcome,dissatisfaction_score,policy_demand_score,coproduction_request_score
```

These scores can then be passed into the existing calibration and QCA modules.
