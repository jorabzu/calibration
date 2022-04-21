# calibration
A small package with calibration tools.


## How to train a calibrator:



There are three calibrators that you can use train in this package: Logistic, Isotonic and Beta calibrators. You can easily train them by running the following lines:

```
import calibration as cal

logistic_calibrator = cal.LogisticCalibrator.train(y_pred_scores_uncalibrated,y_true, positive_class)
isotonic_calibrator = cal.IsotonicCalibrator.train(y_pred_scores_uncalibrated,y_true, positive_class)
beta_calibrator = cal.BetaCalibrator.train(y_pred_scores_uncalibrated,y_true, positive_class)

```

## Calibration maps

Let's suppose that you have trained a logistic calibrator. In order to obtain its calibration map, you can use 

```
logistic_calibrator.plot_calibration_map()
```

## Obtaining calibrated scores:

Scores can be calibrated with the following line: 


```
logistic_calibrated_scores = logistic_calibrator.calibrate(uncalibrated_scores)
```

### Reliability diagram and metrics

Once you have calibrated scores, you can plot a reliability diagram and compute some metrics:

```
frac_true, mean_pred_scores = cal.reliability_diagram(y_true,logistic_calibrated_scores, positive_class: 1,n_bins: int = 10,bin_type: str = "uniform",
cal.plot_reliability_diagram(frac_true, mean_pred_scores)


log_loss = cal.log_loss(y_true,logistic_calibrated_scores,positive_class=1)
brier_score = cal.brier_score_loss(y_true,logistic_calibrated_scores,positive_class=1)
ece = cal.binary_ece(y_true,logistic_calibrated_scores,positive_class=1, n_bins = 10, bin_type = "uniform")
mce = cal.binary_mce(y_true,logistic_calibrated_scores,positive_class=1, n_bins = 10, bin_type = "uniform")
```



