import numpy as np

def return_parameter_uncertainties(samples):
    """
    Return the uncertainties of the sampled variable
    """

    x_low68, x_med, x_high68 = np.percentile(samples, [16, 50, 84])
    x_low95, x_med, x_high95 = np.percentile(samples, [2.5, 50, 97.5])
    x_low99, x_med, x_high99 = np.percentile(samples, [0.5, 50, 99.5])

    results = {
        "lower_68_ci": x_low68,
        "lower_95_ci": x_low95,
        "lower_99_ci": x_low99,
        "upper_68_ci": x_high68,
        "upper_95_ci": x_high95,
        "upper_99_ci": x_high99,
        "median": x_med,
        "error_68": np.mean([abs(x_med - x_low68), abs(x_high68 - x_med)]),
        "error_95": np.mean([abs(x_med - x_low95), abs(x_high95 - x_med)]),
        "error_99": np.mean([abs(x_med - x_low99), abs(x_high99 - x_med)])
    }

    return results

def compute_bayesian_statistics(samples_control, samples_test1, samples_test2, n_round=2):
    """

    """

    lift1 = ((samples_test1-samples_control)/samples_control)*100
    lift2 = ((samples_test2-samples_control)/samples_control)*100

    # neg_lift1 = lift1[lift1 <0]
    # neg_lift2 = lift2[lift2 < 0]
    neg_lift1 = np.minimum(lift1, np.zeros(len(lift1)))
    neg_lift2 = np.minimum(lift2, np.zeros(len(lift2)))

    results_lift1 = return_parameter_uncertainties(lift1)
    results_lift2 = return_parameter_uncertainties(lift2)
    results_lift1_neg = return_parameter_uncertainties(neg_lift1)
    results_lift2_neg = return_parameter_uncertainties(neg_lift2)

    results = {
        'P(test1>control) [%]': round((np.sum(samples_test1 > samples_control)/len(samples_test1))*100, n_round),
        'P(test2>control) [%]': round((np.sum(samples_test2 > samples_control)/len(samples_test1))*100, n_round),
        'P(Control is Best) [%]': round((np.sum((samples_control > samples_test1) & (samples_control > samples_test2))/len(samples_control))*100,n_round),
        'P(test1 is Best) [%]': round((np.sum((samples_test1 > samples_control) & (samples_test1 > samples_test2))/len(samples_control))*100,n_round),
        'P(test2 is Best) [%]': round((np.sum((samples_test2 > samples_test1) & (samples_test2 > samples_control))/len(samples_control))*100,n_round),
        'Lift1 median [%]': round(results_lift1['median'],n_round),
        'Lift1 95% CI range [%]': [round(results_lift1['lower_95_ci'],n_round), round(results_lift1['upper_95_ci'],n_round)],
        'Lift2 median [%]': round(results_lift2['median'], n_round),
        'Lift2 95% CI range [%]': [ round(results_lift2['lower_95_ci'],n_round), round(results_lift2['upper_95_ci'], n_round)],
        'Expected loss 1 median [%]': round(results_lift1_neg['median'], n_round),
        'Expected loss 1 95% CI range [%]': [round(results_lift1_neg['lower_95_ci'], n_round),
                                   round(results_lift1_neg['upper_95_ci'], n_round)],
        'Expected loss 2 median [%]': round(results_lift2_neg['median'], n_round),
        'Expected loss 2 95% CI range [%]': [round(results_lift2_neg['lower_95_ci'], n_round),
                                           round(results_lift2_neg['upper_95_ci'], n_round)]
    }

    return results

def perc(number, decimal=2, string=True):
    if string == False:
        return round(number * 100, decimal)
    else:
        return str(round(number * 100, 2))+"%"