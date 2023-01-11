import numpy as np
from scipy.special import gammaln
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.stats import norm
from tqdm import tqdm
from statsmodels.stats.power import tt_ind_solve_power





class SampleSizeAnalysis:
    """
    This module computes the required sample size of
    a binomial process. Using different normal approximations. For reference, examine:

    For the one-sample estimates refer to:
    Chapter 7 - Hypothesis Testing, One-Sample inference, Bernard Rosner's Fundamentals of Biostatistics

    TODO: add the reference to the two-sample estimates
    """

    def __init__(self):
        self.p0 = 0.0
        self.dp0 = 0.0



    def normal_dist(self, x, mu=0.0, sigma=1.0):
        """
        Defining the normal distribution.

        @param x: the variable
        @param mu: the mean
        @param sigma: the standard deviation
        """

        z = (x - mu) / sigma

        return np.exp(-0.5 * z ** 2) / np.sqrt(2.0 * np.pi * sigma ** 2)

    def determine_critical_point(self, alpha=0.05):
        """
        This function determines the critical points
        needed to evaluate the significance level of a normal-distribution.

        @param alpha: the desired significance level
        """

        area = 1 - alpha
        func = lambda x: np.abs(quad(self.normal_dist, -1.0 * x, 1.0 * x)[0] - area)

        res = minimize(func, 0.0, method="SLSQP")

        return res

    def determine_one_sided_critical_point(self, area):
        """
        Determine the critical point for a z-test using minimization technique

        @param area: Typically, Area = 1.0-p_value
        """

        func = lambda x: np.abs(quad(self.normal_dist, -np.inf, 1.0 * x)[0] - area)
        res = minimize(func, 0.0, method="SLSQP").x[0]

        return res

    def determine_power_u1_greater_u0(self, u0, u1, s0n, s1n, alpha):
        """
        Determine the power of the comparison P(u2 > u1)

        @param u0: The mean of the null hypothesis
        @param u1: the mean of the alternate hypothesis
        @param s0n: The stdev of the null hypothesis, including 1/Sqrt(N) factor for proportions
        @param s1n: The stdev of the alternate hypothesis, including 1/Sqrt(N) factor for proportions
        @param alpha: the p-value
        """

        res = self.determine_critical_point(alpha)
        zc = res.x[0]

        z0 = (u1 - u0 - zc * s0n) / s1n

        power = norm.cdf(z0)
        return power

    def determine_power_u1_less_u0(self, u0, u1, s0n, s1n, alpha):
        """
        Determine the power of the comparison P(u1 < u0)


        @param u0: Mean of the null-hypothesis
        @param u1: Mean of the alternate hypothesis
        @param s1n: Stdev of the alternate hypothesis, including 1/Sqrt(N) factor for proportions
        @param s0n: Stdev of the null hypothesis, including 1/Sqrt(N) factor for proportions
        @param alpha: desired p-value
        """

        res = self.determine_critical_point(alpha)
        zc = res.x[0]

        z0 = (u0 - u1 - zc * s0n) / s1n

        beta = norm.cdf(z0)
        power = beta

        return power

    def determine_power_u2_not_equal_u1(self, u0, u1, s0n, s1n, alpha, test='two-sided'):
        """
        Determine the power for the comparison P(u2 != u1) for a 1-sample test

        Note: Here we use the 2-tail test

        Reference: Chapter 7 - Hypothesis Testing, One-Sample inference, Bernard Rosner's Fundamentals of Biostatistics

        @param u0: Mean of the null-hypothesis
        @param u1: Mean of the alternate hypothesis
        @param s0n: Stdev of the null hypothesis, including 1/Sqrt(N) factor for proportions
        @param s1n: Stdev of the alternate hypothesis, including 1/Sqrt(N) factor for proportions
        @param alpha: desired p-value
        """

        if test == 'two-sided':
            p_tail1 = self.determine_power_u1_less_u0(u0, u1, s0n, s1n, 0.5 * alpha)
            p_tail2 = self.determine_power_u1_greater_u0(u0, u1, s0n, s1n, 0.5 * alpha)
        elif test == 'one-sided':
            p_tail1 = self.determine_power_u1_less_u0(u0, u1, s0n, s1n, alpha)
            p_tail2 = self.determine_power_u1_greater_u0(u0, u1, s0n, s1n, alpha)

        power = p_tail1 + p_tail2

        return power

    def calculate_power(self, u0, u1, s0n, s1n, alpha, test='two-sided'):
        """
        Note: This is the power calculation for a 1-sample test

        Compute the power for H0 vs H1 for a desired p-value (alpha)

        @param u0: Mean of the null-hypothesis
        @param u1: Mean of the alternate hypothesis
        @param s0n: Stdev of the alternate hypothesis
        @param s1n: Stdev of the alternate hypothesis
        @param alpha: desired p-value
        """

        power = self.determine_power_u2_not_equal_u1(u0, u1, s0n, s1n, alpha, test)

        return power

    def calculate_power_approx(self, u0, u1, s0n, s1n, alpha):
        """
        Using the normal approximation

        @param u0: Mean of the null-hypothesis
        @param u1: Mean of the alternate hypothesis
        @param s0n: Stdev of the alternate hypothesis with /Sqrt(N) for proportions
        @param s1n: Stdev of the alternate hypothesis with / sqrt(N) for proportions
        @param alpha: desired p-value
        """

        res = self.determine_critical_point(alpha)
        zc = res.x[0]

        z0 = np.abs(u0 - u1) / s1n - zc * (s0n / s1n)

        power = norm.cdf(z0)

        return power

    def compute_power_vs_sample_size(self, u0, u1, s0, s1, alpha, nmin, nmax, steps=10 ** 3):
        """

        This function computes the number

        @param u0: Mean of the null-hypothesis
        @param u1: Mean of the alternate hypothesis
        @param s0: Stdev of the alternate hypothesis
        @param s1: Stdev of the alternate hypothesis
        @param alpha: desired p-value
        @param nmin: minimum number of samples to use
        @param nmax: max number of samples to use
        @param steps: The incriments for increasing test sample sizes
        """

        niter = range(nmin, nmax, steps)

        power_array = np.array([self.calculate_power(u0, u1, s0 / np.sqrt(n), s1 / np.sqrt(n), alpha) for n in niter])

        return power_array, niter

    def min_number_of_samples(self, u0, u1, s0=None, s1=None, alpha=0.05, power=0.9, test='two-sided'):
        """

        TODO: Add the approximation for two-sample tests

        Computes the minimum number of samples needed to achieve a certain level of power
        for a two-sided test

        @param u0: Mean of the null-hypothesis
        @param u1: Mean of the alternate hypothesis
        @param s0: Stdev of the control hypothesis
        @param s1: Stdev of the alternate hypothesis
        @param alpha: desired p-value
        @param power: The desired power value
        """

        if s0 is None:
            s0 = np.sqrt(u0 * (1 - u0))

        if s1 is None:
            s1 = np.sqrt(u1 * (1 - u1))

        func = lambda x: np.abs(power - self.calculate_power(u0, u1, s0 / np.sqrt(x), s1 / np.sqrt(x), alpha, test))

        # Use the approximate function to find a starting point for the optimization
        n0 = self.min_number_of_samples_normal_approx(u0, u1, s0, s1, alpha, power, test)

        min_samples = minimize(func, n0, method="nelder-mead").x[0]

        return min_samples

    def min_number_of_samples_u2_greater(self, u0, u1, s0, s1, alpha, power):
        """
        Computes the minimum number of samples needed to achieve a certain level of power
        for a one-sided test
        @param u0: Mean of the null-hypothesis
        @param u1: Mean of the alternate hypothesis
        @param s0: Stdev of the alternate hypothesis
        @param s1: Stdev of the alternate hypothesis
        @param alpha: desired p-value
        @param power: The desired power value

        """

        func = lambda x: np.abs(
            power - self.determine_power_u1_greater_u0(u0, u1, s0 / np.sqrt(x), s1 / np.sqrt(x), alpha))

        # Use the approximate function to find a starting point for the optimization
        n0 = self.min_number_of_samples_normal_approx(u0, u1, s0, s1, alpha, power)

        min_samples = minimize(func, n0, method="nelder-mead").x[0]

        return min_samples

    def min_number_of_samples_2_independent(self, u0, u1, s0=None, s1=None, alpha=0.05, power=0.90, test='one-sided'):

        if s0 is None:
            s0 = np.sqrt(u0 * (1 - u0))
        if s1 is None:
            s1 = np.sqrt(u1 * (1 - u1))

        beta = 1 - power
        z_1_m_b = self.determine_one_sided_critical_point(1 - beta)
        if test == 'two-sided':
            z_1_m_0p5_alpha = self.determine_one_sided_critical_point(1.0 - alpha * 0.5)
        elif test == 'one-sided':
            z_1_m_0p5_alpha = self.determine_one_sided_critical_point(1.0 - alpha)

        std_dev_mult = s0 ** 2 + s1 ** 2
        z_calc = (z_1_m_b + z_1_m_0p5_alpha) ** 2
        delta = abs(u1 - u0) ** 2
        N = std_dev_mult * z_calc / delta

        return N

    def min_number_of_samples_normal_approx(self, u0, u1, s0=None, s1=None, alpha=0.05, power=0.90, test='two-sided'):
        """

        TODO: Two sample vs one sample tests

        Compute the minimum number of samples needed for a certain power.

        Reference: Chapter 7 - Hypothesis Testing, One-Sample inference, Bernard Rosner's Fundamentals of Biostatistics
        """

        if s0 is None:
            s0 = np.sqrt(u0 * (1 - u0))

        if s1 is None:
            s1 = np.sqrt(u1 * (1 - u1))

        beta = 1 - power
        z_1_m_b = self.determine_one_sided_critical_point(1 - beta)

        if test == 'two-sided':
            z_1_m_0p5_alpha = self.determine_one_sided_critical_point(1.0 - alpha * 0.5)
        elif test == 'one-sided':
            z_1_m_0p5_alpha = self.determine_one_sided_critical_point(1.0 - alpha)

        u_diff = abs(u0 - u1)
        num = (z_1_m_b * s1 + z_1_m_0p5_alpha * s0)
        denom = u_diff
        N = (num / denom) ** 2

        return N

    def min_number_of_samples_prob_to_be_best(self, p0, p1, target_prob_to_be_best):
        """
        The function which computes the minimum number of samples needed for a given prob-to-be best assuming a normal distribution


        Reference: [TODO: upload the notes for this derivation]
        """

        s12 = np.sqrt(p0 * (1 - p0) + p1 * (1 - p1))
        zc = self.determine_one_sided_critical_point(target_prob_to_be_best)
        delta = p1 - p0

        min_samples = (s12 * zc / delta) ** 2

        return min_samples

    def log_beta(self, x, y):
        """
        The natural logarithm of the beta function

        Beta(x,y) = Gamma(x)Gamma(y)/Gamma(x+y)

        @param x: real number
        @param y: real number
        """

        ln_beta = gammaln(x) + gammaln(y) - gammaln(x + y)

        return ln_beta

    def func(self, n_samples1, n1, n_samples2, n2):

        log_r = self.log_beta(n1 + 1, n_samples1 - n1 + 1) + self.log_beta(n2 + 1, n_samples2 - n2 + 1)  # P(x1 != x2)
        log_r -= self.log_beta(n1 + n2 + 1, n_samples1 + n_samples2 - n1 - n2 + 1)  # # P(x1 = x2)
        r = np.exp(log_r)

        px1x2 = r / (1 + r)
        px12 = 1 - px1x2

        print(' x1!=x2 / p(x1=x2)', r)

        return px12, px1x2

    def compute_resolution_graph(self,
                                 p0_h,
                                 n_shown,
                                 n_cohort_size,
                                 n_proposed_min,
                                 n_proposed_max,
                                 n_steps_cohort=30,
                                 n_steps_p=30,
                                 alpha=0.05,
                                 power=0.9):
        """

        This function generates a plot of the resolution achievable for experiments for different

        Expected funnel process:
        n_cohort_size -> n_shown -> n_converted

        Note: n_converted = p0_h*n_shown

        @param p0_h: The baseline probability for the conversion rates ( n_shown -> n_converted )
        @param n_shown: the start of the funnel from which the conversion rate is measured.
        @param n_cohort_size: The total size of the cohort for the baseline group.
        @param n_proposed_min: The smallest size of the proposed cohort size
        @param n_proposed_max: The largest value of the proposed cohort size
        @param n_steps_cohort: The number of steps to use for iterating over the cohort size.
        @param n_steps_p: The number of steps to use for iterating over the
        @param alpha: The level of significance of the effect.
        @param power: The target power to test for in generating the exclusion area.
        """

        n_proposed_vector = np.linspace(n_proposed_min, n_proposed_max, num=n_steps_cohort)  # resolution on the x-axis

        min_resolution_low_vec = np.zeros(len(n_proposed_vector))
        min_resolution_high_vec = np.zeros(len(n_proposed_vector))

        # We take the value for the control
        f = n_shown / n_cohort_size  # The fraction of shown to the cohort size

        p0_h_min = p0_h / 10.0  # The smallest conversion rate to examine
        p0_h_max = 10 * p0_h

        print('p_min:', p0_h_min)
        print('p_max:', p0_h_max)

        s0_h = np.sqrt(p0_h * (1 - p0_h))

        step_h_min = (p0_h - p0_h_min) / n_steps_p
        step_h_max = (p0_h_max - p0_h) / n_steps_p

        p_lower = np.array([p0_h - (k + 1) * step_h_min for k in range(n_steps_p)])
        p_higher = np.array([p0_h + (k + 1) * step_h_max for k in range(n_steps_p)])

        for i in tqdm(range(len(n_proposed_vector))):

            n_proposed = n_proposed_vector[i]

            low_n_min_samples_array = np.zeros(n_steps_p)
            high_n_min_samples_array = np.zeros(n_steps_p)

            for k in range(n_steps_p):
                p2_k = p_lower[k]
                s2_k = np.sqrt(p2_k * (1 - p2_k))

                low_n_min_samples_array[k] = self.min_number_of_samples_normal_approx(u0=p0_h,
                                                                                      u1=p2_k,
                                                                                      s0=s0_h,
                                                                                      s1=s2_k,
                                                                                      alpha=alpha,
                                                                                      power=power)

                p2_k = p_higher[k]
                s2_k = np.sqrt(p2_k * (1 - p2_k))
                high_n_min_samples_array[k] = self.min_number_of_samples_normal_approx(u0=p0_h,
                                                                                       u1=p2_k,
                                                                                       s0=s0_h,
                                                                                       s1=s2_k,
                                                                                       alpha=alpha,
                                                                                       power=power)

            # Convert the estimated size of the 'shown needed' to the cohort size
            n_cohort_size_array_low = low_n_min_samples_array / f
            n_cohort_size_array_high = high_n_min_samples_array / f

            min_resolution_low = min(np.abs(p_lower[n_cohort_size_array_low < n_proposed] - p0_h))
            min_resolution_high = min(np.abs(p_higher[n_cohort_size_array_high < n_proposed] - p0_h))

            min_resolution_low_vec[i] = min_resolution_low
            min_resolution_high_vec[i] = min_resolution_high

        return n_proposed_vector, min_resolution_low_vec, min_resolution_high_vec

    def compute_bayesian_metrics(self, samples1, samples2):
        """
        This function computes
        """

        # Probability to beat

        return None

    def compute_resolution_graph_v2(self,
                                    p0_h,
                                    n_shown=None,
                                    n_cohort_size=None,
                                    p_min_factor=0.1,
                                    p_max_factor=10.0,
                                    n_steps_p=100,
                                    alpha=0.05,
                                    power=0.9,
                                    approximation=True):

        s0_h = np.sqrt(p0_h * (1 - p0_h))

        p_max = p_max_factor * p0_h
        p_min = p_min_factor * p0_h

        p_higher_values = np.linspace(p0_h, p_max, num=n_steps_p)
        p_lower_values = np.linspace(p_min, p0_h, num=n_steps_p)

        s_higher_values = np.sqrt(p_higher_values * (1 - p_higher_values))
        s_lower_values = np.sqrt(p_lower_values * (1 - p_lower_values))

        if approximation is True:
            nmin_lower_values = self.min_number_of_samples_normal_approx(u0=p0_h,
                                                                         u1=p_lower_values,
                                                                         s0=s0_h,
                                                                         s1=s_lower_values,
                                                                         alpha=alpha,
                                                                         power=power)

            nmin_higher_values = self.min_number_of_samples_normal_approx(u0=p0_h,
                                                                          u1=p_higher_values,
                                                                          s0=s0_h,
                                                                          s1=s_higher_values,
                                                                          alpha=alpha,
                                                                          power=power)
        else:
            nmin_lower_values = self.min_number_of_samples(u0=p0_h,
                                                           u1=p_lower_values,
                                                           s0=s0_h,
                                                           s1=s_lower_values,
                                                           alpha=alpha,
                                                           power=power)

            nmin_higher_values = self.min_number_of_samples(u0=p0_h,
                                                            u1=p_higher_values,
                                                            s0=s0_h,
                                                            s1=s_higher_values,
                                                            alpha=alpha,
                                                            power=power)

        # Now we convert these nmin values to the cohort size
        if n_shown is not None and n_cohort_size is not None:
            f = n_shown / n_cohort_size
        else:
            f = 1.0

        nmin_lower_values = nmin_lower_values / f
        nmin_higher_values = nmin_higher_values / f

        return p_lower_values, nmin_lower_values, p_higher_values, nmin_higher_values

    def sample_required_ttest(self, u0, u1, alpha=0.05, power=0.90):

        prop1 = u0
        prop2 = u1
        diff = abs(prop1 - prop2)
        s_prop1 = np.sqrt(prop1 * (1 - prop1))
        s_prop2 = np.sqrt(prop2 * (1 - prop2))
        z = diff / np.sqrt((s_prop2 ** 2 + s_prop1 ** 2))
        n_ttest = tt_ind_solve_power(effect_size=z,
                                     alpha=alpha,
                                     power=power,
                                     alternative='larger')
        return n_ttest

    def compute_resolution_graph_v2(self,
                                    p0_h,
                                    n_shown=None,
                                    n_cohort_size=None,
                                    p_min_factor=0.1,
                                    p_max_factor=10.0,
                                    n_steps_p=100,
                                    alpha=0.05,
                                    power=0.9,
                                    ptb=0.80,
                                    method="Power"):

        s0_h = np.sqrt(p0_h * (1 - p0_h))

        p_max = p_max_factor * p0_h
        p_min = p_min_factor * p0_h

        p_higher_values = np.linspace(p0_h, p_max, num=n_steps_p)
        p_lower_values = np.linspace(p_min, p0_h, num=n_steps_p)

        s_higher_values = np.sqrt(p_higher_values * (1 - p_higher_values))
        s_lower_values = np.sqrt(p_lower_values * (1 - p_lower_values))

        if method == "Single sample power method":
            nmin_lower_values = self.min_number_of_samples_normal_approx(u0=p0_h,
                                                                         u1=p_lower_values,
                                                                         s0=s0_h,
                                                                         s1=s_lower_values,
                                                                         alpha=alpha,
                                                                         power=power)

            nmin_higher_values = self.min_number_of_samples_normal_approx(u0=p0_h,
                                                                          u1=p_higher_values,
                                                                          s0=s0_h,
                                                                          s1=s_higher_values,
                                                                          alpha=alpha,
                                                                          power=power)
        elif method == "Probability to be the best":
            nmin_lower_values = self.min_number_of_samples_prob_to_be_best(p0=p0_h, p1=p_lower_values,
                                                                           target_prob_to_be_best=ptb)

            nmin_higher_values = self.min_number_of_samples_prob_to_be_best(p0=p0_h, p1=p_higher_values,
                                                                            target_prob_to_be_best=ptb)
        elif method == "T-test":
            nmin_lower_values = self.sample_required_ttest(u0=p0_h, u1=p_lower_values,
                                                           alpha=alpha, power=power)

            nmin_higher_values = self.sample_required_ttest(u0=p0_h, u1=p_higher_values,
                                                            alpha=alpha, power=power)

        elif method == "Power Two Independent Samples":
            nmin_lower_values = self.min_number_of_samples_2_independent(u0=p0_h, u1=p_lower_values, alpha=alpha,
                                                                         power=power,
                                                                         test="two-sided")
            nmin_higher_values = self.min_number_of_samples_2_independent(u0=p0_h, u1=p_higher_values, alpha=alpha,
                                                                          power=power,
                                                                          test="two-sided")
        # Now we convert these nmin values to the cohort size
        if n_shown is not None and n_cohort_size is not None:
            f = n_shown / n_cohort_size
        else:
            f = 1.0
        nmin_lower_values = nmin_lower_values / f
        nmin_higher_values = nmin_higher_values / f

        return p_lower_values, nmin_lower_values, p_higher_values, nmin_higher_values


if __name__ == "__main__":
    pass
