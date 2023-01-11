import streamlit as st
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from src.sample_size_analysis import SampleSizeAnalysis
from src.bernoulli_model import BernoulliModel
from src.utils import return_parameter_uncertainties, perc
import matplotlib.ticker as mtick
import plotly.graph_objects as go

ssa = SampleSizeAnalysis()
model = BernoulliModel()

def AB_page():
    # Helper Methods
    def main_calculation(p0_input, lift_input, ptb_input, alpha_input, power_input, type_of_test, method):

        p1 = (1 + lift_input) * p0_input
        if method == 'Probability to be the best':
            return ssa.min_number_of_samples_prob_to_be_best(p0_input, p1, ptb_input)
        elif method == 'Single sample power method':
            return ssa.min_number_of_samples_normal_approx(u0=p0_input, u1=p1, alpha=alpha_input, power=power_input, test=type_of_test)
        elif method == 'Power Two Independent Samples':
            return ssa.min_number_of_samples_2_independent(u0=p0_input, u1=p1, alpha=alpha_input, power=power_input, test=type_of_test)
        elif method == "T-test":
            return ssa.sample_required_ttest(u0=p0_input, u1=p1, alpha=alpha_input, power=power_input)
        elif method == 'Expected loss':
            return 0
        else:
            pass
    def plot_func(n_samples, p0, lift, show_details):
        p1 = (1 + lift) * p0
        n0 = int(p0 * n_samples)
        n1 = int(p1 * n_samples)
        p0_samples = model.generate_posterior(n_total=n_samples, n_success=n0)
        p1_samples = model.generate_posterior(n_total=n_samples, n_success=n1)
        p0_median = np.median(p0_samples)
        p1_median = np.median(p1_samples)
        fig, ax = plt.subplots(figsize=(10,5))

        sns.kdeplot(p0_samples, fill=True, label='Control')
        sns.kdeplot(p1_samples, fill=True, label='Challenger')
        if show_details:
            plt.axvline(p0_median, c='red')
            plt.text(min(p0_samples)*1.5, .5, 'P0 median is {} %'.format(round(p0_median*100, 2)),
                     horizontalalignment='left', verticalalignment='center', transform=ax.transAxes,
                     bbox=dict(boxstyle="square",
                               facecolor="white")
                     )
            plt.axvline(p1_median, c='red')
            plt.text(min(p0_samples)*20, 0.5, 'P1 median is {} %'.format(round(p1_median*100, 2)),
                     horizontalalignment='left', verticalalignment='center', transform=ax.transAxes,
                     bbox=dict(boxstyle="square",
                               facecolor="white")
                     )

            plt.axvline(return_parameter_uncertainties(p0_samples)["lower_68_ci"], c='blue')
            plt.axvline(return_parameter_uncertainties(p0_samples)["upper_68_ci"], c='blue')

            plt.axvline(return_parameter_uncertainties(p1_samples)["lower_68_ci"], c='blue')
            plt.axvline(return_parameter_uncertainties(p1_samples)["upper_68_ci"], c='blue')

        plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        plt.xlabel("Conversion rate [%]")
        plt.legend()
        st.pyplot(fig)
        return p0_samples, p1_samples

    def plot_lift_sample(p0, alpha, power, ptb, min_sample, max_sample, min_lift, max_lift, method_plot):

        p_paid_low, nmin_paid_low, p_paid_high, nmin_paid_high = ssa.compute_resolution_graph_v2(
            p0_h=p0,
            n_shown=5000000,
            n_cohort_size=5000000,
            p_min_factor=0.1,
            p_max_factor=2.0,
            n_steps_p=5000,
            alpha=alpha,
            power=power,
            ptb=ptb,
            method=method_plot)
        if max_sample > 10**6:
            denom = 10**6
            unit = "M"
        else:
            denom = 10**3
            unit = "K"
        fig, ax = plt.subplots(figsize=(20, 12))
        y1 = 100 * (p_paid_low - p0) / p0
        y2 = 100 * (p_paid_high - p0) / p0

        # interactive
        layout = go.Layout(
            yaxis=dict(title='Percent improvement from Baseline',
                range=[min_lift, max_lift]),
            xaxis=dict(title= f'Cohort Size [{unit}]',
                range=[min_sample/denom, max_sample/denom])
        )
        fig2 = go.Figure(layout=layout)
        fig2.add_trace(go.Scatter(x=nmin_paid_high / denom, y=y2,
                                  fill="tozeroy",
                                  mode='lines',
                                  line_color='indigo',
                                  name='Upper',
                                  hovertemplate='for %{y:.2f} %lift,  %{x:.2f}K samples required'
                              ))
        fig2.add_trace(go.Scatter(x=nmin_paid_low / denom, y=y1,
                                  fill='tozeroy',
                                  mode='lines',
                                  line_color='indigo',
                                  name='Lower',
                                  hovertemplate='for %{y:.2f} %lift,  %{x:.2f}K samples required'

                       ))
        fig2.update_layout(title="Total experiment size vs minimum statistically detectable effect",
                           font=dict(
                               family="Courier New, monospace",
                               size=9,
                               color="RebeccaPurple"
                           ), showlegend=False
                        )
        st.write(fig2)
    ####################
    ### INTRODUCTION ###
    ####################

    row0_1,row0_2 = st.columns((2))
    with row0_1:
        st.title('Experiment Planning Dashboard')
    with row0_2:
        st.text("")
        st.subheader('Application')


    #################
    ### SELECTION ###
    #################

    st.sidebar.text('')
    st.sidebar.text('')
    st.sidebar.text('')

    ################
    ### ANALYSIS ###
    ################

    ### DATA EXPLORER ###
    ### TEAM ###
    row4_spacer1, row4_1, row4_spacer2 = st.columns((.2, 7.1, .2))
    with row4_1:
        st.subheader('Settings')
        ref_expander = st.expander("Reference for methods")
        with ref_expander:
            st.caption(
                   """
                   Single and two sample power method(min_number_of_samples) :A one and two-sample test implemented according to "Fundamentals of Biostatistics" by Bernard Rosner" \n
                   Probability to be the best                                :https://marketing.dynamicyield.com/ab-test-duration-calculator/ \n        
                   T-test (tt_ind_solve_power)                                :A function from statsmodels.stats.power package to compute the sample size based on the T-test 
                   """
            )
    row5_spacer1, row5_1, row5_spacer2, row5_2, row5_spacer3 = st.columns((.2, 2.3, .4, 4.4, .2))
    with row5_1:
        st.markdown('Enter details ')
        no_of_cohorts = st.number_input('Number of Cohorts', format='%d', value=3)
        method_selection = st.multiselect("Method", ['Probability to be the best', 'Expected loss', 'T-test',
                                                     'Single sample power method', 'Power Two Independent Samples'],
                                                      default='Probability to be the best')

        p0_input = st.number_input('Baseline Conversion (%)', format='%g', value=2.5)/100
        lift_input = st.number_input('Expected Lift (%)', format='%g', value=10)/100
        n_samples_input = 100_000
    with row5_2:
        show_details = st.checkbox("Show details")
        p0_samples, p1_samples = plot_func(n_samples_input, p0_input, lift_input, show_details=show_details)
        if show_details:
            p0_median = return_parameter_uncertainties(p0_samples)["median"]
            p1_median = return_parameter_uncertainties(p1_samples)["median"]
            delta0 = (p0_median - p1_median) / p0_median
            delta1 = (p1_median - p0_median) / p1_median

            st.metric('Median for p0', round(p0_median*100, 5), "%")
            st.metric('Median for p1', round(p1_median*100, 5), "%")

            st.write('For p0, the %68 lower and upper boundries are',
                     perc(return_parameter_uncertainties(p0_samples)["lower_68_ci"]),
                     perc(return_parameter_uncertainties(p0_samples)["upper_68_ci"]))
            st.write('For p1, the %68 lower and upper boundries are',
                     perc(return_parameter_uncertainties(p1_samples)["lower_68_ci"]),
                     perc(return_parameter_uncertainties(p1_samples)["upper_68_ci"]))

    ### STATS ###

    row6_1, row6_2 = st.columns((2))
    with row6_1:
        sse_expander = st.expander("Click here for Sample Size Estimation")
        with sse_expander:
            st.subheader('Sample size estimation')
            ptb_input = st.number_input('Target probability to be the best (%) to be used for Probability to be the best method',
                                        value=95)/100
            DAILY_AVERAGE = st.number_input('Expected daily average per cohorts', value=1000)

            dict_result = dict()
            power_input = None
            alpha_input = None
            type_of_test = None
            if "Single sample power method" or "Power Two Independent Samples" in method_selection:
                power_input = st.number_input('Power (%) to be used for the Power method', format='%g', value=80)/100
                alpha_input = st.number_input('Alpha (%) to be used for the Power method', format='%g', value=5)/100
                type_of_test = st.selectbox('Type of test?', ("one-sided", "two-sided"))
            for method in method_selection:
                ss_req = main_calculation(p0_input, lift_input, ptb_input, alpha_input=alpha_input, power_input=power_input,
                                          type_of_test=type_of_test, method=method)
                dict_result[str(method)] = int(ss_req)
            df = pd.DataFrame(dict_result.items(), columns=["Method", "Sample Size per Cohort"])
            df["Total Samples Required"] = df["Sample Size per Cohort"] * no_of_cohorts
            df["Days Required"] = df["Sample Size per Cohort"]/DAILY_AVERAGE
            df["Days Required"] = df["Days Required"].astype(int)
            df["Days Required"] = df["Days Required"]
            df["Sample Size per Cohort"] = df["Sample Size per Cohort"].map('{:,}'.format)
            df["Total Samples Required"] = df["Total Samples Required"].map('{:,}'.format)

            hide_table_row_index = """
                        <style>
                        thead tr th:first-child {display:none}
                        tbody th {display:none}
                        </style>
                        """
            # Inject CSS with Markdown
            st.markdown(hide_table_row_index, unsafe_allow_html=True)

            # Display a st
            st.table(df.sort_values(by="Sample Size per Cohort", ascending=True))

            @st.cache
            def convert_df(df):
                return df.to_csv().encode('utf-8')
            csv = convert_df(df)
            st.download_button(
                "Press to Download",
                csv,
                "ab_testing_results.csv",
                "text/csv",
                key='download-csv'
            )
    with row6_2:
        ls_expander = st.expander("Click here for Lift vs sample size simulation")
        with ls_expander:
            ### Lift vs sample size simulation ###
            st.subheader('Lift vs sample size simulation')
            method_plot_selection = st.selectbox("Which method?",
                                             ['Single sample power method', 'Probability to be the best', 'Expected loss',
                                              'Power Two Independent Samples', 'T-test'], key='method_single')

            baseline_conversion = st.number_input('Baseline conversion [%]', format='%g', value=p0_input*100) / 100
            alpha_sim = None
            power_sim = None
            ptb_sim = None
            if ((method_plot_selection == "Single sample power method") or
                (method_plot_selection == "Power Two Independent Samples")):
                alpha_sim = st.number_input('Alpha [%]', format='%g', value=5)/100
                power_sim = st.number_input('Power [%]', format='%g', value=80) / 100
            if method_plot_selection == 'Probability to be the best':
                ptb_sim = st.number_input('Probability to be the best [%]', format='%g', value=80) / 100

            min_lift = 1
            max_lift = st.number_input('Max Lift [%]', format='%d', value=5)
            min_sample = ssa.min_number_of_samples_2_independent(u0=baseline_conversion, u1=(1+max_lift)*baseline_conversion)
            max_sample = ssa.min_number_of_samples_2_independent(u0=baseline_conversion, u1=(1+0.01)*baseline_conversion)

            plot_lift_sample(p0=baseline_conversion, alpha=alpha_sim,power=power_sim, ptb=ptb_sim,min_sample=min_sample,
                             max_sample=max_sample, min_lift=-max_lift, max_lift=max_lift, method_plot=method_plot_selection)