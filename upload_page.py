import streamlit as st
import json
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
import scipy.stats
from src import utils
from matplotlib import pyplot as plt
from src.utils import perc, return_parameter_uncertainties
from src.bernoulli_model import BernoulliModel
import seaborn as sns
import os
from src.help import write_json


model = BernoulliModel()

def upload_page():

    fig = plt.figure(figsize=(5, 2.5))  # try different values
    ax = plt.axes()

    frame1 = plt.gca()
    master_json = {}

    def formatter(x, round_dec=5, to_int=False):
        if to_int:
            return int(round(x * 100, round_dec))
        else:
            return str(round(x * 100, round_dec))

    def uncertainty_func(stats_raw):

        control_upper, control_lower, control_unc = return_parameter_uncertainties(stats_raw[0])["upper_95_ci"], return_parameter_uncertainties(stats_raw[0])["lower_95_ci"], return_parameter_uncertainties(stats_raw[0])["error_68"]
        test1_upper, test1_lower, test1_unc = return_parameter_uncertainties(stats_raw[1])["upper_95_ci"], return_parameter_uncertainties(stats_raw[1])["lower_95_ci"], return_parameter_uncertainties(stats_raw[1])["error_68"]
        test2_upper, test2_lower, test2_unc = return_parameter_uncertainties(stats_raw[2])["upper_95_ci"], return_parameter_uncertainties(stats_raw[2])["lower_95_ci"], return_parameter_uncertainties(stats_raw[2])["error_68"]

        stats_dict = {'Index': ['Baseline', 'Test 1', 'Test 2'],
                      'Median CR [%]': [formatter(np.median(stats_raw[0])), formatter(np.median(stats_raw[1])), formatter(np.median(stats_raw[2]))],
                      'CR Uncertainty [%]': [formatter(control_unc), formatter(test1_unc), formatter(test2_unc)],
                      '95% Confidence interval [%]': [[formatter(control_lower, to_int=True), formatter(control_upper, to_int=True)],
                                                      [formatter(test1_lower, to_int=True), formatter(test1_upper, to_int=True)],
                                                      [formatter(test2_lower, to_int=True), formatter(test2_upper, to_int=True)]]}
        stats_summary_df = pd.DataFrame(stats_dict).set_index('Index')
        stats_summary_df = stats_summary_df[stats_summary_df["CR Uncertainty [%]"] != "0.0"]
        return stats_summary_df

    def results_func(group_names= ["control_step_", "test1_step_", "test2_step_"]):

        for j in group_names:
            for i in range(1, data_points_input):
                var_dict_results[j + str(i-1) + "_to_" + str(i)] = perc((var_dict[j + str(i)] / var_dict[j + str(i -1)]), string=True)
        df = pd.DataFrame(var_dict_results.items(), columns=["col0", "col1"])
        df = pd.merge(df.col0.str.split("_", 1, expand=True,), df.col1, left_index=True, right_index=True)
        df.columns = ["Group", "Step", "Value"]
        df = df.set_index("Group").sort_values(by="Step", ascending=True)
        for x in range(0, data_points_input):
            df.Step = df.Step.str.replace(str(x), var_dict[str(x)])
        return df.reset_index().pivot(index='Group', columns='Step', values='Value')

    def details_func(var_dict_results_model, group_names=["control", "test1", "test2"]):
        plot_dict = {}

        for i in range(1, data_points_input):

            col1, col2 = st.columns(2)
            for y in group_names:

                frame1.axes.yaxis.set_ticklabels([])
                plot_dict[f"{y}d"] = var_dict_results_model[y + str(i - 1) + "_to_" + str(i)]
                sns.kdeplot(plot_dict[f"{y}d"], fill=True, label=y)
                plt.ylabel("")
                plt.xlabel("Conversion rate [%]")
                plt.yticks([])
                #plt.gca().xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

                plt.gca().xaxis.set_major_formatter(formatter)
                legend_labels = ['Control']
                for legend_id in range(1, number_of_cohort):
                    legend_labels.append(f'Test{legend_id}')
                plt.legend(legend_labels, frameon=True, loc='upper left', ncol=1, shadow=True, borderpad=1, prop={'size': 6})
                plt.title(f"From {var_dict[str(i - 1)]} to {var_dict[str(i)]}")
            if number_of_cohort == 3:
                stats_json = utils.compute_bayesian_statistics(var_dict_results_model["control_step_" + str(i - 1) + "_to_" + str(i)],
                                                               var_dict_results_model["test1_step_" + str(i - 1) + "_to_" + str(i)],
                                                               var_dict_results_model["test2_step_" + str(i - 1) + "_to_" + str(i)])
                third_cohort = var_dict_results_model["test2_step_" + str(i - 1) + "_to_" + str(i)]
            else:
                stats_json = utils.compute_bayesian_statistics(
                    samples_control=var_dict_results_model["control_step_" + str(i - 1) + "_to_" + str(i)],
                    samples_test1=var_dict_results_model["test1_step_" + str(i - 1) + "_to_" + str(i)],
                    samples_test2=np.zeros(len(var_dict_results_model["test1_step_" + str(i - 1) + "_to_" + str(i)])))
                third_cohort = np.zeros(len(var_dict_results_model["test1_step_" + str(i - 1) + "_to_" + str(i)]))

            stats_raw = [var_dict_results_model["control_step_" + str(i - 1) + "_to_" + str(i)],
                         var_dict_results_model["test1_step_" + str(i - 1) + "_to_" + str(i)],
                         third_cohort]
            col2.table(bayesian_result_table(stats_json))
            col2.table(uncertainty_func(stats_raw))
            col1.pyplot(fig)
            plt.clf()
            master_json["Stats" + str(i - 1) + "_to_" + str(i)] = bayesian_result_table(stats_json).to_json()
        st.write("-----" * 20)
        results_json = master_json
        plt.clf()
        control_all_step = model.generate_posterior(n_total=int(var_dict["control_step_0"]),
                                                    n_success=int(var_dict["control_step_"+ str(data_points_input-1)]))
        test1_all_step = model.generate_posterior(n_total=int(var_dict["test1_step_0"]),
                                                  n_success=int(var_dict["test1_step_" + str(data_points_input-1)]))
        test2_all_step = np.zeros(shape=(len(test1_all_step)))
        if number_of_cohort > 2:
            test2_all_step = model.generate_posterior(n_total=int(var_dict["test2_step_0"]), n_success=int(var_dict["test2_step_" + str(data_points_input-1)]))
            sns.kdeplot(test2_all_step, fill=True, label="Test 2")

        sns.kdeplot(control_all_step, fill=True,label="Control")
        sns.kdeplot(test1_all_step, fill=True, label="Test 1")
        plt.xlabel("Conversion rate [%]")
        plt.ylabel("-")
        plt.yticks([])
        plt.gca().xaxis.set_major_formatter(formatter)
        plt.legend(frameon=True, loc='upper left', ncol=1, shadow=True, borderpad=1, prop={'size': 6})
        plt.title(f"From {var_dict[str(0)]} to {var_dict[str(legend_id)]}")
        full_res_col1, full_res_col2 = st.columns(2)
        st.write("-----"* 50)
        full_res_col1.pyplot(fig)

        stats_json = utils.compute_bayesian_statistics(control_all_step, test1_all_step, test2_all_step)
        results_json["From beginning to end"] = bayesian_result_table(stats_json).to_json()
        full_res_col2.table(bayesian_result_table(stats_json))
        full_res_col2.write("-----"* 10)
        stats_raw_full = [control_all_step, test1_all_step, test2_all_step]
        full_res_col2.table(uncertainty_func(stats_raw_full))
        return results_json
    def bayesian_result_table(stats_json):
            stats_dict = {'Index': ['Baseline', 'Test 1', 'Test 2'],
                          'Probability to be the best [%]': [stats_json['P(Control is Best) [%]'],
                                                             stats_json['P(test1 is Best) [%]'],
                                                             stats_json['P(test2 is Best) [%]']],
                          'Probability to beat the Baseline [%]': [0, stats_json['P(test1>control) [%]'],
                                                                   stats_json['P(test2>control) [%]']],
                          'Modelled lift [%]': [0, stats_json['Lift1 median [%]'], stats_json['Lift2 median [%]']],
                      'Expected loss [%]': [0, stats_json['Expected loss 1 median [%]'], stats_json['Expected loss 2 median [%]']]}

            stats_summary_df = pd.DataFrame(stats_dict).set_index('Index')

            stats_summary_df['Probability to be the best [%]'] = stats_summary_df[
                                                                     'Probability to be the best [%]'] / 100
            stats_summary_df['Probability to beat the Baseline [%]'] = stats_summary_df[
                                                                           'Probability to beat the Baseline [%]'] / 100
            stats_summary_df['Modelled lift [%]'] = stats_summary_df['Modelled lift [%]'] / 100
            stats_summary_df['Expected loss [%]'] = stats_summary_df['Expected loss [%]'] / 100
            stats_summary_df['Probability to be the best [%]'] = stats_summary_df['Probability to be the best [%]'].map(
                '{:.2%}'.format)
            stats_summary_df['Probability to beat the Baseline [%]'] = stats_summary_df[
                'Probability to beat the Baseline [%]'].map('{:.2%}'.format)
            stats_summary_df['Modelled lift [%]'] = stats_summary_df['Modelled lift [%]'].map('{:.2%}'.format)
            stats_summary_df['Expected loss [%]'] = stats_summary_df['Expected loss [%]'].map('{:.2%}'.format)
            stats_summary_df.replace("0.000%", "--", inplace=True)
            bad_df = stats_summary_df.index.isin(["Baseline"])
            stats_summary_df.loc[~bad_df].replace("0.00%", "--", inplace=True)
            stats_summary_df.replace("-100.00%", "--", inplace=True)
            stats_summary_df = stats_summary_df[stats_summary_df["Modelled lift [%]"] != "--"]
            return stats_summary_df

    number_of_cohort = st.number_input("Enter no of cohorts", format='%d', value=3)
    data_points_input = st.number_input("Enter no of funnel steps", format='%d', value=4)

    st.title("Enter experiment details")
    var_dict = {}
    var_dict_results = {}
    var_dict_results_model = {}
    row_1, row_2, row_3, *other_rows = st.columns(number_of_cohort+1)
    group_names = ["control_step_", "test1_step_"]
    for i in range(2, number_of_cohort):
        group_names.append(f"test{i}_step_")
    with row_1:
        st.write("Enter names of steps")
        for x in range(data_points_input):
            var_dict[str(x)] = st.text_input(f"enter input for step {x}", key=f"naming input {x}", value=str(x))
    with row_2:
        st.write("For control")
        for x in range(data_points_input):
            var_dict["control_step_" + str(x)] = st.number_input(f"enter input for step {x}", key=f"control input {x}",value=100-10*x)
    with row_3:
        st.write("For test 1")
        for x in range(data_points_input):
            var_dict["test1_step_" + str(x)] = st.number_input(f"enter input for step {x}", key=f"test 1 input {x}",value=100-12*x)
    if number_of_cohort > 2:
        for idx,row in enumerate(other_rows):
            with row:
                st.write(f"For test {idx+2}")
                for x in range(data_points_input):
                    var_dict[f"test{idx+2}_step_" + str(x)] = st.number_input(f"enter input for step {x}", key=f"test{idx+2} input {x}",value=100-13*x)

    results_dict = results_func(group_names)
    st.table(results_dict)

    mcol1, *mcol2 = st.columns(number_of_cohort)
    with mcol1:
        st.metric("Control total conversion rate", perc(var_dict["control_step_" + str(data_points_input-1)]/var_dict["control_step_0"],string=True))
    for idx, x in enumerate(mcol2):
        with x:
            st.metric(f"Test {idx+1} total conversion rate", perc(var_dict[f"test{idx+1}_step_" + str(data_points_input - 1)] / var_dict[f"test{idx+1}_step_0"], string=True))

    with st.expander("Details..."):
        for j in group_names:
            for i in range(1, data_points_input):
                var_dict_results_model[j + str(i - 1) + "_to_" + str(i)] = model.generate_posterior(
                    n_total=var_dict[j + str(i - 1)], n_success=var_dict[j + str(i)])

        master_json = details_func(var_dict_results_model, group_names)
    """
    with st.form("Enter credentials to upload", clear_on_submit=False):
        name_input = st.text_input("Enter name")
        dict_input = {'Name':name_input, "Step Names":[var_dict[str(x)] for x in range(data_points_input)],
                      "Control Input": [var_dict["control_step_" + str(x)] for x in range(data_points_input)],
                      "Test 1 Input": [var_dict["test1_step_" + str(x)] for x in range(data_points_input)],
                      "Conversion Table": results_dict.to_json(),
                      "From beginning to end ": master_json["From beginning to end"]}

        for idx in range(1, data_points_input):
            dict_input[f"From {idx-1} to {idx} "] = master_json[f"Stats{idx-1}_to_{idx}"]
        if number_of_cohort > 2:
            dict_input["Test 2 Input"] = [var_dict["test2_step_" + str(x)] for x in range(data_points_input)],
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.write("Uploaded", today)
            file_name = str(today.date()) + str(name_input)
            write_json('/Users/mertcancoskun/PycharmProjects/Dashboards/data', file_name, f"{file_name}.json")

            os.system(f"aws s3 cp {file_name}.json s3://adhoc.product-analytics.acc/test_oscar/ --region eu-central-1")
    """