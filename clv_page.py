import streamlit as st
import pandas as pd
from src.clv_calculation import CLV_calculation
import numpy as np
import itertools
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
from src import utils

def clv_page():
    st.markdown("CLV calculation")
    # HELPER FUNCTIONS###

    # Code snippet to hide index
    hide_table_row_index = """
                <style>
                thead tr th:first-child {display:none}
                tbody th {display:none}
                </style>
                """
    # Inject CSS with Markdown
    st.markdown(hide_table_row_index, unsafe_allow_html=True)

    def lift_model_plot(lift1, lift2):
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.kdeplot(lift1, fill=True, color='orange', label='test1')
        sns.kdeplot(lift2, fill=True, color='green', label='test2')
        plt.xlabel('Modelled Lift [%]', size=15)
        plt.legend()
        ax.set_yticklabels([])
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(12))
        st.pyplot(fig)
        return None

    def conversion_rate_plot():
        pass

    ### NEW PLOT ###
    def total_clr_plot(clr_samples_control_sum, clr_samples_test1_sum, clr_samples_test2_sum):
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.kdeplot(clr_samples_control_sum / 10 ** 3, fill=True, label='Control')
        sns.kdeplot(clr_samples_test1_sum / 10 ** 3, fill=True, label='Test1')
        sns.kdeplot(clr_samples_test2_sum / 10 ** 3, fill=True, label='Test2')
        plt.xlabel("Sum of CLR [K] Euros", size=15)
        plt.legend()
        ax.set_yticklabels([])
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(2))
        st.pyplot(fig)
        return None

    def conv_plot(clr_samples_control_sum, clr_samples_test1_sum, clr_samples_test2_sum):
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.kdeplot(clr_samples_control_sum, fill=True, label='Control')
        sns.kdeplot(clr_samples_test1_sum, fill=True, label='Test1')
        sns.kdeplot(clr_samples_test2_sum, fill=True, label='Test2')
        plt.xlabel("Conversion Rates", size=15)
        plt.legend()
        ax.set_yticklabels([])
        plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        st.pyplot(fig)
        return None

    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:

        dict_df = pd.read_excel(uploaded_file, sheet_name=['Control', 'Test 1', 'Test 2'])

        control_df = dict_df.get('Control').fillna("")
        test1_df = dict_df.get('Test 1').fillna("")
        test2_df = dict_df.get('Test 2').fillna("")
        groups = [control_df, test1_df, test2_df]

        product_names = [s.replace('\u200b', '') for s in control_df['Product'].unique()]
        product_names = list(filter(None, product_names))
        project_runtimes = [int(s.replace('\u200b', '')) for s in control_df['Runtime'].unique()]

        for group in groups:
            group['Group'] = group['Group'].apply(lambda a: a if len(str(a)) > 2 else np.NaN).ffill(axis=0)
            group['Product'] = group['Product'].apply(lambda a: a if len(str(a)) > 2 else np.NaN).ffill(axis=0)
            group['Discount'] = group['Discount'].apply(lambda s: int(s.replace('\u200b', '')))
            group['Amount'] = group['Amount'].apply(lambda s: int(s.replace('\u200b', '')))
            group['Runtime'] = group['Runtime'].apply(lambda s: int(s.replace('\u200b', '')))

        st.markdown(f'Product names are {" & ".join(product_names)}')
        st.markdown(f'Project runtimes are {project_runtimes}')

        control_view = st.number_input('Control views K', format='%g', value=50) * 1000
        test1_view = st.number_input('Test 1 views K', format='%g', value=50) * 1000
        test2_view = st.number_input('Test 2 views K', format='%g', value=50) * 1000

        # Input data for the CLV calculation
        product_runtimes = ['runtime_1', 'runtime_12']

        product_list = list(itertools.product(product_names, product_runtimes))

        orders = {'control': np.array(control_df['Amount']), 'test1': np.array(test1_df['Amount']),
                  'test2': np.array(test2_df['Amount'])}

        discounts = {'control': np.array(control_df['Discount']), 'test1': np.array(test1_df['Discount']),
                     'test2': np.array(test2_df['Discount'])}

        views = {
            'control': control_view,
            'test1': test1_view,
            'test2': test2_view
        }

        cycles = {
            'runtime_1': max(control_df['Runtime']) * 6,
            'runtime_12': min(control_df['Runtime']) * 6
        }

        calc = CLV_calculation(
            product_names=product_names,
            runtimes=product_runtimes,
            product_list=product_list,
            discounts=discounts,
            orders=orders,
            views=views,
            cycles=cycles)

        clr_samples_control_sum, clr_samples_test1_sum, clr_samples_test2_sum, df_summary = calc.compute_clr_group()

        lift1 = ((clr_samples_test1_sum - clr_samples_control_sum) / clr_samples_control_sum) * 100
        lift2 = ((clr_samples_test2_sum - clr_samples_control_sum) / clr_samples_control_sum) * 100

        ### CONVERSION RATE ###

        result_table = {'Index': ['Baseline', 'Test 1', 'Test 2'],
                        'Upsell Pages Shown': [int(control_view), int(test1_view), int(test2_view)],
                        'Paid Purchases': [np.sum(np.array(control_df['Amount'])), np.sum(np.array(test1_df['Amount'])),
                                           np.sum(np.array(test2_df['Amount']))]}
        result_df = pd.DataFrame(result_table).set_index('Index').round(1)
        result_df["Conversion Rate"] = result_df['Paid Purchases'] / result_df['Upsell Pages Shown']
        result_df["Conversion Rate"] = result_df["Conversion Rate"].map('{:.2%}'.format)
        st.title("F2P Conversion statistics ")
        st.table(result_df)

        p0_samples_conv = calc.p0_samples
        p1_samples_conv = calc.p1_samples
        p2_samples_conv = calc.p2_samples

        row1_conv, row2_conv = st.columns(2)
        with row1_conv:
            lift1_conv = ((p1_samples_conv - p0_samples_conv) / p0_samples_conv) * 100
            lift2_conv = ((p2_samples_conv - p0_samples_conv) / p0_samples_conv) * 100
            lift_model_plot(lift1_conv, lift2_conv)
        with row2_conv:
            conv_plot(p0_samples_conv, p1_samples_conv, p2_samples_conv)

        stats_json_conv = utils.compute_bayesian_statistics(p0_samples_conv, p1_samples_conv, p2_samples_conv)
        stats_dict_conv = {'Index': ['Baseline', 'Test 1', 'Test 2'],
                           'Probability to be the best [%]': [stats_json_conv['P(Control is Best) [%]'],
                                                              stats_json_conv['P(test1 is Best) [%]'],
                                                              stats_json_conv['P(test2 is Best) [%]']],
                           'Probability to beat the Baseline [%]': [0, stats_json_conv['P(test1>control) [%]'],
                                                                    stats_json_conv['P(test2>control) [%]']],
                           'Modelled lift [%]': [0, stats_json_conv['Lift1 median [%]'],
                                                 stats_json_conv['Lift2 median [%]']],
                           'Expected loss [%]': [0, stats_json_conv['Expected loss 1 median [%]'],
                                                 stats_json_conv['Expected loss 2 median [%]']]
                           }
        stats_summary_df_conv = pd.DataFrame(stats_dict_conv).set_index('Index')

        stats_summary_df_conv['Probability to be the best [%]'] = stats_summary_df_conv[
                                                                      'Probability to be the best [%]'] / 100
        stats_summary_df_conv['Probability to beat the Baseline [%]'] = stats_summary_df_conv[
                                                                            'Probability to beat the Baseline [%]'] / 100
        stats_summary_df_conv['Expected loss [%]'] = stats_summary_df_conv[
                                                         'Expected loss [%]'] / 100
        stats_summary_df_conv['Modelled lift [%]'] = stats_summary_df_conv['Modelled lift [%]'] / 100

        stats_summary_df_conv['Probability to be the best [%]'] = stats_summary_df_conv[
            'Probability to be the best [%]'].map(
            '{:.2%}'.format)
        stats_summary_df_conv['Probability to beat the Baseline [%]'] = stats_summary_df_conv[
            'Probability to beat the Baseline [%]'].map('{:.2%}'.format)
        stats_summary_df_conv['Modelled lift [%]'] = stats_summary_df_conv['Modelled lift [%]'].map('{:.2%}'.format)
        stats_summary_df_conv['Expected loss [%]'] = stats_summary_df_conv['Expected loss [%]'].map('{:.2%}'.format)
        stats_summary_df_conv.replace("0.00%", "--", inplace=True)
        st.table(stats_summary_df_conv)

        details_expander_f2p = st.expander('Click for conversion rate breakdown by product')
        with details_expander_f2p:
            st.write("Conversion rate breakdown...")

        ### CLR ANALYSIS ###
        st.title("CLR Analysis")

        stats_json = utils.compute_bayesian_statistics(clr_samples_control_sum, clr_samples_test1_sum,
                                                       clr_samples_test2_sum)

        stats_dict = {'Index': ['Baseline', 'Test 1', 'Test 2'],
                      'Probability to be the best [%]': [stats_json['P(Control is Best) [%]'],
                                                         stats_json['P(test1 is Best) [%]'],
                                                         stats_json['P(test2 is Best) [%]']],
                      'Probability to beat the Baseline [%]': [0, stats_json['P(test1>control) [%]'],
                                                               stats_json['P(test2>control) [%]']],
                      'Modelled lift [%]': [0, stats_json['Lift1 median [%]'], stats_json['Lift2 median [%]']],
                      'Expected loss [%]': [0, stats_json['Expected loss 1 median [%]'],
                                            stats_json['Expected loss 2 median [%]']]}

        stats_summary_df = pd.DataFrame(stats_dict).set_index('Index')

        stats_summary_df['Probability to be the best [%]'] = stats_summary_df[
                                                                 'Probability to be the best [%]'] / 100
        stats_summary_df['Probability to beat the Baseline [%]'] = stats_summary_df[
                                                                       'Probability to beat the Baseline [%]'] / 100
        stats_summary_df['Expected loss [%]'] = stats_summary_df[
                                                    'Expected loss [%]'] / 100
        stats_summary_df['Modelled lift [%]'] = stats_summary_df['Modelled lift [%]'] / 100

        stats_summary_df['Probability to be the best [%]'] = stats_summary_df['Probability to be the best [%]'].map(
            '{:.2%}'.format)
        stats_summary_df['Probability to beat the Baseline [%]'] = stats_summary_df[
            'Probability to beat the Baseline [%]'].map('{:.2%}'.format)
        stats_summary_df['Modelled lift [%]'] = stats_summary_df['Modelled lift [%]'].map('{:.2%}'.format)
        stats_summary_df['Expected loss [%]'] = stats_summary_df['Expected loss [%]'].map('{:.2%}'.format)
        stats_summary_df.replace("0.00%", "--", inplace=True)

        df_clv = df_summary.drop('clr', axis=1).sort_values('runtime')
        df_clr = df_summary.drop('clv', axis=1).sort_values('runtime')

        df_clv = pd.pivot_table(df_clv, values='clv', index=['product_name', 'runtime'], columns='group').reset_index()
        df_clr = pd.pivot_table(df_clr, values='clr', index=['product_name', 'runtime'], columns='group').reset_index()
        df_clv.runtime = df_clv.runtime.str.replace("runtime_", "")
        df_clr.runtime = df_clr.runtime.str.replace("runtime_", "")

        df_clr.reset_index(drop=True, inplace=True)
        row2_1, row2_2 = st.columns(2)
        with row2_1:
            for i in range(len(project_runtimes)):
                st.subheader(f"CLV summary for runtime {project_runtimes[i]}")
                st.table(df_clv[df_clv.runtime == str(project_runtimes[i])].style.format(
                    {'control': '{:.1f}', 'test1': '{:.1f}', 'test2': '{:.1f}'}))
        clv_dataframe = {}
        with row2_2:
            for i in range(len(project_runtimes)):
                st.subheader(f"CLR summary for runtime {project_runtimes[i]}")
                df_temp = df_clr[df_clr.runtime == str(project_runtimes[i])]

                st.table(df_temp)
                # st.write(df_temp.style.format({'control': '{:.1f}', 'test1': '{:.1f}', 'test2': '{:.1f}'}).to_html(),unsafe_allow_html=True)
                st.write(f"Total for for runtime {str(project_runtimes[i])}")
                st.table(df_temp.groupby('runtime').agg({'control': 'sum', 'test1': 'sum', 'test2': 'sum'}).round(1))
                clv_dataframe[i] = df_temp.groupby('runtime').agg(
                    {'control': ['sum'], 'test1': ['sum'], 'test2': ['sum']}).round(1)

            control_grand_total, test1_grand_total, test2_grand_total = 0, 0, 0

            for i in range(len(project_runtimes)):
                control_grand_total += clv_dataframe[i].iloc[0, 0]
                test1_grand_total += clv_dataframe[i].iloc[0, 1]
                test2_grand_total += clv_dataframe[i].iloc[0, 2]
            dict_grand_total = {"control": round(control_grand_total, 1), 'test1': round(test1_grand_total, 1),
                                'test2': round(test2_grand_total, 1)}
            st.subheader(f"Grand CLR Totals")
            st.table(pd.DataFrame(dict_grand_total.items(), columns=['Group', 'Value']).T)

        st.table(stats_summary_df)
        row1, row2 = st.columns(2)
        with row1:
            lift_model_plot(lift1, lift2)
        with row2:
            total_clr_plot(clr_samples_control_sum, clr_samples_test1_sum, clr_samples_test2_sum)

        details_expander = st.expander('Click for breakdown of CLR by product')
        with details_expander:
            st.write("Breakdown by product...")
