import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from src.bernoulli_model import BernoulliModel
import src.utils as utils


class CLV_calculation():

    def __init__(self,
                 product_names=None,
                 runtimes=None,
                 product_list=None,
                 views=None,
                 cycles=None,
                 orders=None,
                 discounts=None):

        # Yearly parameters taken from Norton calculation
        self.ar2ar_rr_yearly = 0.725
        self.rol2ar_rr_yearly = 0.654
        self.rol2rol_rr_yearly = 0.115
        self.ar2rol_rr_yearly = 0.029

        # The monthly parameters taken from Norton
        self.ar2ar_rr_monthly = 0.966
        self.rol2ar_rr_monthly = 0.758
        self.rol2rol_rr_monthly = 0.0
        self.ar2rol_rr_monthly = 0.0

        # TODO: We need to enter two-year retention rates here
        self.ar2ar_rr_two_year = 0.544591
        self.rol2ar_rr_two_year = 0.54936
        self.rol2rol_rr_two_year = 0.032191
        self.ar2rol_rr_two_year = 0.02436

        self.dc_yearly = 0.08  # The yearly discount rate (set by finance)
        self.dc_monthly = (1+self.dc_yearly)**(1.0/12.0)-1.0 # 0.0064 # The monthly discount rate
        self.dc_two_years = (1+self.dc_yearly)**2-1.0 #

        # Data from Jay that is used for the extrapolation
        self.discount_data ={
        'discount': np.array([0.611, 0.556, 0.389, 0]),
        'price': np.array([41.61,47.56,65.4,107.03]),
        'rr_auto': np.array([0.45,0.511,0.568,0.846]),
        'rr_manual': np.array([0.22,0.264,0.319,0.036])
        }

        # Here we conduct a simple linear extrapolation
        x_discount = self.discount_data['discount']

        # We divide by the initial discount price to have a relative value
        y_manual = self.discount_data['rr_manual']/self.discount_data['rr_manual'][-1]
        y_auto = self.discount_data['rr_auto']/self.discount_data['rr_auto'][-1]

        # Here we generate
        self.f_auto = interp1d(x_discount,y_auto,kind='linear')
        self.f_manual = interp1d(x_discount,y_manual,kind='linear')

        # Initialize the data
        print('='*100)
        self.initialize_rr_data()
        self.initialize_sale_data(product_names=product_names,
                                  runtimes=runtimes,
                                  product_list=product_list,
                                  views=views,
                                  cycles=cycles,
                                  orders=orders,
                                  discounts=discounts)


    def initialize_rr_data(self):
        """
        This function instantiates all of the data that is used for the retention rates. The dashboard containing this information is here:


        link: https://tableau.lifelock.com/#/site/Avira/views/RRandAOVforCLRCalculation/Expirations_1?:iid=1


        """

        #------------------------------------------
        # Internet security data

        # Auto renewals
        data_auto_1 = {
            'dates': ['2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06'],
            'rr_auto': np.array([0.914, 0.901, 0.897, 0.897, 0.925, 0.928]),
            'aov_auto': np.array([4.97, 5.00, 5.01, 5.04, 5.14, 5.14]),
        }

        data_auto_12 = {
            'dates': ['2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06'],
            'rr_auto': np.array([0.78, 0.769, 0.727, 0.722, 0.727, 0.896]),
            'aov_auto': np.array([39, 35, 38.57, 39.86, 40.93, 41.79, 41.95]),
        }

        data_auto_24 = {
            'dates': ['2022-01', '2022-02', '2022-03', '2022-04', '2022-05'],
            'rr_auto': np.array([71.7, 66.8, 50.6, 47.8, 61.8])/100.0,
            'aov_auto': np.array([72.14, 69.75, 77.77, 81.65, 79.38]),
        }


        # Manual renewals
        data_manual_1 = {
            'dates': ['2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06'],
            'rr_manual': np.array([1.87 / 100.0, 0.77 / 100.0, 0.84 / 100.0, 0.71 / 100.0, 1.03 / 100.0, 0.63 / 100.0]),
            'aov_manual': np.array([5.26, 4.28, 4.81, 2.67, 5.16, 4.66]),
        }

        data_manual_12 = {
            'dates': ['2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06'],
            'rr_manual': np.array([8.52 / 100.0, 6.34 / 100.0, 3.37 / 100.0, 6.59 / 100.0, 6.16 / 100.0, 5.39 / 100.0]),
            'aov_manual': np.array([34.25, 29.59, 28.65, 31.81, 33.50, 30.01]),
        }

        data_manual_24 = {
            'dates': ['2022-01', '2022-02', '2022-03', '2022-04', '2022-05'],
            'rr_manual': np.array([47.96, 32.98, 29.75, 55.0, 70.21])/100.0,
            'aov_manual': np.array([66.39, 64.74, 55.43, 50.64, 52.14]),
        }

        data_internet_security = {
            'runtime_auto_1': data_auto_1,
            'runtime_auto_12': data_auto_12,
            'runtime_auto_24': data_auto_24,
            'runtime_manual_1': data_manual_1,
            'runtime_manual_12': data_manual_12,
            'runtime_manual_24': data_manual_24
        }

        # ===============================================================================================
        # ===============================================================================================
        # Prime services (max 5 devices)

        # Product id: 1955
        # Auto renewals
        data_auto_1 = {
            'dates': ['2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06'],
            'rr_auto': np.array([0.919, 0.911, 0.905, 0.922, 0.924, 0.929]),
            'aov_auto': np.array([8.17, 8.12, 8.16, 8.15, 8.19, 8.17]),
        }

        data_auto_12 = {
            'dates': ['2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06'],
            'rr_auto': np.array([0.762, 0.750, 0.728, 0.706, 0.674, 0.757]),
            'aov_auto': np.array([77.98, 78.56, 81.90, 83.55, 85.05, 85.58]),
        }

        data_auto_24 = {
            'dates': ['2022-01', '2022-02', '2022-03', '2022-04', '2022-05'],
            'rr_auto': np.array([67.0, 66.2, 51.3, 53.8, 51.2])/100.0,
            'aov_auto': np.array([131.28, 127.45, 148.02, 161.82, 167.73]),
        }

        # Manual renewals
        data_manual_1 = {
            'dates': ['2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06'],
            'rr_manual': np.array([1.12 / 100.0, 0.69 / 100.0, 0.48 / 100.0, 0.75 / 100.0, 0.74 / 100.0, 0.73 / 100.0]),
            'aov_manual': np.array([7.60, 7.19, 6.53, 5.36, 7.22, 7.22]),
        }

        data_manual_12 = {
            'dates': ['2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06'],
            'rr_manual': np.array(
                [12.04 / 100.0, 6.24 / 100.0, 3.29 / 100.0, 10.77 / 100.0, 6.17 / 100.0, 9.62 / 100.0]),
            'aov_manual': np.array([65.24, 58.85, 52.97, 54.46, 63.14, 50.30]),
        }

        # TODO: Input 2-year retention rates here
        # TODO: Ask Robert about these retention rates as-well, they appear to vary alot
        data_manual_24 = {
            'dates': ['2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06'],
            'rr_manual': np.array([8.05, 8.98, 13.95, 20.87, 29.31, 15.68, 43.71])/100.0,
            'aov_manual': np.array([125.4, 121.1, 120.9, 114.5, 110.9, 93.1, 92.2]),
        }

        data_prime_services_max5 = {
            'runtime_auto_1': data_auto_1,
            'runtime_auto_12': data_auto_12,
            'runtime_auto_24': data_auto_24,
            'runtime_manual_1': data_manual_1,
            'runtime_manual_12': data_manual_12,
            'runtime_manual_24': data_manual_24
        }



        # ===============================================================================================
        # ===============================================================================================
        # ===============================================================================================
        # Prime Services Max (25)

        # Product id: 1956
        # Auto renewals
        data_auto_1 = {
            'dates': ['2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06'],
            'rr_auto': np.array([0.919, 0.911, 0.905, 0.922, 0.924, 0.929]),
            'aov_auto': np.array([8.17, 8.12, 8.16, 8.15, 8.19, 8.17]),
        }

        data_auto_12 = {
            'dates': ['2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06'],
            'rr_auto': np.array([0.762, 0.750, 0.728, 0.706, 0.674, 0.757]),
            'aov_auto': np.array([77.98, 78.56, 81.90, 83.55, 85.05, 85.58]),
        }

        data_auto_24 = {
            'dates': ['2022-01', '2022-02', '2022-03'],
            'rr_auto': np.array([84.6, 78.9, 60.0])/100.0,
            'aov_auto': np.array([224.85, 211.37, 203.42]),
        }

        # Manual renewals
        data_manual_1 = {
            'dates': ['2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06'],
            'rr_manual': np.array([2.96 / 100, 2.21 / 100.0, 2.55 / 100.0, 1.67 / 100.0, 2.26 / 100.0, 2.36 / 100.0]),
            'aov_manual': np.array([9.51, 7.87, 7.53, 9.26, 10.04, 7.75]),
        }

        data_manual_12 = {
            'dates': ['2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06'],
            'rr_manual': np.array([0.2508, 0.2966, 0.1915, 0.1433, 0.2795, 0.1756]),
            'aov_manual': np.array([86.27, 77.66, 74.21, 86.26, 72.64, 81.69]),
        }

        # TODO: Input 2-year retention rates here
        # TODO: Ask robert about the retention rates for this product... the rates do not make sense here
        data_manual_24 = {
            'dates': ['2022-01'],
            'rr_manual': np.array([0.0]),
            'aov_manual': np.array([0.0]),
        }

        data_prime_services_max25 = {
            'runtime_auto_1': data_auto_1,
            'runtime_auto_12': data_auto_12,
            'runtime_auto_24': data_auto_24,
            'runtime_manual_1': data_manual_1,
            'runtime_manual_12': data_manual_12,
            'runtime_manual_24': data_manual_24
        }

        # ===============================================================================================
        # ===============================================================================================
        # ===============================================================================================
        # Avira Phantom VPN Pro

        # Auto renewals (product ID = 1704)
        data_auto_1 = {
            'dates': ['2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06'],
            'rr_auto': np.array([83.2/100.0, 81.2/100, 77.1/100.0, 81.1/100.0, 81.9/100.0, 81.6/100.0]),
            'aov_auto': np.array([6.83, 6.77, 6.88, 6.94, 7.09, 7.05]),
        }

        data_auto_12 = {
            'dates': ['2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06'],
            'rr_auto': np.array([69.4/100.0, 68.7/100.0, 69.2/100.0, 66.4/100.0, 62.6/100.0, 69.6/100.0]),
            'aov_auto': np.array([52.08, 50.38, 52.42, 53.02, 53.96, 53.15]),
        }

        data_auto_24 = {
            'dates': ['2022-01', '2022-02', '2022-03', '2022-04', '2022-05'],
            'rr_auto': np.array([58.3, 70.0, 54.5, 50.0, 64.7])/100.0,
            'aov_auto': np.array([76.17, 72.29, 86.30, 93.66, 88.30]),
        }

        # Manual renewals
        data_manual_1 = {
            'dates': ['2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06'],
            'rr_manual': np.array([0.84/100, 0.61/100, 2.17/100,0.98/100,0.29/100,0.24/100]),
            'aov_manual': np.array([5.51, 4.09, 3.53, 7.03, 7.39, 5.42]),
        }

        data_manual_12 = {
            'dates': ['2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06'],
            'rr_manual': np.array([5.79/100, 3.72/100, 1.17/100, 7.71/100, 4.06/100, 2.39/100]),
            'aov_manual': np.array([50.68, 45.35, 37.22, 49.72, 50.86, 43.29]),
        }

        data_manual_24 = {
            'dates': ['2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06'],
            'rr_manual': np.array([3.23, 11.48, 30.0])/100.0,
            'aov_manual': np.array([88.6, 95.3, 73.7]),
        }

        data_phantom_vpn = {
            'runtime_auto_1': data_auto_1,
            'runtime_auto_12': data_auto_12,
            'runtime_auto_24': data_auto_24,
            'runtime_manual_1': data_manual_1,
            'runtime_manual_12': data_manual_12,
            'runtime_manual_24': data_manual_24
        }


        # ===============================================================================================
        # ===============================================================================================
        # ===============================================================================================
        # Avira Antivirus Pro


        # Auto renewals (product ID = 2215)

        data_auto_1 = {
            'dates': ['2022-01', '2022-02', '2022-03', '2022-04'],
            'rr_auto': np.array([93.1/100.0, 88.1/100.0, 85.1/100.0, 86.2/100.0]),
            'aov_auto': np.array([3.41, 3.50, 3.49, 3.48]),
        }

        data_auto_12 = {
            'dates': ['2022-01', '2022-02', '2022-03', '2022-04'],
            'rr_auto': np.array([77.9/100.0, 76.3/100.0, 72.7/100.0, 72.6/100.0]),
            'aov_auto': np.array([30.81, 30.25, 31.18, 30.57]),
        }

        data_auto_24 = {
            'dates': ['2022-01', '2022-02', '2022-03', '2022-04','2022-05'],
            'rr_auto': np.array([70.4, 67.5, 51.9, 57.4, 61.3]),
            'aov_auto': np.array([52.26, 50.32, 57.57, 61.06, 62.24]),
        }

        # Manual renewals
        data_manual_1 = {
            'dates': ['2022-01', '2022-02', '2022-03', '2022-04'],
            'rr_manual': np.array([3.52/100.0, 1.23/100.0, 1.83/100.0, 1.58/100.0]),
            'aov_manual': np.array([3.64, 3.56, 3.18, 3.43]),
        }

        data_manual_12 = {
            'dates': ['2022-01', '2022-02', '2022-03', '2022-04'],
            'rr_manual': np.array([13.90/100.0, 35.02/100.0, 12.63/100.0, 8.88/100.0]),
            'aov_manual': np.array([22.84, 9.92, 10.54, 18.89]),
        }

        data_manual_24 = {
            'dates': ['2021-11', '2021-12', '2022-01', '2022-02', '2022-03'],
            'rr_manual': np.array([6.94, 3.99, 5.90, 4.96, 8.42])/100.0,
            'aov_manual': np.array([50.14, 45.26, 51.35, 50.35, 46.27]),
        }

        data_av_pro = {
            'runtime_auto_1': data_auto_1,
            'runtime_auto_12': data_auto_12,
            'runtime_auto_24': data_auto_24,
            'runtime_manual_1': data_manual_1,
            'runtime_manual_12': data_manual_12,
            'runtime_manual_24': data_manual_24
        }

        # ===============================================================================================
        # ===============================================================================================
        # ===============================================================================================
        # Avira system speedup

        data_auto_1 = {
            'dates': ['2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06', '2022-07', '2022-08'],
            'rr_auto': np.array([94.3, 93.4, 89.9, 92.5, 93.4, 94.7, 95.2, 95.2])/100.0,
            'aov_auto': np.array([2.55, 2.54, 2.56, 2.57, 2.58, 2.57, 2.58, 2.59]),
        }

        data_auto_12 = {
            'dates': ['2022-01', '2022-02', '2022-03', '2022-04', '2022-05'],
            'rr_auto': np.array([75.8, 74.9, 73.0, 72.4, 70.8])/100.0,
            'aov_auto': np.array([21.46, 21.15, 21.57, 21.87, 22.21]),
        }

        data_auto_24 = {
            'dates': ['2022-01', '2022-02', '2022-03', '2022-04'],
            'rr_auto': np.array([66.0, 64.6, 48.9, 61.8, 52.0])/100.0,
            'aov_auto': np.array([31.55, 31.20, 33.70, 35.15, 35.79]),
        }

        #  Manual renewals
        data_manual_1 = {
            'dates': ['2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06','2022-07', '2022-08'],
            'rr_manual': np.array([0.18, 0.27, 0.37, 0.28, 0.48, 0.27, 0.21, 0.22])/100.0,
            'aov_manual': np.array([2.95, 2.64, 2.51, 2.19, 2.04, 2.82, 2.52, 2.39]),
        }

        data_manual_12 = {
            'dates': ['2022-01', '2022-02', '2022-03', '2022-04', '2022-05'],
            'rr_manual': np.array([2.42, 1.05, 5.01, 1.67, 2.01])/100.0,
            'aov_manual': np.array([18.02, 18.11, 3.83, 17.90, 22.53]),
        }

        data_manual_24 = {
            'dates': ['2022-01', '2022-02', '2022-03', '2022-04'],
            'rr_manual': np.array([13.08, 7.26, 4.35, 10.63])/100.0,
            'aov_manual': np.array([31.14, 33.61, 34.22, 31.73]),
        }

        data_system_speedup = {
            'runtime_auto_1': data_auto_1,
            'runtime_auto_12': data_auto_12,
            'runtime_auto_24': data_auto_24,
            'runtime_manual_1': data_manual_1,
            'runtime_manual_12': data_manual_12,
            'runtime_manual_24': data_manual_24
        }



        #===============================================================
        #
        # Compute the retention data
        #
        # ===============================================================


        self.data_retention = {
            'prime-services-max-5': {
                'runtime_1': {
                    "aov_auto": data_prime_services_max5['runtime_auto_1']['aov_auto'].mean(),
                    "rr_auto": data_prime_services_max5['runtime_auto_1']['rr_auto'].mean(),
                    "aov_manual": data_prime_services_max5['runtime_manual_1']['aov_manual'].mean(),
                    "rr_manual": data_prime_services_max5['runtime_manual_1']['rr_manual'].mean()
                },
                'runtime_12': {
                    "aov_auto": data_prime_services_max5['runtime_auto_12']['aov_auto'].mean(),
                    "rr_auto": data_prime_services_max5['runtime_auto_12']['rr_auto'].mean(),
                    "aov_manual": data_prime_services_max5['runtime_manual_12']['aov_manual'].mean(),
                    "rr_manual": data_prime_services_max5['runtime_manual_12']['rr_manual'].mean()
                },
                'runtime_24': {
                    "aov_auto": data_prime_services_max5['runtime_auto_24']['aov_auto'].mean(),
                    "rr_auto": data_prime_services_max5['runtime_auto_24']['rr_auto'].mean(),
                    "aov_manual": data_prime_services_max5['runtime_manual_24']['aov_manual'].mean(),
                    "rr_manual": data_prime_services_max5['runtime_manual_24']['rr_manual'].mean()
                }
            },
            'prime-services-max-25': {
                'runtime_1': {
                    "aov_auto": data_prime_services_max25['runtime_auto_1']['aov_auto'].mean(),
                    "rr_auto": data_prime_services_max25['runtime_auto_1']['rr_auto'].mean(),
                    "aov_manual": data_prime_services_max25['runtime_manual_1']['aov_manual'].mean(),
                    "rr_manual": data_prime_services_max25['runtime_manual_1']['rr_manual'].mean()
                },
                'runtime_12': {
                    "aov_auto": data_prime_services_max25['runtime_auto_12']['aov_auto'].mean(),
                    "rr_auto": data_prime_services_max25['runtime_auto_12']['rr_auto'].mean(),
                    "aov_manual": data_prime_services_max25['runtime_manual_12']['aov_manual'].mean(),
                    "rr_manual": data_prime_services_max25['runtime_manual_12']['rr_manual'].mean()
                },
                'runtime_24': {
                    "aov_auto": data_prime_services_max25['runtime_auto_24']['aov_auto'].mean(),
                    "rr_auto": data_prime_services_max25['runtime_auto_24']['rr_auto'].mean(),
                    "aov_manual": data_prime_services_max25['runtime_manual_24']['aov_manual'].mean(),
                    "rr_manual": data_prime_services_max25['runtime_manual_24']['rr_manual'].mean()
                }
            },
            'internet-security': {
                "runtime_1": {
                    "aov_auto": data_internet_security['runtime_auto_1']['aov_auto'].mean(),
                    "rr_auto": data_internet_security['runtime_auto_1']['rr_auto'].mean(),
                    "aov_manual": data_internet_security['runtime_manual_1']['aov_manual'].mean(),
                    "rr_manual": data_internet_security['runtime_manual_1']['rr_manual'].mean()
                },
                "runtime_12": {
                    "aov_auto": data_internet_security['runtime_auto_12']['aov_auto'].mean(),
                    "rr_auto": data_internet_security['runtime_auto_12']['rr_auto'].mean(),
                    "aov_manual": data_internet_security['runtime_manual_12']['aov_manual'].mean(),
                    "rr_manual": data_internet_security['runtime_manual_12']['rr_manual'].mean()
                },
                "runtime_24": {
                    "aov_auto": data_internet_security['runtime_auto_24']['aov_auto'].mean(),
                    "rr_auto": data_internet_security['runtime_auto_24']['rr_auto'].mean(),
                    "aov_manual": data_internet_security['runtime_manual_24']['aov_manual'].mean(),
                    "rr_manual": data_internet_security['runtime_manual_24']['rr_manual'].mean()
                }
            },
            'phantom-vpn': {
                "runtime_1": {
                    "aov_auto": data_phantom_vpn['runtime_auto_1']['aov_auto'].mean(),
                    "rr_auto": data_phantom_vpn['runtime_auto_1']['rr_auto'].mean(),
                    "aov_manual": data_phantom_vpn['runtime_manual_1']['aov_manual'].mean(),
                    "rr_manual": data_phantom_vpn['runtime_manual_1']['rr_manual'].mean()
                },
                "runtime_12": {
                    "aov_auto": data_phantom_vpn['runtime_auto_12']['aov_auto'].mean(),
                    "rr_auto": data_phantom_vpn['runtime_auto_12']['rr_auto'].mean(),
                    "aov_manual": data_phantom_vpn['runtime_manual_12']['aov_manual'].mean(),
                    "rr_manual": data_phantom_vpn['runtime_manual_12']['rr_manual'].mean()
                },
                "runtime_24": {
                    "aov_auto": data_phantom_vpn['runtime_auto_24']['aov_auto'].mean(),
                    "rr_auto": data_phantom_vpn['runtime_auto_24']['rr_auto'].mean(),
                    "aov_manual": data_phantom_vpn['runtime_manual_24']['aov_manual'].mean(),
                    "rr_manual": data_phantom_vpn['runtime_manual_24']['rr_manual'].mean()
                }
            },
            'av-pro': {
                "runtime_1": {
                    "aov_auto": data_av_pro['runtime_auto_1']['aov_auto'].mean(),
                    "rr_auto": data_av_pro['runtime_auto_1']['rr_auto'].mean(),
                    "aov_manual": data_av_pro['runtime_manual_1']['aov_manual'].mean(),
                    "rr_manual": data_av_pro['runtime_manual_1']['rr_manual'].mean()
                },
                "runtime_12": {
                    "aov_auto": data_av_pro['runtime_auto_12']['aov_auto'].mean(),
                    "rr_auto": data_av_pro['runtime_auto_12']['rr_auto'].mean(),
                    "aov_manual": data_av_pro['runtime_manual_12']['aov_manual'].mean(),
                    "rr_manual": data_av_pro['runtime_manual_12']['rr_manual'].mean()
                },
                "runtime_24": {
                    "aov_auto": data_av_pro['runtime_auto_24']['aov_auto'].mean(),
                    "rr_auto": data_av_pro['runtime_auto_24']['rr_auto'].mean(),
                    "aov_manual": data_av_pro['runtime_manual_24']['aov_manual'].mean(),
                    "rr_manual": data_av_pro['runtime_manual_24']['rr_manual'].mean()
                }
            },
            'system-speedup': {
                "runtime_1": {
                    "aov_auto": data_system_speedup['runtime_auto_1']['aov_auto'].mean(),
                    "rr_auto": data_system_speedup['runtime_auto_1']['rr_auto'].mean(),
                    "aov_manual": data_system_speedup['runtime_manual_1']['aov_manual'].mean(),
                    "rr_manual": data_system_speedup['runtime_manual_1']['rr_manual'].mean()
                },
                "runtime_12": {
                    "aov_auto": data_system_speedup['runtime_auto_12']['aov_auto'].mean(),
                    "rr_auto": data_system_speedup['runtime_auto_12']['rr_auto'].mean(),
                    "aov_manual": data_system_speedup['runtime_manual_12']['aov_manual'].mean(),
                    "rr_manual": data_system_speedup['runtime_manual_12']['rr_manual'].mean()
                },
                "runtime_24": {
                    "aov_auto": data_system_speedup['runtime_auto_24']['aov_auto'].mean(),
                    "rr_auto": data_system_speedup['runtime_auto_24']['rr_auto'].mean(),
                    "aov_manual": data_system_speedup['runtime_manual_24']['aov_manual'].mean(),
                    "rr_manual": data_system_speedup['runtime_manual_24']['rr_manual'].mean()
                }
            }
        }

        return None

    def initialize_sale_data(self,
                             product_names=None,
                             runtimes=None,
                             product_list=None,
                             views=None,
                             cycles=None,
                             orders=None,
                             discounts=None
                             ):

        if product_names is None:
            self.product_names = ['prime-services-max-5', 'prime-services-max-25', 'internet-security']
        else:
            self.product_names = product_names

        if runtimes is None:
            self.runtimes = ['runtime_1', 'runtime_12']
        else:
            self.runtimes = runtimes

        if product_list is None:
            self.product_list = [('prime-services-max-5', 'runtime_1'),
                                 ('prime-services-max-5', 'runtime_12'),
                                 ('internet-security', 'runtime_1'),
                                 ('internet-security', 'runtime_12'),
                                 ('prime-services-max-25', 'runtime_1'),
                                 ('prime-services-max-25', 'runtime_12')]
        else:
            self.product_list = product_list

            # The number of yearly and monthly cycles to carry out the calculation. Default is 6 years (72 months)
        if cycles is None:
            self.cycles = {
                'runtime_1': 72,
                'runtime_12': 6
            }
        else:
            self.cycles = cycles

        # The data from the orders that were obtained
        #Order: Avira prime (max 5): monthly, yearly
        #Order: Avira internet security: monthly, yearly
        #Order: Avira prime (max 25): monthly, yearly

        if orders is None:
            self.orders = {}
            self.orders['control'] = np.array([94,298,4,44,3,9])
            self.orders['test1'] = np.array([103,479,7,27,9,17])
            self.orders['test2'] = np.array([60,459,5,22,2,20])
        else:
            self.orders = orders

        if discounts is None:
            self.discounts = {}
            self.discounts['control'] = np.array([0,0,0,0,0,0])
            self.discounts['test1'] = np.array([0.6, 0.6, 0.5, 0.5, 0.54, 0.6])
            self.discounts['test2'] = np.array([0.0, 0.6, 0.0, 0.5, 0.00, 0.6])
        else:
            self.discounts = discounts

        if views is None:
            self.views = {
                'control': 507.2*10**3,
                'test1': 506.8*10**3,
                'test2': 507.1*10**3
            }
        else:
            self.views = views

        # Compute the overall conversion rates
        self.model0 = BernoulliModel()
        self.model1 = BernoulliModel()
        self.model2 = BernoulliModel()

        self.n0_total = self.views['control']
        self.n1_total = self.views['test1']
        self.n2_total = self.views['test2']

        # Compute the total number of orders
        self.n0 = np.sum(self.orders['control'])
        self.n1 = np.sum(self.orders['test1'])
        self.n2 = np.sum(self.orders['test2'])


        print('='*100)
        print('Calculating conversion rates')
        self.p0_samples = self.model0.generate_posterior(n_total=self.n0_total, n_success=self.n0, n_iter=8000)
        self.p1_samples = self.model1.generate_posterior(n_total=self.n1_total, n_success=self.n1, n_iter=8000)
        self.p2_samples = self.model2.generate_posterior(n_total=self.n2_total, n_success=self.n2, n_iter=8000)
        print('Conversion rates are completed')
        print('='*100)

        # Here we

        # Find the non-zero subset of order array
        x_control_orders = np.asarray(self.orders['control'])
        x_control_zeros = np.argwhere(x_control_orders == 0).flatten()  # Extract the location of the zeros
        x_control_non_zeros = np.argwhere(x_control_orders > 0).flatten()  # Extract the location of the non-zeros
        x_control_orders = x_control_orders[x_control_orders > 0]

        x_test1_orders = np.asarray(self.orders['test1'])
        x_test1_zeros = np.argwhere(x_test1_orders == 0).flatten()
        x_test1_non_zeros = np.argwhere(x_test1_orders > 0).flatten()  # Extract the location of the non-zeros
        x_test1_orders = x_test1_orders[x_test1_orders > 0]

        x_test2_orders = np.asarray(self.orders['test2'])
        x_test2_zeros = np.argwhere(x_test2_orders == 0).flatten()
        x_test2_non_zeros = np.argwhere(x_test2_orders > 0).flatten()  # Extract the location of the non-zeros
        x_test2_orders = x_test2_orders[x_test2_orders > 0]

        # Next, we generate samples from a dirichlet distribution using the order values
        self.lambda_samples_control = np.zeros(shape=(len(self.p0_samples), len(self.product_list)))
        self.lambda_samples_test1 = np.zeros(shape=(len(self.p0_samples), len(self.product_list)))
        self.lambda_samples_test2 = np.zeros(shape=(len(self.p0_samples), len(self.product_list)))

        l_subset_samples_control = np.random.dirichlet(alpha=x_control_orders, size=len(self.p0_samples))
        l_subset_samples_test1 = np.random.dirichlet(alpha=x_test1_orders, size=len(self.p0_samples))
        l_subset_samples_test2 = np.random.dirichlet(alpha=x_test2_orders, size=len(self.p0_samples))

        # Fill in with zeros the sample array
        for i in range(len(x_control_non_zeros)):
            k = x_control_non_zeros[i]
            self.lambda_samples_control[:, k] = l_subset_samples_control[:, i]

        for i in range(len(x_test1_non_zeros)):
            k = x_test1_non_zeros[i]
            self.lambda_samples_test1[:, k] = l_subset_samples_test1[:, i]

        for i in range(len(x_test2_non_zeros)):
            k = x_test2_non_zeros[i]
            self.lambda_samples_test2[:, k] = l_subset_samples_test2[:, i]


        self.n_samples = len(self.p0_samples)

        self.conversion_rate_simulation = {
            'control': self.p0_samples,
            'test1': self.p1_samples,
            'test2': self.p2_samples
        }

        self.lambda_simulation = {
            'control': self.lambda_samples_control,
            'test1': self.lambda_samples_test1,
            'test2': self.lambda_samples_test2
        }

        return None

    def return_order_samples(self, group, product_name, runtime,n_cycles):

        n0 = self.views[group]

        visitor_cr = self.conversion_rate_simulation[group]


        i=-1

        for k in range(0, len(self.product_list)):

            prod_k , runtime_k = self.product_list[k]

            if product_name == prod_k and runtime == runtime_k:
                i = k

        if i==-1:
            print(product_name, runtime)
            raise Exception('No match found for product name and runtime: {} , {} '.format(product_name, runtime))

        discount = self.discounts[group][i]


        ar_rr, rol_rr, cycle_npv, clv = self.compute_clv_product(
            rr_auto=self.data_retention[product_name][runtime]['rr_auto'],
            rr_manual=self.data_retention[product_name][runtime]['rr_manual'],
            aov_auto=self.data_retention[product_name][runtime]['aov_auto'],
            aov_manual=self.data_retention[product_name][runtime]['aov_manual'],
            n_cycles=n_cycles,
            discount=discount,
            runtime=runtime)

        order_samples = n0*visitor_cr*self.lambda_simulation[group][:,i]


        # THis must be corrected to account for adjusted units...
        adjustment_factor = (self.views['control']/self.views[group])

        clr_samples = order_samples*adjustment_factor*clv

        return order_samples, clr_samples, clv

    def compute_clv_product(self,
                            product_name = None,
                            rr_auto=None,
                            rr_manual=None,
                            aov_auto=None,
                            aov_manual=None,
                            n_cycles=1,
                            discount=0.0,
                            runtime='runtime_1',
                            debug=False):
        """
        Compute the CLV of a specific product with a specific runtime
        """


        ar_rr = np.zeros(n_cycles)
        rol_rr = np.zeros(n_cycles)
        cycle_npv = np.zeros(n_cycles)

        if discount > 0:
            ar_rr[0] = rr_auto*self.f_auto(discount)
            z = rr_manual*self.f_manual(discount)

            # For some values, the extrapolation does not work as the overall RR > 100%, therefore we use the monthly retention values
            if z > 1.0:
                rol_rr[0] = 0.0802*self.f_manual(discount) # Use the prime_5 rentention rate as agreed multiplied by factor from extrapolation
            else:
                rol_rr[0] = z
        else:
            ar_rr[0] = rr_auto
            rol_rr[0] = rr_manual

        if debug is True:
            print()
            print('rol_rr= {}'.format(rol_rr[0]))
            print('ar_rr= {}'.format(ar_rr[0]))
            print()

        cycle_npv[0] = aov_auto * (1.0 - discount)

        # Choose the appropriate parameters according to the runtime of the product
        if runtime == 'runtime_1':
            dr = self.dc_monthly
            ar2ar_rr = self.ar2ar_rr_monthly
            rol2ar_rr = self.rol2ar_rr_monthly
            rol2rol_rr = self.rol2rol_rr_monthly
            ar2rol_rr = self.ar2rol_rr_monthly

        elif runtime == 'runtime_12':
            dr = self.dc_yearly
            ar2ar_rr = self.ar2ar_rr_yearly
            rol2ar_rr = self.rol2ar_rr_yearly
            rol2rol_rr = self.rol2rol_rr_yearly
            ar2rol_rr = self.ar2rol_rr_yearly

        elif runtime == 'runtime_24':
            dr = self.dc_two_years
            ar2ar_rr = self.ar2ar_rr_two_year
            rol2ar_rr = self.rol2ar_rr_two_year
            rol2rol_rr = self.rol2rol_rr_two_year
            ar2rol_rr = self.ar2rol_rr_two_year

        for i in range(1, n_cycles):
            ar_rr[i] = ar_rr[i-1]*ar2ar_rr+rol_rr[i-1]*rol2ar_rr
            rol_rr[i] = rol_rr[i-1]*rol2rol_rr+ar_rr[i-1]*ar2rol_rr
            cycle_npv[i] = (aov_auto*ar_rr[i-1]+aov_manual*rol_rr[i-1])/(1+dr)**i

        return ar_rr, rol_rr, cycle_npv, np.sum(cycle_npv)

    def update_data_conversions(self, group,product_list,lambda_vector,cr_value, data):

        data[group]['overall_conversion_rate'] =  cr_value

        for i in range(len(product_list)):
            (product_name, runtime) = product_list[i]
            data[group][product_name][runtime]['lambda'] =  lambda_vector[i]

        return data


    def compute_bayesian_statistics(self, samples_control, samples_test1, samples_test2):
        """

        """

        lift1 = ((samples_test1-samples_control)/samples_control)*100
        lift2 = ((samples_test2-samples_control)/samples_control)*100

        results_lift1 = utils.return_parameter_uncertainties(lift1)
        results_lift2 = utils.return_parameter_uncertainties(lift2)

        results = {
            'P(test1>control) [%]': (np.sum(samples_test1 > samples_control)/len(samples_test1))*100,
            'P(test2>control) [%]': (np.sum(samples_test2 > samples_control)/len(samples_test1))*100,
            'P(Control is Best) [%]': (np.sum((samples_control > samples_test1) & (samples_control > samples_test2))/len(samples_control))*100,
            'P(test1 is Best) [%]': (np.sum((samples_test1 > samples_control) & (samples_test1 > samples_test2))/len(samples_control))*100,
            'P(test2 is Best) [%]': (np.sum((samples_test2 > samples_test1) & (samples_test2 > samples_control))/len(samples_control))*100,
            'Lift1 median [%]': results_lift1['median'],
            'Lift1 95% CI range [%]': [results_lift1['lower_95_ci'], results_lift1['upper_95_ci']],
            'Lift2 median [%]': results_lift2['median'],
            'Lift2 95% CI range [%]': [results_lift2['lower_95_ci'], results_lift2['upper_95_ci']]
        }

        return results

    def compute_clr_group(self):
        """
        Compute the CLR of the entire group

        TODO: Output the results of CLR to a dataframe file
        """

        clr_samples_control_sum = np.zeros(self.n_samples)
        clr_samples_test1_sum = np.zeros(self.n_samples)
        clr_samples_test2_sum = np.zeros(self.n_samples)

        final_data = []

        for i in range(len(self.product_list)):

            product_name, runtime = self.product_list[i]
            n_cycles = self.cycles[runtime]

            order_samples_control, clr_samples_control, clv_control = self.return_order_samples(group='control',
                                                                                                product_name=product_name,
                                                                                                runtime=runtime,
                                                                                                n_cycles=n_cycles)

            order_samples_test1, clr_samples_test1, clv_test1 = self.return_order_samples(group='test1',
                                                                                          product_name=product_name,
                                                                                          runtime=runtime,
                                                                                          n_cycles=n_cycles)

            order_samples_test2, clr_samples_test2, clv_test2 = self.return_order_samples(group='test2',
                                                                                          product_name=product_name,
                                                                                          runtime=runtime,
                                                                                          n_cycles=n_cycles)


            clr_samples_control_sum += clr_samples_control
            clr_samples_test1_sum += clr_samples_test1
            clr_samples_test2_sum += clr_samples_test2

            print('-')
            print(product_name)
            print('runtime: ', runtime)
            print(' clv = {:.2f} | {:.2f} | {:.2f}'.format(clv_control, clv_test1, clv_test2))
            print(' clr = {:.2f} | {:.2f} | {:.2f}'.format(clr_samples_control.mean(),
                                                                                  clr_samples_test1.mean(),
                                                                                  clr_samples_test2.mean()))
            print('orders = {:.2f} {:.2f} {:.2f}'.format(order_samples_control.mean(),
                                                         order_samples_test1.mean(),
                                                         order_samples_test2.mean()))

            final_data.append([product_name, runtime, 'control', clv_control, clr_samples_control.mean()])
            final_data.append([product_name, runtime, 'test1', clv_test1, clr_samples_test1.mean()])
            final_data.append([product_name, runtime, 'test2', clv_test2, clr_samples_test2.mean()])

        df_results = pd.DataFrame(columns=['product_name','runtime','group', 'clv', 'clr'], data=final_data)

        print(df_results)

        return clr_samples_control_sum, clr_samples_test1_sum, clr_samples_test2_sum, df_results


if __name__ == '__main__':

    product_list = [('prime-services-max-5', 'runtime_1'),
                    ('prime-services-max-5', 'runtime_12')]

    product_names = ['prime-services-max-5']
    product_runtimes = ['runtime_1', 'runtime_12']

    orders = {}
    orders['control'] = np.array([94,2])
    orders['test1'] = np.array([34,0])
    orders['test2'] = np.array([60,0])

    discounts = {}
    discounts['control'] = np.array([0, 0])
    discounts['test1'] = np.array([0.6, 0.6])
    discounts['test2'] = np.array([0.0, 0.6])

    views = {
        'control': 507.2 * 10 ** 3,
        'test1': 506.8 * 10 ** 3,
        'test2': 507.1 * 10 ** 3
    }

    cycles = {
        'runtime_1': 72,
        'runtime_12': 6
    }

    # Instantiate the calculation
    calc = CLV_calculation(
        product_names=product_names,
        runtimes=product_runtimes,
        product_list=product_list,
        orders=orders,
        views=views,
        cycles=cycles
    )

    clr_samples_control_sum = np.zeros(calc.n_samples)
    clr_samples_test1_sum = np.zeros(calc.n_samples)
    clr_samples_test2_sum = np.zeros(calc.n_samples)

    for i in range(len(calc.product_list)):
        product_name, runtime = calc.product_list[i]
        n_cycles = calc.cycles[runtime]

        order_samples_control, clr_samples_control, clv_control = calc.return_order_samples(group='control',
                                                                                            product_name=product_name,
                                                                                            runtime=runtime,
                                                                                            n_cycles=n_cycles)
        order_samples_test1, clr_samples_test1, clv_test1 = calc.return_order_samples(group='test1',
                                                                                      product_name=product_name,
                                                                                      runtime=runtime,
                                                                                      n_cycles=n_cycles)

        order_samples_test2, clr_samples_test2, clv_test2 = calc.return_order_samples(group='test2',
                                                                                      product_name=product_name,
                                                                                      runtime=runtime,
                                                                                      n_cycles=n_cycles)

        clr_samples_control_sum += clr_samples_control
        clr_samples_test1_sum += clr_samples_test1
        clr_samples_test2_sum += clr_samples_test2

        print('-')
        print(product_name, runtime, ' clv = {:.2f} | {:.2f} | {:.2f}'.format(clv_control, clv_test1, clv_test2))
        print('orders = {:.2f} {:.2f} {:.2f}'.format(order_samples_control.mean(),
                                                   order_samples_test1.mean(),
                                                   order_samples_test2.mean()))

    print('exit')