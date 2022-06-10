# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 20:32:54 2022
UCB ClassFile
@author: TSE
"""

# =============================================================================
# Import Packages
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

simulation_dataset = pd.read_csv('Ads ClickThroughRate.csv')
# =============================================================================
# Variable Declaration
# =============================================================================
N = 10000  # Number of Rounds
d = 10     # Number of Ads
#                                          Ad0  Ad1  Ad2   Ad3
number_of_times_ad_displayed = [0]*d   # [1,    0,    0,   0,    0, 0, 0, 0, 0, 0]
sum_of_rewards = [0]*d
ad_displayed = []
# =============================================================================
# Ad Choosing Logic for Each Round
# =============================================================================
for n in range(0,N):              # Number of Rounds
    max_ucb=0
    for i in range(0,d):          # Number of Ads        
        if number_of_times_ad_displayed[i] > 0:        
            average_reward = sum_of_rewards[i]/number_of_times_ad_displayed[i]
            delta_i = math.sqrt(3/2*math.log(n+1)/number_of_times_ad_displayed[i])        
            current_ucb = average_reward + delta_i
        else:
            current_ucb = 1e400   # is used as assumed UCB for Ads that have got no chance uptil round 10
            
        if current_ucb > max_ucb:            
            max_ucb = current_ucb
            ad = i 
                                   
    ad_displayed.append(ad)            
    number_of_times_ad_displayed[ad] = number_of_times_ad_displayed[ad] + 1
    """In round n for Ad ad is it a click or no-click"""
    clicked_reward = simulation_dataset.values[n,ad]
    sum_of_rewards[ad] = sum_of_rewards[ad] + clicked_reward
    
    
plt.hist(ad_displayed)
plt.show()























