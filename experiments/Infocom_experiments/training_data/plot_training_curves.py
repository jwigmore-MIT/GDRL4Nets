import matplotlib.pyplot as plt
import pandas as pda
import matplotlib as mpl
import numpy as np

# import csv data

pd = pda.read_csv('SP_training_data2.csv')
step_key = 'Step'

# get column names and keep only step and the first one that contains PMN and MLP
PMN_columns = [col for col in pd.columns if 'PMN' in col]
MLP_columns = [col for col in pd.columns if 'MLP' in col]

pd = pd[[step_key, PMN_columns[-1], MLP_columns[-1]]]
# only keep row if step is divisible by 10000
pd = pd[pd[step_key] % 10000 == 0]

#multiply all training steps by 2
# pd[step_key] = pd[step_key] * 2


## plot the data
# increase the font
plt.figure(figsize=(15, 10))
mpl.rcParams.update({'font.size': 24, "font.family": "Arial"})

plt.plot(pd[step_key], np.log(pd[MLP_columns[-1]]), label='MLP')
plt.plot(pd[step_key], np.log(pd[PMN_columns[-1]]), label='STN')
plt.xlabel('Training Step', fontsize=24)
plt.ylabel('Log Average Normalized Cost', fontsize=24)
plt.title('Training Curves', fontsize=24)
plt.legend(fontsize=24)
# Save the figure
plt.savefig('training_curves_SP.pdf')
plt.show()

# save the plot

