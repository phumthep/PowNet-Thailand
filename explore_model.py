from pyomo.environ import *
from pyomo.core.expr.current import identify_variables
from pyomo.repn.standard_repn import generate_standard_repn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

#%% Extract info on the variables
var_dict = {}
var_names_list = []
for vari in instance.component_objects(Var):
    var_dict[vari.name] = {
        'values': list(vari.values()),
        'num_vars': len(list(vari.values()))
        }
    for idx in vari.keys():
        idx_name =  [str(x) for x in idx]
        var_names_list.append(
            vari.name + '[' + idx_name[0] + ',' + idx_name[1] + ']'
            )
        

# Calculate the number of total variables
num_tot_var = 0
for k, v in var_dict.items():
    num_tot_var += v['num_vars']
print('Total number of variables: {}'.format(num_tot_var))


#%% Extract info on the constraints
# c_dict = {}
# counting = 0
# for c in instance.component_objects(Constraint):
#     for cc in list(c.values()):
#         # Print Diagnostics
#         if counting % 2500 == 0:
#             print(
#                 'Still processing... Constraint #{} {}'\
#                     .format(counting, cc.name)
#                   )
#         # Extract variables from the current constraint
#         c_dict[cc.name] = {
#             'constraints': [str(x) for x in identify_variables(cc.body)]
#             }
#         counting += 1
    


# # Save the c_dict as pickle
# with open('constraint_data.pickle', 'wb') as f:
#     pickle.dump(c_dict, f)

with open('constraint_data.pickle', 'rb') as f:
    c_dict_loaded = pickle.load(f)
    
print('Number of constraints: ', len(c_dict))

#%% Create A matrix
# Initialize an empty dataframe
if 
A_mat = pd.DataFrame(
    False, dtype='bool', columns=var_names_list, index = c_dict.keys()
    )

print(A_mat.memory_usage().sum() / 1e3)


# Iterate through the constraints and change the coef to 1 if the
# variable is present in the constraint
counting = 0
for c_name, c_vars in c_dict.items():
    for _, c_vals in c_vars.items():
        for c_val in c_vals:
            # Print Diagnostics
            if counting % 2500 == 0:
                print('Still processing... Constraint # {}'.format(counting))
            
            A_mat.loc[c_name, c_val] = True
        counting += 1

A_mat = A_mat.astype(pd.SparseDtype('bool'), np.nan)
print(A_mat.memory_usage().sum() / 1e3)

# Save the dataframe
A_mat.to_pickle('A_mat.pickle')

with open('A_mat.pickle', 'rb') as f:
    A_mat = pickle.load(f)
    

#%% Visualize the A matrix
#yticks_labels = range(0, 97000, 5000)
yticks_labels = [
    0, 6312, 10008, 13704, 14040, 14400, 14448, 14664, 32472,
    50280, 50304, 50328, 54042, 57720, 61416, 61570,
    61724, 65420, 69116, 72812, 82554, 92296, 95992
    ]
fig, ax = plt.subplots(figsize=(10,10), dpi=500)
plt.spy(A_mat, markersize=1)
plt.xticks(
    [0, 3850, 7700, 11550, 15400, 19250, 19600, 19975, 20025, 20250, 26800], 
    rotation=40
    )
plt.yticks(yticks_labels)
plt.show()

#%% Stats analysis
# Max number of variables in a constraint
max_vars = A_mat.sum(axis=1).max()

# Constraint breakdown
c_info = {}
for c in instance.component_objects(Constraint):
    c_info[c.name] = {'num_rows': len(c._data)}

# Get the cumulative sum
cons_df = pd.DataFrame(c_info).T
cons_df['cumsum'] = cons_df['num_rows'].cumsum()

# Var breakdown
var_info = {}
for va_name, va_val in var_dict.items():
    var_info[va_name] = va_val['num_vars']
    
# Get the cumulative sum
var_df = pd.DataFrame(var_info, index=['num_vars']).T
var_df['cumsum'] = var_df['num_vars'].cumsum()
