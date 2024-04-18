"""
Author: Sadegh ----> sadeghghasr8817@gmail.com
"""

import cvxpy as cp
import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import block_diag as sparse_block_diag
from scipy.sparse import csc_matrix, hstack, vstack

# parameters:
max_number_fiber = 1
n_nodes = 2
list_of_links = [(1, 2), (2, 1)]
n_links = len(list_of_links)
list_of_demands = [(1, 2), (2, 1)]
list_of_nodes = [1, 2]
n_lightpaths_on_each_fiber = 1
n_demands = len(list_of_demands)
n_transponder_slots = 1  # number of 25 Gbps slots
n_periods = 1

# define the ILP problem:
    ### variables ###
f_l_m = cp.Variable((n_links * max_number_fiber, 1), boolean=True)
T_i_l_m_n = cp.Variable((n_nodes * n_links * max_number_fiber, 1), boolean=True)
c_d_l_m_n_f = cp.Variable((n_demands * n_links * max_number_fiber * n_lightpaths_on_each_fiber * n_lightpaths_on_each_fiber, 1), boolean=True)
s_t_d_i_l_m_n_k = cp.Variable((n_periods * n_demands * n_nodes * n_links * max_number_fiber * n_lightpaths_on_each_fiber * n_transponder_slots, 1), boolean=True)
gamma_t_d_n = cp.Variable((n_periods * n_demands * n_lightpaths_on_each_fiber, 1), boolean=True)

    ### Constraints ###
# cons. 1:
basic_block_A_1_left = np.eye(n_lightpaths_on_each_fiber)
for _ in range(n_lightpaths_on_each_fiber - 1):
    basic_block_A_1_left = np.concatenate((basic_block_A_1_left, np.eye(n_lightpaths_on_each_fiber)), axis=1)

sub_basic_A_1_left = basic_block_A_1_left
for _ in range(max_number_fiber - 1):
    sub_basic_A_1_left = block_diag(sub_basic_A_1_left, basic_block_A_1_left)

sub_sub_basic_A_1_left = sub_basic_A_1_left
for _ in range(n_links - 1):
    sub_sub_basic_A_1_left = block_diag(sub_sub_basic_A_1_left, sub_basic_A_1_left)

A_1_left = sub_sub_basic_A_1_left
for _ in range(n_demands - 1):
    A_1_left = np.concatenate((A_1_left, sub_sub_basic_A_1_left), axis=1)

cons_1 = A_1_left@c_d_l_m_n_f <= 1

#############################################################
# constraint 2:


############################################################
# constraint 3:
sub_basic_block_A_3 = np.ones((1, n_lightpaths_on_each_fiber * n_lightpaths_on_each_fiber))
for _ in range(max_number_fiber - 1):
    sub_basic_block_A_3 = block_diag(sub_basic_block_A_3, np.ones((1, n_lightpaths_on_each_fiber * n_lightpaths_on_each_fiber)))

basic_block_A_3 = sub_basic_block_A_3
for _ in range(n_links - 1):
    basic_block_A_3 = block_diag(basic_block_A_3, sub_basic_block_A_3)

A_3_left = basic_block_A_3
for _ in range(n_demands - 1):
    A_3_left = np.concatenate((A_3_left, basic_block_A_3), axis=1)

cons_3 = A_3_left@c_d_l_m_n_f <= n_demands*n_lightpaths_on_each_fiber*n_lightpaths_on_each_fiber*f_l_m
##########################################################
# constraint 4:
basic_block_A_4_left = np.ones((1, n_lightpaths_on_each_fiber))
for _ in range(n_lightpaths_on_each_fiber*max_number_fiber - 1):
    basic_block_A_4_left = block_diag(basic_block_A_4_left, np.ones((1, n_lightpaths_on_each_fiber)))

A_4_left = np.array([]).reshape(0, 0)
for d_idx in range(n_demands):
    sub_basic_block_A_4_left = np.array([]).reshape(0, n_links*max_number_fiber*n_lightpaths_on_each_fiber*n_lightpaths_on_each_fiber)
    for node_idx in range(n_nodes):
        new_basic_block_for_A4 = np.array([]).reshape(0, 0)
        for link_idx in range(n_links):
            if list_of_links[link_idx][0] == list_of_nodes[node_idx]:
                new_basic_block_for_A4 = block_diag(new_basic_block_for_A4, basic_block_A_4_left)
            else:
                new_basic_block_for_A4 = block_diag(new_basic_block_for_A4, np.zeros(basic_block_A_4_left.shape))

        # sub_basic_block_A_4_left = np.concatenate((sub_basic_block_A_4_left, new_basic_block_for_A4), axis=0)
        #
        if list_of_nodes[node_idx] == list_of_demands[d_idx][0]:
            sub_basic_block_A_4_left = np.concatenate((sub_basic_block_A_4_left, new_basic_block_for_A4), axis=0)
        else:
            sub_basic_block_A_4_left = np.concatenate((sub_basic_block_A_4_left, np.zeros(new_basic_block_for_A4.shape)), axis=0)

    A_4_left = block_diag(A_4_left, sub_basic_block_A_4_left)

A_4_right = np.array([]).reshape(0, n_nodes * n_links * max_number_fiber * n_lightpaths_on_each_fiber)

for d_idx in range(n_demands):
    basic_block_A4_right = np.array([]).reshape(0, 0)
    for n_idx in range(n_nodes):
        sub_basic_A4_right = np.array([]).reshape(0, 0)
        for l_idx in range(n_links):
            if list_of_links[l_idx][0] == list_of_nodes[n_idx]:
                sub_basic_A4_right = block_diag(sub_basic_A4_right, np.eye(max_number_fiber * n_lightpaths_on_each_fiber))
            else:
                sub_basic_A4_right = block_diag(sub_basic_A4_right, 0*np.eye(max_number_fiber * n_lightpaths_on_each_fiber))
        if list_of_demands[d_idx][0] == list_of_nodes[n_idx]:
            basic_block_A4_right = block_diag(basic_block_A4_right, sub_basic_A4_right)
        else:
            basic_block_A4_right = block_diag(basic_block_A4_right, 0 * sub_basic_A4_right)

    A_4_right = np.concatenate((A_4_right, basic_block_A4_right), axis=0)

cons_4 = A_4_left@c_d_l_m_n_f == A_4_right@T_i_l_m_n
################################################################
# constraint 5:
A_5_right = A_4_left


