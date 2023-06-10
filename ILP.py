"""
This a project defined by professor Luis Velasco for PhD interview
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
n_lp = 1  # number of lightpath on each fiber of each link
n_demands = len(list_of_demands)
n_transponder_slots = 1  # number of 25 Gbps slots
n_periods = 2
n_paths = 3  # number of paths for each demand
R_t_d = np.array([[1, 1], [2, 2]]).reshape(n_periods * n_demands, 1)
set_of_paths_for_each_demand = [[[1, 2]]]  # pattern: [dem0: [path0, path1, path2], dem1: [path0, path1, path2], ...]

# ILP variables:
x_i_l_f_n = cp.Variable((n_nodes * n_links * max_number_fiber * n_lp, 1), boolean=True)
x_t_d_i_l_f_n = cp.Variable((n_periods * n_demands * n_nodes * n_links * max_number_fiber * n_lp, 1), boolean=True)
y_t_d_i_l_f_n_k = cp.Variable((n_periods * n_demands * n_nodes * n_links * max_number_fiber * n_lp *
                               n_transponder_slots, 1), boolean=True)
c_t_d_l_f_n = cp.Variable((n_periods * n_demands * n_links * max_number_fiber * n_lp, 1), boolean=True)
x_l_f = cp.Variable((n_links * max_number_fiber, 1), boolean=True)
x_t_d_p_f_n = cp.Variable((n_periods * n_demands * n_paths * max_number_fiber * n_lp, 1))

# ILP constraints:

    ### Constraint 1 ###
A_1_left = np.array([]).reshape((n_nodes * n_links * max_number_fiber * n_lp, 0))

for t_period in range(n_periods):
    basic_block_A_1_left = np.array([]).reshape((n_nodes * n_links * max_number_fiber * n_lp, 0))
    for dem in range(n_demands):
        sub_basic_block_A_1_left = np.array([]).reshape((0, 0))
        for n_idx in range(n_nodes):
            sub_sub_basic_block = np.array([]).reshape((0, 0))
            for l_idx in range(n_links):
                if list_of_links[l_idx][0] == list_of_nodes[n_idx]:
                    sub_sub_basic_block = block_diag(sub_sub_basic_block, np.eye(max_number_fiber * n_lp))
                else:
                    sub_sub_basic_block = block_diag(sub_sub_basic_block, 0 * np.eye(max_number_fiber * n_lp))
            if list_of_demands[dem][0] == list_of_nodes[n_idx]:
                sub_basic_block_A_1_left = block_diag(sub_basic_block_A_1_left, sub_sub_basic_block)
            else:
                sub_basic_block_A_1_left = block_diag(sub_basic_block_A_1_left, 0 * sub_sub_basic_block)
        basic_block_A_1_left = np.concatenate((basic_block_A_1_left, sub_basic_block_A_1_left), axis=1)
    A_1_left = np.concatenate((A_1_left, basic_block_A_1_left), axis=1)

A_1_middle = np.array([]).reshape((0, 0))
for n_idx in range(n_nodes):
    basic_block_A_1_middle = np.array([]).reshape((0, 0))
    for l_idx in range(n_links):
        if list_of_links[l_idx][0] == list_of_nodes[n_idx]:
            basic_block_A_1_middle = block_diag(basic_block_A_1_middle, np.eye(max_number_fiber * n_lp))
        else:
            basic_block_A_1_middle = block_diag(basic_block_A_1_middle, 0 * np.eye(max_number_fiber * n_lp))
    A_1_middle = block_diag(A_1_middle, basic_block_A_1_middle)

const1_1 = (1/(n_demands * n_periods)) * A_1_left @ x_t_d_i_l_f_n <= A_1_middle @ x_i_l_f_n
const1_2 = A_1_middle @ x_i_l_f_n <= A_1_left @ x_t_d_i_l_f_n

    ### constraint 2 ###
A_2_left = np.array([]).reshape((0, 0))
for t_period in range(n_periods):
    basic_block_A_2_left = np.array([]).reshape((n_links * max_number_fiber * n_lp, 0))
    for dem in range(n_demands):
        basic_block_A_2_left = np.concatenate((basic_block_A_2_left, np.eye(n_links * max_number_fiber * n_lp)), axis=1)
    A_2_left = block_diag(A_2_left, basic_block_A_2_left)

const2 = A_2_left @ c_t_d_l_f_n <= 1

    ### constraint 3 ###
A_3_left = np.array([]).reshape((0, 0))
for t_period in range(n_periods):
    basic_block_A_3_left = np.array([]).reshape((0, 0))
    for dem in range(n_demands):
        sub_basic_block_A_3_left = np.array([]).reshape((1, 0))
        for n_idx in range(n_nodes):
            sub_sub_basic_block = np.array([]).reshape((1, 0))
            for l_idx in range(n_links):
                if list_of_links[l_idx][0] == list_of_nodes[n_idx]:
                    sub_sub_basic_block = np.concatenate((sub_sub_basic_block,
                                                          np.ones((1, max_number_fiber * n_lp * n_transponder_slots))),
                                                         axis=1)
                else:
                    sub_sub_basic_block = np.concatenate((sub_sub_basic_block,
                                                          np.zeros((1, max_number_fiber * n_lp * n_transponder_slots))),
                                                         axis=1)
            if list_of_demands[dem][0] == list_of_nodes[n_idx]:
                sub_basic_block_A_3_left = np.concatenate((sub_basic_block_A_3_left, sub_sub_basic_block), axis=1)
            else:
                sub_basic_block_A_3_left = np.concatenate((sub_basic_block_A_3_left, 0 * sub_sub_basic_block), axis=1)

        basic_block_A_3_left = block_diag(basic_block_A_3_left, sub_basic_block_A_3_left)

    A_3_left = block_diag(A_3_left, basic_block_A_3_left)

const3 = A_3_left @ y_t_d_i_l_f_n_k == R_t_d  # this 3 number should be changed to R_d_t

    ### constraint 4 ###
pre_block_for_A_4_left = np.array([]).reshape((0, 0))
for _ in range(max_number_fiber * n_lp):
    pre_block_for_A_4_left = block_diag(pre_block_for_A_4_left, np.ones((1, n_transponder_slots)))

A_4_left = np.array([]).reshape((0, 0))
for t_period in range(n_periods):
    basic_block_A_4_left = np.array([]).reshape((0, 0))
    for dem in range(n_demands):
        sub_basic_block_A_4_left = np.array([]).reshape((0, 0))
        for n_idx in range(n_nodes):
            sub_sub_basic_block = np.array([]).reshape(0, 0)
            for l_idx in range(n_links):
                if list_of_links[l_idx][0] == list_of_nodes[n_idx]:
                    sub_sub_basic_block = block_diag(sub_sub_basic_block, pre_block_for_A_4_left)
                else:
                    sub_sub_basic_block = block_diag(sub_sub_basic_block, 0 * pre_block_for_A_4_left)

            if list_of_demands[dem][0] == list_of_nodes[n_idx]:
                sub_basic_block_A_4_left = block_diag(sub_basic_block_A_4_left, sub_sub_basic_block)
            else:
                sub_basic_block_A_4_left = block_diag(sub_basic_block_A_4_left, 0 * sub_sub_basic_block)
        basic_block_A_4_left = block_diag(basic_block_A_4_left, sub_basic_block_A_4_left)
    A_4_left = block_diag(A_4_left, basic_block_A_4_left)

pre_block_for_A_4_middle = np.array([]).reshape((0, 0))
for _ in range(max_number_fiber * n_lp):
    pre_block_for_A_4_middle = block_diag(pre_block_for_A_4_middle, np.ones((1, 1)))

A_4_middle = np.array([]).reshape((0, 0))
for t_period in range(n_periods):
    basic_block_A_4_middle = np.array([]).reshape((0, 0))
    for dem in range(n_demands):
        sub_basic_block_A_4_middle = np.array([]).reshape((0, 0))
        for n_idx in range(n_nodes):
            sub_sub_basic_block = np.array([]).reshape(0, 0)
            for l_idx in range(n_links):
                if list_of_links[l_idx][0] == list_of_nodes[n_idx]:
                    sub_sub_basic_block = block_diag(sub_sub_basic_block, pre_block_for_A_4_middle)
                else:
                    sub_sub_basic_block = block_diag(sub_sub_basic_block, 0 * pre_block_for_A_4_middle)

            if list_of_demands[dem][0] == list_of_nodes[n_idx]:
                sub_basic_block_A_4_middle = block_diag(sub_basic_block_A_4_middle, sub_sub_basic_block)
            else:
                sub_basic_block_A_4_middle = block_diag(sub_basic_block_A_4_middle, 0 * sub_sub_basic_block)
        basic_block_A_4_middle = block_diag(basic_block_A_4_middle, sub_basic_block_A_4_middle)
    A_4_middle = block_diag(A_4_middle, basic_block_A_4_middle)

const4_1 = (1/n_transponder_slots) * A_4_left @ y_t_d_i_l_f_n_k <= A_4_middle @ x_t_d_i_l_f_n
const4_2 = A_4_middle @ x_t_d_i_l_f_n <= A_4_left @ y_t_d_i_l_f_n_k

    ### constraint 5 ###
A_5_left = A_4_middle

A_5_right = np.array([]).reshape((0, 0))
for t_period in range(n_periods):
    basic_block_A_5_right = np.array([]).reshape((0, 0))
    for dem in range(n_demands):
        sub_basic_block_A_5_right = np.array([]).reshape((0, n_links * n_transponder_slots * max_number_fiber))
        for n_idx in range(n_nodes):
            sub_sub_basic_block = np.array([]).reshape((0, 0))
            for l_idx in range(n_links):
                if list_of_links[l_idx][0] == list_of_nodes[n_idx]:
                    sub_sub_basic_block = block_diag(sub_sub_basic_block, np.eye(max_number_fiber * n_transponder_slots))
                else:
                    sub_sub_basic_block = block_diag(sub_sub_basic_block, 0 * np.eye(max_number_fiber * n_transponder_slots))

            if list_of_demands[dem][0] == list_of_nodes[n_idx]:
                sub_basic_block_A_5_right = np.concatenate((sub_basic_block_A_5_right, sub_sub_basic_block), 0)
            else:
                sub_basic_block_A_5_right = np.concatenate((sub_basic_block_A_5_right, 0 * sub_sub_basic_block), 0)

        basic_block_A_5_right = block_diag(basic_block_A_5_right, sub_basic_block_A_5_right)

    A_5_right = block_diag(A_5_right, basic_block_A_5_right)

const5 = A_5_left @ x_t_d_i_l_f_n <= A_5_right @ c_t_d_l_f_n

    ### constraint 6 ###
A_6_left = np.array([]).reshape((0, 0))
for _ in range(n_periods * n_demands):
    A_6_left = block_diag(A_6_left, np.ones((1, n_paths * max_number_fiber * n_lp)))

const6 = A_6_left @ x_t_d_p_f_n == np.ceil((1/n_transponder_slots) * R_t_d)

    ### constraint 7 ###
A_7_left = np.array([]).reshape((0, 0))
for t_period in range(n_periods):
    basic_block_A_7_left = np.array([]).reshape((0, 0))
    for dem in range(n_demands):
        sub_basic_block_A_7_left = np.array([]).reshape((0, n_links * n_lp * max_number_fiber))
        for p_idx in range(n_paths):
            sub_sub_basic_block = np.array([]).reshape((0, 0))
            for l_idx in range(n_links):
                if list_of_links[l_idx] in set_of_paths_for_each_demand[dem][p_idx]:
                    sub_sub_basic_block = block_diag(sub_sub_basic_block, np.eye(n_lp * max_number_fiber))
                else:
                    sub_sub_basic_block = block_diag(sub_sub_basic_block, 0 * np.eye(n_lp * max_number_fiber))
            sub_basic_block_A_7_left = np.concatenate((sub_basic_block_A_7_left, sub_sub_basic_block), axis=0)
        basic_block_A_7_left = block_diag(basic_block_A_7_left, sub_basic_block_A_7_left)
    A_7_left = block_diag(A_7_left, basic_block_A_7_left)

A_7_right = np.array([]).reshape((0, 0))
for t_period in range(n_periods):
    basic_block_A_7_right = np.array([]).reshape((0, 0))
    for dem in range(n_demands):
        sub_basic_block_A_7_right = np.array([]).reshape((0, 0))
        for p_idx in range(n_paths):
            sub_sub_basic_block = np.array([]).reshape((0, n_lp * max_number_fiber))
            for l_idx in range(n_links):
                if list_of_links[l_idx] in set_of_paths_for_each_demand[dem][p_idx]:
                    sub_sub_basic_block = np.concatenate((sub_sub_basic_block, np.eye(n_lp * max_number_fiber)), axis=0)
                else:
                    sub_sub_basic_block = np.concatenate((sub_sub_basic_block, 0 * np.eye(n_lp * max_number_fiber)), axis=0)
            sub_basic_block_A_7_right = block_diag(sub_basic_block_A_7_right, sub_sub_basic_block)
        basic_block_A_7_right = block_diag(basic_block_A_7_right, sub_basic_block_A_7_right)
    A_7_right = block_diag(A_7_right, basic_block_A_7_right)

const7 = A_7_left @ c_t_d_l_f_n == A_7_right @ x_t_d_p_f_n

    ### constraint 8 ###




