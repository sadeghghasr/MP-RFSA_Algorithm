"""
Author: Sadegh ----> sadeghghasr8817@gmail.com
"""

import cvxpy as cp
import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import block_diag as sparse_block_diag  # sparse matrices are used for problem with higher dimensions
from scipy.sparse import csc_matrix, hstack, vstack
import networkx as nx


######## Definition of functions ########
def get_three_shortest_paths(net_top, src, dst, nPaths):
    three_shortest_paths = []
    sp = nx.shortest_simple_paths(net_top, src, dst)
    k = nPaths  # for simplicity, we use only the shortest path
    for counter, path in enumerate(sp):
        three_shortest_paths.append(path)
        if counter == k - 1:
            break
    return three_shortest_paths

# parameters:
max_number_fiber = 2
n_nodes = 4
list_of_links = [(1, 2), (2, 1), (1, 3), (3, 1), (1, 4), (4, 1), (2, 3), (3, 2), (2, 4), (4, 2), (3, 4), (4, 3)]
n_links = len(list_of_links)
# list_of_demands = [(1, 2), (2, 1), (1, 3), (3, 1)]
list_of_demands = [(1, 2), (2, 1), (1, 3), (3, 1), (1, 4), (4, 1), (2, 3), (3, 2), (2, 4), (4, 2), (3, 4), (4, 3)]
list_of_nodes = [1, 2, 3, 4]

network = nx.Graph()
network.add_nodes_from(list_of_nodes)
network.add_edges_from(list_of_links)



n_lp = 3  # number of lightpath on each fiber of each link
n_demands = len(list_of_demands)
n_transponder_slots = 4  # number of 25 Gbps slots
n_periods = 2
n_paths = 3  # number of paths for each demand
R_t_d = np.array([[1, 1, 4, 4, 6, 6, 3, 3, 5, 5, 1, 1],
                  [2, 2, 4, 4, 4, 4, 3, 3, 7, 7, 6, 6]]).reshape(n_periods * n_demands, 1)
                  # [4, 4, 2, 2, 2, 2, 3, 3, 6, 6, 4, 4, 2, 2, 6, 6, 5, 5, 3, 3]])

# set_of_paths_for_each_demand = [[[[1, 2]]], [[[2, 1]]], [[[1, 2], [2, 3]]], [[[3, 2], [2, 1]]]]  # pattern: [dem0: [path0:[link1, link2, ...], path1, path2], dem1: [path0, path1, path2], ...]
set_of_paths_for_each_demand = []
for dem in list_of_demands:
    shortest_paths = get_three_shortest_paths(network, dem[0], dem[1], n_paths)
    lis_for_each_dem = []
    for path in shortest_paths:
        links_on_each_path = []
        for i in range(len(path) - 1):
            links_on_each_path.append([path[i], path[i+1]])
        lis_for_each_dem.append(links_on_each_path)
    set_of_paths_for_each_demand.append(lis_for_each_dem)

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
        sub_basic_block_A_5_right = np.array([]).reshape((0, n_links * n_lp * max_number_fiber))
        for n_idx in range(n_nodes):
            sub_sub_basic_block = np.array([]).reshape((0, 0))
            for l_idx in range(n_links):
                if list_of_links[l_idx][0] == list_of_nodes[n_idx]:
                    sub_sub_basic_block = block_diag(sub_sub_basic_block, np.eye(max_number_fiber * n_lp))
                else:
                    sub_sub_basic_block = block_diag(sub_sub_basic_block, 0 * np.eye(max_number_fiber * n_lp))

            if list_of_demands[dem][0] == list_of_nodes[n_idx]:
                sub_basic_block_A_5_right = np.concatenate((sub_basic_block_A_5_right, sub_sub_basic_block), axis=0)
            else:
                sub_basic_block_A_5_right = np.concatenate((sub_basic_block_A_5_right, 0 * sub_sub_basic_block), axis=0)

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

    ### constraint 7 + 1 (7p1) ###
A_7p1_left = np.array([]).reshape((0, 0))
for t_period in range(n_periods):
    basic_block_7p1_left = np.array([]).reshape((0, 0))
    for dem in range(n_demands):
        sub_basic_block_A_7p1_left = np.array([]).reshape((0, n_links * n_lp * max_number_fiber))
        for dem_prime in range(n_demands):
            sub_sub_basic_block = np.array([]).reshape((0, 0))
            for l_idx in range(n_links):
                sub_sub_sub_basic_block = np.array([]).reshape((0, max_number_fiber * n_lp))
                for l_idx_prime in range(n_links):
                    if (list_of_links[l_idx][0] == list_of_links[l_idx_prime][1]) and\
                                (list_of_links[l_idx][1] == list_of_links[l_idx_prime][0]):
                        sub_sub_sub_basic_block = np.concatenate((sub_sub_sub_basic_block,
                                                                  np.eye(max_number_fiber * n_lp)), axis=0)
                    else:
                        sub_sub_sub_basic_block = np.concatenate((sub_sub_sub_basic_block,
                                                                  0 * np.eye(max_number_fiber * n_lp)), axis=0)
                sub_sub_basic_block = block_diag(sub_sub_basic_block, sub_sub_sub_basic_block)

            if (list_of_demands[dem][0] == list_of_demands[dem_prime][1]) and \
                    (list_of_demands[dem][1] == list_of_demands[dem_prime][0]):
                sub_basic_block_A_7p1_left = np.concatenate((sub_basic_block_A_7p1_left, sub_sub_basic_block), axis=0)
            else:
                sub_basic_block_A_7p1_left = np.concatenate((sub_basic_block_A_7p1_left, 0 * sub_sub_basic_block), axis=0)

        basic_block_7p1_left = block_diag(basic_block_7p1_left, sub_basic_block_A_7p1_left)

    A_7p1_left = block_diag(A_7p1_left, basic_block_7p1_left)

A_7p1_right = np.array([]).reshape((0, 0))
for t_period in range(n_periods):
    basic_block_7p1_right = np.array([]).reshape((0, n_demands * n_links * n_lp * max_number_fiber))
    for dem in range(n_demands):
        sub_basic_block_A_7p1_right = np.array([]).reshape((0, 0))
        for dem_prime in range(n_demands):
            sub_sub_basic_block = np.array([]).reshape((0, n_links * max_number_fiber * n_lp))
            for l_idx in range(n_links):
                sub_sub_sub_basic_block = np.array([]).reshape((0, 0))
                for l_idx_prime in range(n_links):
                    if (list_of_links[l_idx][0] == list_of_links[l_idx_prime][1]) and \
                            (list_of_links[l_idx][1] == list_of_links[l_idx_prime][0]):
                        sub_sub_sub_basic_block = block_diag(sub_sub_sub_basic_block, np.eye(n_lp * max_number_fiber))
                    else:
                        sub_sub_sub_basic_block = block_diag(sub_sub_sub_basic_block, 0 * np.eye(n_lp * max_number_fiber))

                sub_sub_basic_block = np.concatenate((sub_sub_basic_block, sub_sub_sub_basic_block), axis=0)

            if (list_of_demands[dem][0] == list_of_demands[dem_prime][1]) and \
                    (list_of_demands[dem][1] == list_of_demands[dem_prime][0]):
                sub_basic_block_A_7p1_right = block_diag(sub_basic_block_A_7p1_right, sub_sub_basic_block)
            else:
                sub_basic_block_A_7p1_right = block_diag(sub_basic_block_A_7p1_right, 0 * sub_sub_basic_block)

        basic_block_7p1_right = np.concatenate((basic_block_7p1_right, sub_basic_block_A_7p1_right), axis=0)

    A_7p1_right = block_diag(A_7p1_right, basic_block_7p1_right)

const7p1 = A_7p1_left @ c_t_d_l_f_n == A_7p1_right @ c_t_d_l_f_n

    ### constraint 8 ###
pre_block_A_8 = np.array([]).reshape((0, 0))
for _ in range(max_number_fiber * n_lp):
    pre_block_A_8 = block_diag(pre_block_A_8, np.ones((1, n_transponder_slots)))

A_8 = np.array([]).reshape((0, 0))
for t_period in range(n_periods):
    basic_A_8 = np.array([]).reshape((n_nodes * n_links * max_number_fiber * n_lp, 0))
    for dem in range(n_demands):
        sub_basic_block_A_8 = np.array([]).reshape((0, 0))
        for n_idx in range(n_nodes):
            sub_sub_basic_block = np.array([]).reshape((0, 0))
            for l_idx in range(n_links):
                if list_of_links[l_idx][0] == list_of_nodes[n_idx]:
                    sub_sub_basic_block = block_diag(sub_sub_basic_block, pre_block_A_8)
                else:
                    sub_sub_basic_block = block_diag(sub_sub_basic_block, 0 * pre_block_A_8)

            if list_of_demands[dem][0] == list_of_nodes[n_idx]:
                sub_basic_block_A_8 = block_diag(sub_basic_block_A_8, sub_sub_basic_block)
            else:
                sub_basic_block_A_8 = block_diag(sub_basic_block_A_8, 0 * sub_sub_basic_block)
        basic_A_8 = np.concatenate((basic_A_8, sub_basic_block_A_8), axis=1)
    A_8 = block_diag(A_8, basic_A_8)

const8 = A_8 @ y_t_d_i_l_f_n_k <= n_transponder_slots

    ### constraint 9 ###
basic_block_A_9 = np.array([]).reshape((0, 0))
for _ in range(n_links * max_number_fiber):
    basic_block_A_9 = block_diag(basic_block_A_9, np.ones((1, n_lp)))

A_9 = basic_block_A_9
for _ in range(n_demands * n_periods - 1):
    A_9 = np.concatenate((A_9, basic_block_A_9), axis=1)

big_M = n_periods * n_demands * n_lp
const9_1 = (1/big_M) * A_9 @ c_t_d_l_f_n <= x_l_f
const9_2 = x_l_f <= A_9 @ c_t_d_l_f_n
##################################################################
# Problem Definition:

constrains = [const1_1, const1_2, const2, const3, const4_1, const4_2, const5,
              const6, const7, const7p1, const8, const9_1, const9_2]
objective = cp.Minimize(sum(x_i_l_f_n) + sum(x_l_f))

prob = cp.Problem(objective, constrains)
prob.solve()
print(prob.value)

### Print the results ###
dict_list_fiber = {}
for lin in list_of_links:
    dict_list_fiber['{}'.format(lin)] = 0

for idx in range(len(x_l_f.value.tolist())):
    if x_l_f.value.tolist()[idx][0] == 1:
        link_idx = idx // max_number_fiber
        fiber_idx = idx % max_number_fiber
        dict_list_fiber['{}'.format(list_of_links[link_idx])] += 1
        dict_list_fiber['{}'.format((list_of_links[link_idx][1], list_of_links[link_idx][0]))] += 1

print("In the network we need {} fiber pairs".format(sum(dict_list_fiber.values())/4))

