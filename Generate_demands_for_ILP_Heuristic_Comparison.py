import heuristic
import pickle


demand_used_in_ILP = [(1, 2), (2, 1), (1, 3), (3, 1), (1, 4), (4, 1), (2, 3), (3, 2), (2, 4), (4, 2), (3, 4), (4, 3)]
data_rate_in_each_period_for_each_demand = [[1, 1, 4, 4, 6, 6, 3, 3, 5, 5, 1, 1],
                                            [2, 2, 4, 4, 4, 4, 3, 3, 7, 7, 6, 6]]

t_mat = []  # initilizing traffic matrix

for period_rate in data_rate_in_each_period_for_each_demand:
    dem_for_this_t_period = []
    for rate_idx in range(len(period_rate)):
        dem_for_this_t_period.append(heuristic.Demand(demand_used_in_ILP[rate_idx][0], demand_used_in_ILP[rate_idx][1],
                                                      period_rate[rate_idx], heuristic.my_network))
    t_mat.append(dem_for_this_t_period)

# save traffic matrix for first time:
with open('traffic_matrix_for_ILP_Heuristic_comparison.pickle', 'wb') as f:
    pickle.dump(t_mat, f, protocol=pickle.HIGHEST_PROTOCOL)

