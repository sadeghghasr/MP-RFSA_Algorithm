"""
This program addresses the multi-fiber, multi-period planning for optical networks with fixed-rate transponders.
Author: Sadegh
Email: sadeghghasr8817@gmail.com
Title: MP-RFSA Algorithm
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from tabulate import tabulate
from contextlib import redirect_stdout
import openpyxl
import pickle


############## Global Variables ##############
global n_channels_in_each_fiber
n_channels_in_each_fiber = 3

global random_fit_spectrum_assignment
random_fit_spectrum_assignment = 1

global perform_grooming
perform_grooming = 0

############# Class Definitions ###############
class Link:
    """
        This class determines the features of links
    """
    def __init__(self, src, dst, length):
        self.max_number_of_fibers_in_this_link = 4
        self.src = src
        self.dst = dst
        self.length = length
        self.list_of_fibers = []
        for _ in range(self.max_number_of_fibers_in_this_link):
            self.list_of_fibers.append(Fiber())
        self.state_of_fibers = [0 for _ in range(self.max_number_of_fibers_in_this_link)]


class Fiber:
    """
        This class determines the features of fibers
    """
    def __init__(self):
        self.channels_state = [0 for _ in range(n_channels_in_each_fiber)]

    def fill_channel(self, idx_of_channel):
        self.channels_state[idx_of_channel] = 1

    def find_index_of_first_available_channel(self):
        if random_fit_spectrum_assignment == 1:
            return random.randint(0, n_channels_in_each_fiber-1)
        else:
            if 0 in self.channels_state:
                return self.channels_state.index(0)
            else:
                print("there is no available channel")
                return None


class Transponder:
    """
    This class addresses the features of transponders
    """
    def __init__(self, basic_demand, all_obj_links):
        self.reference_node = basic_demand.src
        self.all_obj_links = all_obj_links
        self.reference_fiber = None
        self.other_terminal_node = basic_demand.dst
        self.reference_link = None
        self.assigned_path_for_demand = None
        self.assigned_demand = basic_demand
        self.assigned_channel_idx = None
        self.n_slots = 4  # this is number of 25 Gbps slots
        self.state_of_slots = [0 for _ in range(self.n_slots)]
        self.state_of_time_period_each_slot = [-1 for _ in range(self.n_slots)]
        self.transponder_is_released_after_usage = 0
        self.transponder_is_active = 1

    def set_assigned_path_for_reference_demand(self, path):
        self.assigned_path_for_demand = path
        self.reference_link = get_all_links_of_path(path)

    def set_fibers_idx_on_each_link(self, set_of_fiber_idx):
        self.reference_fiber = set_of_fiber_idx

    def fill_slots(self, number_of_required_slots, time_period):
        n_assigned = 0
        if number_of_required_slots > self.get_remain_capacity():
            print("This transponder does not have enough capacity to accommodate {} slots.".format(number_of_required_slots))
        for i in range(self.n_slots):
            if self.state_of_slots[i] == 0:
                self.state_of_slots[i] = 1
                self.state_of_time_period_each_slot[i] = time_period
                n_assigned += 1
            if n_assigned == number_of_required_slots:
                break

    def get_remain_capacity(self):
        return self.n_slots - sum(self.state_of_slots)

    def release_transponder_slot(self, idx_of_released_slot, n_slot=1):
        self.state_of_slots[idx_of_released_slot] = 0
        self.state_of_time_period_each_slot[idx_of_released_slot] = -1
        if sum(self.state_of_slots) == 0:
            self.fully_release_transponder()

    def fully_release_transponder(self):
        for l_id in range(len(self.reference_link)):
            for obj_ll in self.all_obj_links:
                if (obj_ll.src == self.reference_link[l_id][0] and obj_ll.dst == self.reference_link[l_id][1]) or (
                        obj_ll.src == self.reference_link[l_id][1] and obj_ll.dst == self.reference_link[l_id][0]):
                    obj_ll.list_of_fibers[self.reference_fiber[l_id]].channels_state[self.assigned_channel_idx] = 0
                # if sum(obj_ll.list_of_fibers[self.reference_fiber[l_id]].channels_state) == 0:
                #     obj_ll.state_of_fibers[self.reference_fiber[l_id]] = 0

        # everything must be reset except the installed node
        self.reference_fiber = None
        self.reference_link = None
        self.assigned_path_for_demand = None
        self.assigned_channel_idx = None
        self.transponder_is_released_after_usage = 1
        self.transponder_is_active = 0
        print("The transponder is released")

    def reset_assigned_demand(self, b_dem):
        self.assigned_demand = b_dem

    def activate_transponder(self):
        self.transponder_is_active = 1


class Lightpath:
    """
        This class addresses the features of transponders
    """
    def __init__(self, basic_demand, dedicated_transponder):
        self.basic_demand = basic_demand
        self.path = None
        self.selected_channel_idx = None
        self.dedicated_transponder = dedicated_transponder

    def assign_selected_path(self, path_idx):
        self.path = self.basic_demand.three_shortest_paths[path_idx]

    def assign_channel_idx(self, idx_of_channel):
        self.selected_channel_idx = idx_of_channel
        self.dedicated_transponder.assigned_channel_idx = idx_of_channel

    def release_transponders_slot(self, number_of_released_slots=1):
        final_index_of_1_in_dedicated_transponder_slot_state = max(index for index, item in enumerate(self.dedicated_transponder.state_of_slots) if item == 1)
        self.dedicated_transponder.release_transponder_slot(final_index_of_1_in_dedicated_transponder_slot_state,
                                                            number_of_released_slots)

    def get_transponder(self):
        return self.dedicated_transponder

    def is_completely_empty(self):
        return True if sum(self.dedicated_transponder.state_of_slots) == 0 else False


class Demand:
    """
        This class addresses the features of demands
    """
    def __init__(self, src, dst, n_25G_traffic, network_topology):
        self.src = src
        self.dst = dst
        self.n_25G_traffic = n_25G_traffic
        self.network_topology = network_topology
        self.three_shortest_paths = []
        self.number_of_served_slots = 0

    def assign_three_shortest_path(self):
        sp = nx.shortest_simple_paths(self.network_topology, self.src, self.dst)
        k = 3
        for counter, path in enumerate(sp):
            self.three_shortest_paths.append(path)
            if counter == k - 1:
                break

    def get_remained_25G_slots(self):
        return self.n_25G_traffic - self.number_of_served_slots

    def get_three_shortest_paths(self):
        self.three_shortest_paths = []
        sp = nx.shortest_simple_paths(self.network_topology, self.src, self.dst)
        k = 3  # for simplicity, we use only the shortest path
        for counter, path in enumerate(sp):
            self.three_shortest_paths.append(path)
            if counter == k - 1:
                break
        return self.three_shortest_paths


class TimePeriod:
    """
        This class determines the features of time periods
    """
    def __init__(self):
        self.traffic_matrix = []
    def generate_traffic_matrix(self, list_of_nodes, net):
        for _ in range(1):
            for idx in range(len(list_of_nodes)):
                for idx_prime in range(len(list_of_nodes)):
                    if idx != idx_prime:
                        if there_is_a_demand_with_these_terminal(self.traffic_matrix, list_of_nodes[idx], list_of_nodes[idx_prime]):
                            continue
                        else:
                            n_traffic = random.randint(1, 7)
                            self.traffic_matrix.append(Demand(list_of_nodes[idx], list_of_nodes[idx_prime],
                                                         n_traffic , net))  #random.randint(5, 7)
                            # self.traffic_matrix.append(Demand(list_of_nodes[idx_prime], list_of_nodes[idx],
                            #                                   n_traffic, net))  # random.randint(5, 7)

        # self.traffic_matrix = [self.traffic_matrix[1]]#, self.traffic_matrix[3], self.traffic_matrix[6],
        #                        self.traffic_matrix[8], self.traffic_matrix[9], self.traffic_matrix[13],
        #                        self.traffic_matrix[19], self.traffic_matrix[18], self.traffic_matrix[15]]
        # self.traffic_matrix = random.sample(self.traffic_matrix, 1)
        return self.traffic_matrix

    def load_traffic_matrix(self, traffic_mat):
        self.traffic_matrix = traffic_mat


################## Function Definitions ##################

def get_all_links_of_path(ph, reverse=False):
    list_of_links = []
    for n in range(len(ph) - 1):
        list_of_links.append((ph[n], ph[n+1]))
        if reverse == True:
            list_of_links.append((ph[n + 1], ph[n]))
    return list_of_links


def find_index_of_available_fiber_for_specific_channel(set_of_fibers, my_channel_idx):
    r_idx = None
    for f_idx in range(len(set_of_fibers)):
        if set_of_fibers[f_idx].channels_state[my_channel_idx] == 0:
            r_idx = f_idx
            break

    return r_idx


def there_is_a_demand_with_these_terminal(dem_list, t1, t2):
    r = 0
    for d in dem_list:
        if (d.src == t1 and d.dst == t2) or (d.src == t2 and d.dst == t1):
            r = 1
            break
    return r


################# Main Body of The Project ##############

### read network topology ###:
# wb = openpyxl.load_workbook("Reference_Network_TID_RegionalA.xlsx")
# ws = wb.active
# cell_range = 'A2:C52'
# link_list = [[cell.value for cell in row] for row in ws[cell_range]]

network = nx.Graph()
# all_nodes = [k + 1 for k in range(max([l[1] for l in link_list]))]
all_nodes = [1, 2, 3, 4]
num_nodes = len(all_nodes)
network.add_nodes_from(all_nodes)
# list_of_all_links = [tuple(j) for j in link_list]
list_of_all_links = [(1, 2, 100), (1, 3, 100), (1, 4, 100), (2, 3, 100), (2, 4, 100), (3, 4, 100)]

network.add_weighted_edges_from(list_of_all_links, weight='length')
all_links = []
for lin in list_of_all_links:
    all_links.append(Link(lin[0], lin[1], lin[2]))
    all_links.append(Link(lin[1], lin[0], lin[2]))

my_network = nx.DiGraph(network)

### Time periods and generate traffic matrix ###
total_number_of_periods = 2
time_periods = []
traffic_matrix = []
for i in range(total_number_of_periods):
    tp = TimePeriod()
    traffic_matrix.append(tp.generate_traffic_matrix(all_nodes, my_network))
    time_periods.append(tp)


if __name__ == "__main__":

    # save traffic matrix for first time:
    # with open('traffic_matrix10.pickle', 'wb') as f:
    #     pickle.dump(traffic_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)

    # load traffic matrix:
    with open('traffic_matrix_for_ILP_Heuristic_comparison.pickle', 'rb') as f:
        traffic_matrix = pickle.load(f)

    ### main algorithm ###
    with open('results_v3.txt', 'w', encoding="utf-8") as f:
        with redirect_stdout(f):
            list_of_all_used_transponders = []
            list_of_all_used_lightpaths = []
            list_of_maximum_used_fiber_index_for_each_period = []
            for period_idx in range(len(time_periods)):
                period = time_periods[period_idx]
                period.load_traffic_matrix(traffic_matrix[period_idx])#[traffic_matrix[period_idx][1], traffic_matrix[period_idx][4]])
                print("\n***************************************** Time Period {} ****************************************\n".format(period_idx))
                for dem_idx in range(len(period.traffic_matrix)):
                    dem = period.traffic_matrix[dem_idx]
                    if period_idx == 0:
                        n_lightpaths = int(np.ceil(dem.n_25G_traffic/4))
                        list_of_lightpath = []
                        for _ in range(n_lightpaths):
                            trans = Transponder(dem, all_links)
                            list_of_lightpath.append(Lightpath(dem, trans))
                            list_of_all_used_transponders.append(trans)

                        for lightpath in list_of_lightpath:
                            list_of_all_used_lightpaths.append(lightpath)
                            # print("this is for lp {}".format(lightpath))
                            served_slots = dem.number_of_served_slots
                            for i in range(dem.n_25G_traffic - served_slots):
                                if i > 3:
                                    break
                                else:
                                    dem.number_of_served_slots += 1
                                    lightpath.dedicated_transponder.fill_slots(1, period_idx)

                            index_of_last_used_fiber_for_each_path = []
                            available_index_of_channel = []
                            available_fibers_on_each_link_of_the_path = []
                            fiber_object_for_each_link_of_the_path = []
                            # print(dem.get_three_shortest_paths())
                            for path_idx in range(len(dem.get_three_shortest_paths())):
                                path = dem.get_three_shortest_paths()[path_idx]
                                # print("hello {} and index {}".format(path, path_idx))
                                fibers_list = []
                                fiber_object = []
                                links_on_this_path = get_all_links_of_path(path)
                                max_index_of_fiber_along_this_path = -1
                                for lin_idx in range(len(links_on_this_path)):
                                    # print("sadegh {}".format(lin_idx))
                                    if lin_idx == 0:
                                        for link in all_links:
                                            if (link.src == links_on_this_path[lin_idx][0]) and\
                                                    (link.dst == links_on_this_path[lin_idx][1]):
                                                # print("hello world")
                                                fiber_on_first_link_is_found = 0
                                                for f_idx in range(len(link.list_of_fibers)):
                                                    first_idx_of_available_channels =\
                                                        link.list_of_fibers[f_idx].find_index_of_first_available_channel()
                                                    # print("hello {}".format(first_idx_of_available_channels))
                                                    if first_idx_of_available_channels != None:
                                                        available_index_of_channel.append(first_idx_of_available_channels)
                                                        fibers_list.append(f_idx)
                                                        # for reverse direction
                                                        fibers_list.append(f_idx)
                                                        fiber_object.append(link.list_of_fibers[f_idx])
                                                        fiber_on_first_link_is_found = 1
                                                        break
                                                    else:
                                                        continue
                                                if fiber_on_first_link_is_found == 0:
                                                    fibers_list.append(
                                                        1000)  #this is a large number to prevent selecting unavailable path
                                                break
                                        for link in all_links:
                                            if (link.src == links_on_this_path[lin_idx][1]) and \
                                                    (link.dst == links_on_this_path[lin_idx][0]):
                                                fiber_object.append(link.list_of_fibers[f_idx])
                                                break
                                    else:
                                        for link in all_links:
                                            if (link.src == links_on_this_path[lin_idx][0]) and\
                                                    (link.dst == links_on_this_path[lin_idx][1]):
                                                fiber_index = find_index_of_available_fiber_for_specific_channel(
                                                    link.list_of_fibers, available_index_of_channel[-1])
                                                if fiber_index != None:
                                                    fibers_list.append(fiber_index)
                                                    # for reverse direction
                                                    fibers_list.append(fiber_index)
                                                    fiber_object.append(link.list_of_fibers[fiber_index])
                                                else:
                                                    print("{}th part of Demand {} can not be routed on path {}".format(
                                                        list_of_lightpath.index(lightpath), [dem.src, dem.dst], path))
                                                    fibers_list.append(1000)
                                                break
                                            ### do same for reverse links ###
                                        for link in all_links:
                                            if (link.src == links_on_this_path[lin_idx][1]) and \
                                                    (link.dst == links_on_this_path[lin_idx][0]):
                                                fiber_object.append(link.list_of_fibers[fiber_index])
                                                break


                                available_fibers_on_each_link_of_the_path.append(fibers_list)
                                fiber_object_for_each_link_of_the_path.append(fiber_object)

                            index_of_selected_path_for_this_lightpath = available_fibers_on_each_link_of_the_path.index(
                                min(available_fibers_on_each_link_of_the_path, key=lambda x: max(x)))
                            selected_channel_for_this_lightpath = available_index_of_channel[
                                index_of_selected_path_for_this_lightpath]
                            for fb in fiber_object_for_each_link_of_the_path[index_of_selected_path_for_this_lightpath]:
                                fb.fill_channel(selected_channel_for_this_lightpath)

                            lightpath.assign_selected_path(index_of_selected_path_for_this_lightpath)
                            lightpath.assign_channel_idx(selected_channel_for_this_lightpath)
                            lightpath.dedicated_transponder.set_assigned_path_for_reference_demand(
                                dem.get_three_shortest_paths()[index_of_selected_path_for_this_lightpath])
                            lightpath.dedicated_transponder.set_fibers_idx_on_each_link(
                                available_fibers_on_each_link_of_the_path[index_of_selected_path_for_this_lightpath])
                            list_of_maximum_used_fiber_index_for_each_period.append(
                                available_fibers_on_each_link_of_the_path[index_of_selected_path_for_this_lightpath])
                            links_for_the_selected_path = get_all_links_of_path(
                                dem.get_three_shortest_paths()[index_of_selected_path_for_this_lightpath], reverse=True)
                            for lll_idx in range(len(links_for_the_selected_path)):
                                for obj_link in all_links:
                                    if (obj_link.src == links_for_the_selected_path[lll_idx][0]) and\
                                            (obj_link.dst == links_for_the_selected_path[lll_idx][1]):
                                        obj_link.state_of_fibers[available_fibers_on_each_link_of_the_path[
                                            index_of_selected_path_for_this_lightpath][lll_idx]] = 1
                            if True:
                                print("Demand {} with rate {} Gbps in time period {} uses {} Gbps of {}th transponders on path {} with this fiber indices {}".format(
                                [dem.src, dem.dst], dem.n_25G_traffic * 25, period_idx,
                                100 * sum(lightpath.dedicated_transponder.state_of_slots)/4,
                                selected_channel_for_this_lightpath,
                                dem.get_three_shortest_paths()[index_of_selected_path_for_this_lightpath],
                                available_fibers_on_each_link_of_the_path[index_of_selected_path_for_this_lightpath]))

                    else:
                        ### This part of code tries to serve new demands in previous assigned transponders ###
                        difference_between_required_traffic = dem.n_25G_traffic - [ddd.n_25G_traffic for ddd in
                                                                                   time_periods[
                                                                                       period_idx - 1].traffic_matrix if (
                                                                            ddd.src == dem.src and ddd.dst == dem.dst)][0]
                        if True:
                            print("for demand {}: this is difference {}".format([dem.src, dem.dst], difference_between_required_traffic))
                        if difference_between_required_traffic == 0:
                            continue
                        elif difference_between_required_traffic < 0:
                            print("demand between {} with traffic {} Gbps uses required lower traffic than previous period".format([dem.src, dem.dst], dem.n_25G_traffic * 25))
                            number_of_released_transponder_slots = 0
                            flag = 0  # to break this "For" statement and avoid releasing extra transponder slots
                            lightpaths_to_be_completely_removed = []
                            for my_lp_idx in reversed(range(len(list_of_all_used_lightpaths))):
                                if list_of_all_used_lightpaths[my_lp_idx].basic_demand.src != dem.src or list_of_all_used_lightpaths[my_lp_idx].basic_demand.dst != dem.dst:
                                    continue

                                for _ in range(sum(list_of_all_used_lightpaths[my_lp_idx].dedicated_transponder.state_of_slots)):
                                    list_of_all_used_lightpaths[my_lp_idx].release_transponders_slot()
                                    number_of_released_transponder_slots += 1
                                    if number_of_released_transponder_slots == np.abs(difference_between_required_traffic):
                                        flag = 1
                                        break

                                # difference_between_required_traffic = np.abs(difference_between_required_traffic) - number_of_released_transponder_slots
                                if list_of_all_used_lightpaths[my_lp_idx].is_completely_empty():
                                    lightpaths_to_be_completely_removed.append(list_of_all_used_lightpaths[my_lp_idx])
                                    print("hello world")

                                if flag == 1:
                                    break

                            for llpp in lightpaths_to_be_completely_removed:
                                list_of_all_used_lightpaths.remove(llpp)

                        else:
                            dem.number_of_served_slots = [ddd.n_25G_traffic for ddd in
                                                                                   time_periods[
                                                                                       period_idx - 1].traffic_matrix if (
                                                                            ddd.src == dem.src and ddd.dst == dem.dst)][0]
                            if perform_grooming == 1:
                                for used_lp in list_of_all_used_lightpaths:
                                    if (used_lp.basic_demand.src == dem.src) and (used_lp.basic_demand.dst == dem.dst):
                                        remained_cap = used_lp.dedicated_transponder.get_remain_capacity()
                                        if remained_cap:
                                            for sl in range(remained_cap):
                                                if dem.number_of_served_slots < dem.n_25G_traffic:
                                                    dem.number_of_served_slots += 1
                                                    used_lp.dedicated_transponder.fill_slots(1, period_idx)
                                            if True:
                                                print(
                                                "{} Gbps of Demand between {} with total traffic {} in time period {} is groomed with previously used transponder initiated at fiber {} of link {}".format(
                                                    (remained_cap - used_lp.dedicated_transponder.get_remain_capacity()) * 25,
                                                    [dem.src, dem.dst], dem.n_25G_traffic * 25, period_idx,
                                                    used_lp.dedicated_transponder.reference_fiber[0],
                                                    used_lp.dedicated_transponder.reference_link[0]))


                            n_lightpaths = int(np.ceil((dem.n_25G_traffic - dem.number_of_served_slots)/4))
                            list_of_lightpath = []
                            for _ in range(n_lightpaths):
                                trans = None
                                for tt in list_of_all_used_transponders:
                                    if tt.transponder_is_released_after_usage and tt.transponder_is_active == 0:
                                        if tt.reference_node == dem.src:
                                            trans = tt
                                            trans.reset_assigned_demand(dem)
                                            trans.activate_transponder()
                                            break
                                if trans == None:
                                    trans = Transponder(dem, all_links)
                                    list_of_all_used_transponders.append(trans)
                                list_of_lightpath.append(Lightpath(dem, trans))


                            for lightpath in list_of_lightpath:
                                list_of_all_used_lightpaths.append(lightpath)
                                # print("this is for lp {}".format(lightpath))
                                served_slots = dem.number_of_served_slots
                                for i in range(dem.n_25G_traffic - served_slots):
                                    if i > 3:
                                        break
                                    else:
                                        dem.number_of_served_slots += 1
                                        lightpath.dedicated_transponder.fill_slots(1, period_idx)

                                index_of_last_used_fiber_for_each_path = []
                                available_index_of_channel = []
                                available_fibers_on_each_link_of_the_path = []
                                fiber_object_for_each_link_of_the_path = []
                                # print(dem.get_three_shortest_paths())
                                for path_idx in range(len(dem.get_three_shortest_paths())):
                                    path = dem.get_three_shortest_paths()[path_idx]
                                    # print("hello {} and index {}".format(path, path_idx))
                                    fibers_list = []
                                    fiber_object = []
                                    links_on_this_path = get_all_links_of_path(path)
                                    max_index_of_fiber_along_this_path = -1
                                    for lin_idx in range(len(links_on_this_path)):
                                        # print("sadegh {}".format(lin_idx))
                                        if lin_idx == 0:
                                            for link in all_links:
                                                if (link.src == links_on_this_path[lin_idx][0]) and\
                                                        (link.dst == links_on_this_path[lin_idx][1]):
                                                    # print("hello world")
                                                    fiber_on_first_link_is_found = 0
                                                    for f_idx in range(len(link.list_of_fibers)):
                                                        first_idx_of_available_channels = \
                                                            link.list_of_fibers[f_idx].find_index_of_first_available_channel()
                                                        # print("hello {}".format(first_idx_of_available_channels))
                                                        # print("$$%#%#$%#$@##$$%# available channel = {} %$$$%^$^@###$%5$$^#".format(first_idx_of_available_channels))
                                                        if first_idx_of_available_channels != None:
                                                            available_index_of_channel.append(first_idx_of_available_channels)
                                                            fibers_list.append(f_idx)
                                                            # for reverse direction
                                                            fibers_list.append(f_idx)
                                                            fiber_object.append(link.list_of_fibers[f_idx])
                                                            fiber_on_first_link_is_found = 1
                                                            break
                                                        else:
                                                            continue
                                                    if fiber_on_first_link_is_found == 0:
                                                        fibers_list.append(
                                                            1000)  #this is a large number to prevent selecting unavailable path
                                                    break

                                            for link in all_links:
                                                if (link.src == links_on_this_path[lin_idx][1]) and \
                                                        (link.dst == links_on_this_path[lin_idx][0]):
                                                    fiber_object.append(link.list_of_fibers[f_idx])
                                                    break
                                        else:
                                            for link in all_links:
                                                if (link.src == links_on_this_path[lin_idx][0]) and\
                                                        (link.dst == links_on_this_path[lin_idx][1]):
                                                    fiber_index = find_index_of_available_fiber_for_specific_channel(
                                                        link.list_of_fibers, available_index_of_channel[-1])
                                                    if fiber_index != None:
                                                        fibers_list.append(fiber_index)
                                                        # for reverse direction
                                                        fibers_list.append(fiber_index)
                                                        fiber_object.append(link.list_of_fibers[fiber_index])
                                                    else:
                                                        print("{}th part of Demand {} can not be routed on path {}".format(
                                                            list_of_lightpath.index(lightpath), [dem.src, dem.dst], path))
                                                        fibers_list.append(1000)
                                                    break

                                                ### do same for reverse links ###
                                            for link in all_links:
                                                if (link.src == links_on_this_path[lin_idx][1]) and \
                                                        (link.dst == links_on_this_path[lin_idx][0]):
                                                    fiber_object.append(link.list_of_fibers[fiber_index])
                                                    break

                                    available_fibers_on_each_link_of_the_path.append(fibers_list)
                                    fiber_object_for_each_link_of_the_path.append(fiber_object)

                                index_of_selected_path_for_this_lightpath = available_fibers_on_each_link_of_the_path.index(
                                    min(available_fibers_on_each_link_of_the_path, key=lambda x: max(x)))
                                selected_channel_for_this_lightpath = available_index_of_channel[
                                    index_of_selected_path_for_this_lightpath]
                                for fb in fiber_object_for_each_link_of_the_path[index_of_selected_path_for_this_lightpath]:
                                    fb.fill_channel(selected_channel_for_this_lightpath)

                                lightpath.assign_selected_path(index_of_selected_path_for_this_lightpath)
                                lightpath.assign_channel_idx(selected_channel_for_this_lightpath)
                                lightpath.dedicated_transponder.set_assigned_path_for_reference_demand(
                                    dem.get_three_shortest_paths()[index_of_selected_path_for_this_lightpath])
                                lightpath.dedicated_transponder.set_fibers_idx_on_each_link(
                                    available_fibers_on_each_link_of_the_path[index_of_selected_path_for_this_lightpath])
                                list_of_maximum_used_fiber_index_for_each_period.append(
                                    available_fibers_on_each_link_of_the_path[index_of_selected_path_for_this_lightpath])
                                links_for_the_selected_path = get_all_links_of_path(
                                    dem.get_three_shortest_paths()[index_of_selected_path_for_this_lightpath], reverse=True)
                                for lll_idx in range(len(links_for_the_selected_path)):
                                    for obj_link in all_links:
                                        if (obj_link.src == links_for_the_selected_path[lll_idx][0]) and\
                                                (obj_link.dst == links_for_the_selected_path[lll_idx][1]):
                                            obj_link.state_of_fibers[available_fibers_on_each_link_of_the_path[
                                                index_of_selected_path_for_this_lightpath][lll_idx]] = 1
                                if True:
                                    print("Demand between {} with rate {} Gbps in time period {} uses {} Gbps of {}th transponders on path {} with this fiber indices {}".format(
                                    [dem.src, dem.dst], dem.n_25G_traffic * 25, period_idx,
                                    100 * sum(lightpath.dedicated_transponder.state_of_slots)/4, selected_channel_for_this_lightpath,
                                    dem.get_three_shortest_paths()[index_of_selected_path_for_this_lightpath],
                                    available_fibers_on_each_link_of_the_path[index_of_selected_path_for_this_lightpath]))

                    # show table of the state of different transponders:
                table = [['Index', 'Installed in Node', 'Source', 'Destination', 'Used Capacity [Gbps]', 'Residual Capacity [Gbps]', 'Path', 'Channel']]
                used_src_dst_for_transponders = []
                n_activate_transponders = 0
                for my_tr_idx in range(len(list_of_all_used_transponders)):
                    my_tr = list_of_all_used_transponders[my_tr_idx]
                    if my_tr.transponder_is_active:
                        if [my_tr.assigned_demand.dst, my_tr.assigned_demand.src] in used_src_dst_for_transponders:
                            continue
                        else:
                            n_activate_transponders += 1
                            tab_content = [n_activate_transponders, my_tr.assigned_demand.src, my_tr.assigned_demand.src, my_tr.assigned_demand.dst, 25 * sum(my_tr.state_of_slots),
                                       25 * my_tr.get_remain_capacity(), my_tr.assigned_path_for_demand,
                                           #my_tr.reference_fiber,
                                       my_tr.assigned_channel_idx]
                            table.append(tab_content)

                            n_activate_transponders += 1
                            tab_content = [n_activate_transponders, my_tr.assigned_demand.dst, my_tr.assigned_demand.src, my_tr.assigned_demand.dst,
                                           25 * sum(my_tr.state_of_slots),
                                           25 * my_tr.get_remain_capacity(), my_tr.assigned_path_for_demand,
                                           #my_tr.reference_fiber,
                                           my_tr.assigned_channel_idx]
                            table.append(tab_content)
                            used_src_dst_for_transponders.append([my_tr.assigned_demand.src, my_tr.assigned_demand.dst])

                print(tabulate(table, headers='firstrow', tablefmt='fancy_grid'))

    # print("Maximum index of used fiber is {}".format(max(list_of_maximum_used_fiber_index_for_each_period)))
    total_number_of_fibers_in_networks = 0
    for my_object_link_idx in range(len(all_links)):
        my_object_link = all_links[my_object_link_idx]
        if my_object_link_idx % 2 == 0:
            total_number_of_fibers_in_networks += sum(my_object_link.state_of_fibers)
            print("For link {} we need to locate {} fiber pairs".format([my_object_link.src, my_object_link.dst],
                                                               max([sum(my_object_link.state_of_fibers),
                                                                    sum(all_links[my_object_link_idx].state_of_fibers)])))

    print("************ total number of required fiber pairs in the network is {}".format(total_number_of_fibers_in_networks))

    number_of_transponders_for_each_node = [len(
                        [tr for tr in list_of_all_used_transponders if (tr.reference_node == node or tr.other_terminal_node == node)]) for node in all_nodes]

    ####### The END ########
