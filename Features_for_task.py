import numpy as np
import random
import csv

class TaskFeatures:
    def __init__(self):
        self.num_tasks = 20
        self.max_deadline_time = 15
        self.grid_world_size = 15
        self.high_prob = .9
        self.low_prob = 1 - self.high_prob
        self.task_list = list(range(1, self.num_tasks + 1))  # the plus 1 is cuz its exclusive

        self.agentIsIdle = np.random.choice(2, p=[self.low_prob, self.high_prob]) # is agent Idle? 1 if it is, 0 if not
        self.isTaskFeasible = np.random.choice(2, self.num_tasks, p=[self.low_prob, self.high_prob])  # can actually be found by (f_i + dij/s <= t)
        self.grid_world = np.zeros((1, self.grid_world_size)) # 5 by 3 grid world
        self.fill_grid_world()
        self.is_there_an_agent_at_location = np.random.choice(2, self.grid_world_size, p=[self.high_prob, self.low_prob])
        self.deadline = np.random.randint(self.max_deadline_time, size=self.num_tasks) + np.random.normal(0, .1, size=self.num_tasks) # when is the deadline
        self.is_task_alive = np.random.choice(2, self.num_tasks, p=[self.low_prob,self.high_prob])   # boolean if each can be scheduled
        self.is_task_enabled = np.random.choice(2, self.num_tasks, p=[.8,.2])  # boolean if each task is scheduled
        self.is_task_finished = np.random.choice(2, self.num_tasks, p=[self.high_prob,self.low_prob])  # 1 is finished, 0 is unfinished
        # self.isConstraintsSatisfied = np.random.choice(2,self.num_tasks,p=[.8,.2])
        self.distance_from_agent_to_task = np.random.choice(20, size=self.num_tasks)
        self.orientation = np.random.uniform(0, np.pi, size=self.num_tasks)

        self.init_hyperparameters()




    def init_hyperparameters(self):
        self.alpha = .8
        self.alpha2 = 1
        self.alpha3 = .6

    def fill_grid_world(self):
        for i in range(0, self.num_tasks):
            element_to_add_to = np.random.randint(0,self.grid_world.shape[1])
            self.grid_world[0][element_to_add_to] += 1
        self.resource_locations= [[] for x in range(self.grid_world.shape[1])]
            # put tasks randomly into bins based on the numbers previously
        num_list = list(range(1, self.num_tasks + 1))  # the plus 1 is cuz its exclusive

        for c, element in enumerate(self.grid_world[0]):
            l = 0
            while l < element:
                rand_task = random.choice(num_list)
                l += 1
                num_list.remove(rand_task)
                self.resource_locations[c].append(rand_task)


    def output_task(self, heuristic_num):
        if self.agentIsIdle == 0:
            self.optimal_task = 0
        else:
            self.find_subset_of_feasible_tasks()
            optimal_task_number = self.heuristic(heuristic_num)
            self.optimal_task = self.feasible_task_list[optimal_task_number]

    def heuristic(self, heuristic_num):
        if heuristic_num == 1: # earliest deadline first
            feasibility_deadlines = self.cut_deadline_by_feasible()
            print(feasibility_deadlines)
            return np.argmin(feasibility_deadlines)
            # TODO: Add method to figure out what to do if there is ties

        if heuristic_num == 2: # rule to mitigate resource contention
            task_numbers_at_busy_areas = self.get_tasks_from_busy_areas()  # all these have highest busy area
            # we want earliest deadline after this
            minimum_deadline = np.inf
            minimum_deadline_task = 0
            for i, each_task in enumerate(task_numbers_at_busy_areas):
                if self.deadline[each_task-1] < minimum_deadline:
                    minimum_deadline = self.deadline[each_task-1]
                    minimum_deadline_task = each_task
            optimal = self.feasible_task_list.index(minimum_deadline_task)
            return optimal

        if heuristic_num == 3:
            combo = self.cut_distance_from_agent_to_task() + self.alpha * self.cut_orientation() + \
                    self.alpha2 * self.cut_orientation() * self.cut_distance_from_agent_to_task()
            print(combo)
            return np.argmin(combo)


    def cut_distance_from_agent_to_task(self):
        new_distance_matrix = np.zeros((1,len(self.feasible_task_list)))
        for i, task in enumerate(self.feasible_task_list):
            new_distance_matrix[0][i] = self.distance_from_agent_to_task[task-1]
        return new_distance_matrix

    def cut_orientation(self):
        new_orientation_matrix = np.zeros((1, len(self.feasible_task_list)))
        for i, task in enumerate(self.feasible_task_list):
            new_orientation_matrix[0][i] = self.orientation[task-1]
        return new_orientation_matrix

    def cut_deadline_by_feasible(self):
        new_deadline_matrix = np.zeros((1, len(self.feasible_task_list)))
        for i, task in enumerate(self.feasible_task_list):
            new_deadline_matrix[0][i] = self.deadline[task-1]
        return new_deadline_matrix

    def get_tasks_from_busy_areas(self):
        length_of_biggest_array = 0
        elements_in_busy_area = []
        for feasible_task in self.feasible_task_list:
            for counter, every_location in enumerate(self.resource_locations):
                for task in every_location:
                    if task == feasible_task:  # location of task is found
                        found_set = every_location
                        if len(found_set) > length_of_biggest_array:
                            elements_in_busy_area = []  # reset elements contained in this list
                            length_of_biggest_array = len(found_set)
                            elements_in_busy_area.append(feasible_task)
                        elif len(found_set) == length_of_biggest_array:
                             elements_in_busy_area.append(feasible_task)
                        else:
                            continue
        print(elements_in_busy_area)
        return elements_in_busy_area

    def find_subset_of_feasible_tasks(self):
        self.feasible_task_list = self.task_list.copy()
        for every_task in self.task_list:
            # Check for exit conditions
            if self.is_task_alive[every_task-1] == 0 or self.isTaskFeasible[every_task-1] == 0 or self.is_task_finished[every_task-1] == 1 or self.is_task_enabled[every_task - 1] == 1:
                self.feasible_task_list.remove(every_task)
        for each_feasible_task in self.feasible_task_list:
            for counter, every_location in enumerate(self.resource_locations):
                for task in every_location:
                    if task == each_feasible_task:  # location of task is found
                        found_set = every_location
                        index_of_found_set = counter
                        for sub_task in found_set:  # recheck the set
                            if self.is_task_enabled[sub_task - 1] == 1:  # if task is enabled, it means agent is there
                                try:
                                    self.feasible_task_list.remove(each_feasible_task)  # remove this task from feasible
                                except ValueError:
                                    pass
                            if self.is_there_an_agent_at_location[index_of_found_set] == 1:  # if agent occupying
                                try:
                                    self.feasible_task_list.remove(each_feasible_task)
                                except ValueError:
                                    pass


    def print_important_elements(self, heuristic_num):
        print('Tasks available are ', self.task_list)
        print('Feasible tasks are ', self.feasible_task_list)
        print('Heuristic chosen was', heuristic_num)
        print('Output was task ', self.optimal_task)

    def save_data(self, heuristic_num, filename='file.txt', ):
        save_dict = {}
        save_dict['isAgentIdle'] = self.agentIsIdle
        save_dict['isTaskFeasible'] = self.isTaskFeasible
        save_dict['task_occupancy_grid'] = self.resource_locations
        save_dict['deadlines'] = self.deadline
        save_dict['isTaskFinished'] = self.is_task_finished
        save_dict['isTaskEnabled'] = self.is_task_enabled
        save_dict['isTaskAlive'] = self.is_task_alive
        save_dict['distance_to_task'] = self.distance_from_agent_to_task
        save_dict['isOtherAgentAtLocation'] = self.is_there_an_agent_at_location
        save_dict['orientation'] = self.orientation
        save_dict['heuristic_num'] = heuristic_num
        save_dict['optimal_task'] = self.optimal_task
        save_dict_str = str(save_dict)
        with open(filename, 'a') as f:
            f.write(save_dict_str)

    def write_csv(self,heuristic_num):
        data = []
        data.append(self.agentIsIdle)
        data.extend(self.isTaskFeasible)
        data.extend(self.resource_locations)
        data.extend(self.deadline)
        data.extend(self.is_task_finished)
        data.extend(self.is_task_enabled)
        data.extend(self.is_task_alive)
        data.extend( self.distance_from_agent_to_task)
        data.extend(self.is_there_an_agent_at_location)
        data.extend(self.orientation)
        data.append(heuristic_num)
        data.append(self.optimal_task)
        with open('distance_and_orientation.csv', 'a') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(data)

