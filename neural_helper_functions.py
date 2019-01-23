import numpy as np
import pickle
import os
import csv
import ast
import torch


def altered_get_Data(directory, DEBUG=False):
    total_data = []
    labels = []
    with open(directory) as csvfile:
        data = (csv.reader(csvfile))
        for row in data:
            total_data.append(row)
    X = np.ones((0))
    schedule_array = []
    schedule = [0]
    zero_counter = 0
    for counter, i in enumerate(total_data):  # for all row of the csv file
        new_data = np.ones((0))
        # check first column (time)

        time = ast.literal_eval(i[0])
        if time == 0:
            if counter != 0:
                zero_counter += 1
        if zero_counter == 2:
            schedule.append(counter - 1)
            schedule_array.append(schedule)
            schedule = [counter]
            zero_counter = 0
        if counter == len(total_data) - 1:  # last one
            schedule.append(counter)
            schedule_array.append(schedule)

        for j in range(5, 75):
            if j == 74:
                label = ast.literal_eval(i[j])
                labels.append(label)
            else:
                data_to_concat = ast.literal_eval(i[j])
                data_to_concat_as_numpy = np.asarray(data_to_concat)
                try:
                    new_data = np.concatenate([new_data, data_to_concat_as_numpy], axis=0)
                except ValueError:
                    data_to_concat_as_numpy = data_to_concat_as_numpy.reshape(1, )
                    new_data = np.concatenate([new_data, data_to_concat_as_numpy], axis=0)
        if DEBUG:
            print('percentage done: ', counter / len(total_data))

        # new_data should continously grow

        if counter == 0:
            X = new_data
        else:
            X = np.vstack((X, new_data))  # this will to be a t by 194 vector
    return X, labels, schedule_array


def get_embedding_num(schedule_array, sample_val):
    for i, each_array in enumerate(schedule_array):
        if sample_val >= each_array[0] and sample_val <= each_array[1]:
            return i
        else:
            continue


def load_in_embedding(NeuralNet, embedding_list, player_id):
    curr_embedding = embedding_list[player_id]
    curr_dict = NeuralNet.state_dict()
    curr_dict['EmbeddingList.0.embedding'] = curr_embedding
    NeuralNet.load_state_dict(curr_dict)


def store_embedding_back(NeuralNet, embedding_list, player_id, DEBUG=False):
    curr_dict = NeuralNet.state_dict()
    new_embedding = curr_dict['EmbeddingList.0.embedding']
    curr_embedding = embedding_list[player_id]
    if DEBUG:
        print(curr_embedding)
        print(new_embedding)
    embedding_list[player_id] = new_embedding


def scalar_into_one_hot_encoding_size_21(scalar):
    """
    turns a scalar into a onehot encoding for label comparision
    :param scalar: truth value for schedule
    :return: vector of size 1 x 21
    """
    output = np.zeros((1, 21))
    output[0][scalar] = 1  # task 20 is null
    return output

def scalar_into_one_hot_encoding_size_20(scalar):
    """
    turns a scalar into a onehot encoding for label comparision
    :param scalar: truth value for schedule
    :return: vector of size 1 x 21
    """
    output = np.zeros((1, 20))
    output[0][scalar] = 1  # task 20 is null
    return output



def save_pickle(file_location, file, special_string, want_to_print=False):
    pickle.dump(file, open(os.path.join(file_location, special_string)
                           , 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    if want_to_print:
        print("Dumped", file, "into ", file_location, "safely!")


def get_save_path(num_schedules, which_type='baseline'):
    return '/home/rohanpaleja/PycharmProjects/scheduling_environment/' + str(which_type) + '/state_dicts/num_sched' \
           + str(num_schedules) + '_state_dict.tar'

def get_save_only_task_path(num_schedules, which_type='baseline'):
    return '/home/rohanpaleja/PycharmProjects/scheduling_environment/' + str(which_type) + '/state_dicts/num_sched' \
           + str(num_schedules) + 'only_task_state_dict.tar'

def get_save_path_cluster(num_schedules, cluster_num, which_type='baseline'):
    return '/home/rohanpaleja/PycharmProjects/scheduling_environment/' + str(which_type) + '/state_dicts/' + 'num_sched' + str(num_schedules) + 'cluster' + str(cluster_num) + '_state_dict.tar'

def get_save_only_task_path_cluster(num_schedules, cluster_num, which_type='baseline'):
    return '/home/rohanpaleja/PycharmProjects/scheduling_environment/' + str(which_type) + '/state_dicts/' + 'num_sched' + str(num_schedules) + 'cluster' + str(cluster_num) + 'only_task_state_dict.tar'


def add_noise(vect, noise_percentage=0):
    choice = np.random.choice(2, p=[noise_percentage * .01, 1 - noise_percentage * .01])  # zero: no noise, one: add noise
    if choice == 0:
        noisy_vector = np.zeros((1, 21))
        which_element = np.random.choice(21)
        noisy_vector[0][which_element] = 1
        print('added noise')
        return noisy_vector
    else:
        return vect


def print_network_loss(NN, iteration, loss_variable):
    print(NN, " loss for iteration : ", iteration, " is ", loss_variable.item())


def print_correct_null_actual(epoch, num_iterations, running_number_of_correct_actual, running_number_of_correct_null):
    print(str(epoch) + ': percentage of correct null: ', running_number_of_correct_null / num_iterations)
    print(str(epoch) + ': percentage of correct actual: ', running_number_of_correct_actual / num_iterations)
    print('----------------------')


def one_hot_encoding_of_cluster(cluster_num):
    hot = np.zeros((3))
    hot[cluster_num] = 1
    return hot


def get_Data(directory, DEBUG=False):
    total_data = []
    labels = []
    with open(directory) as csvfile:
        data = (csv.reader(csvfile))
        for row in data:
            total_data.append(row)
    X = np.ones((0))
    zero_counter = 0
    schedule = [0]
    schedule_array = []
    for counter, i in enumerate(total_data):  # for all row of the csv file
        new_data = np.ones((0))

        time = ast.literal_eval(i[0])
        if time == 0:
            if counter != 0:
                zero_counter += 1
        if zero_counter == 2:
            schedule.append(counter - 1)
            schedule_array.append(schedule)
            schedule = [counter]
            zero_counter = 0
        if counter == len(total_data) - 1:  # last one
            schedule.append(counter)
            schedule_array.append(schedule)


        for j in range(5, 113):
            if j == 112:
                label = ast.literal_eval(i[j])
                labels.append(label)
            else:
                data_to_concat = ast.literal_eval(i[j])
                data_to_concat_as_numpy = np.asarray(data_to_concat)
                try:
                    new_data = np.concatenate([new_data, data_to_concat_as_numpy], axis=0)
                except ValueError:
                    data_to_concat_as_numpy = data_to_concat_as_numpy.reshape(1, )
                    new_data = np.concatenate([new_data, data_to_concat_as_numpy], axis=0)
        if DEBUG:
            print('percentage done: ', counter / len(total_data))

        # new_data should continously grow

        if counter == 0:
            X = new_data
        else:
            X = np.vstack((X, new_data))  # this will to be a t by 194 vector
    return X, labels, schedule_array


###################### PAIRWISE ####################################

def load_data_pairwise(directory, DEBUG=False):
    total_data = []
    labels = []
    with open(directory) as csvfile:
        data = (csv.reader(csvfile))
        for row in data:
            total_data.append(row)
    X = np.ones((0))
    schedule_array = []
    schedule = [0]
    zero_counter = 0
    for counter, i in enumerate(total_data):  # for all row of the csv file
        new_data = np.ones((0))
        # check first column (time)

        time = ast.literal_eval(i[0])
        if time == 0:
            if counter != 0:
                zero_counter += 1
        if zero_counter == 40:
            schedule.append(counter - 1)
            schedule_array.append(schedule)
            schedule = [counter]
            zero_counter = 0
        if counter == len(total_data) - 1:  # last one
            schedule.append(counter)
            schedule_array.append(schedule)

        for j in range(5, 19):
            if j == 18:
                label = ast.literal_eval(i[j])
                labels.append(label)
            elif j == 17:
                continue
            else:
                data_to_concat = ast.literal_eval(i[j])
                data_to_concat_as_numpy = np.asarray(data_to_concat)
                try:
                    new_data = np.concatenate([new_data, data_to_concat_as_numpy], axis=0)
                except ValueError:
                    data_to_concat_as_numpy = data_to_concat_as_numpy.reshape(1, )
                    new_data = np.concatenate([new_data, data_to_concat_as_numpy], axis=0)
        if DEBUG:
            print('percentage done: ', counter / len(total_data))

        # new_data should continously grow

        if counter == 0:
            X = new_data
        else:
            X = np.vstack((X, new_data))

    each_time_agent_schedule_array = []
    loaded_all_schedules = False
    while not loaded_all_schedules:
        each_time_schedule = []
        if len(each_time_agent_schedule_array) == 0:
            each_time_schedule.append(0)
            each_time_schedule.append(19)
            each_time_agent_schedule_array.append(each_time_schedule)
        else:
            last_ele = each_time_agent_schedule_array[-1]
            second_ele_of_last_ele = last_ele[1]
            each_time_schedule.append(second_ele_of_last_ele + 1)
            each_time_schedule.append(second_ele_of_last_ele + 20)
            each_time_agent_schedule_array.append(each_time_schedule)
        # finish condition
        last_ele = each_time_agent_schedule_array[-1]
        second_ele_of_last_ele = last_ele[1]
        if second_ele_of_last_ele > len(X):
            each_time_agent_schedule_array.remove(last_ele)
            loaded_all_schedules = True

    return X, labels, schedule_array, each_time_agent_schedule_array


def find_set_of_twenty(sample_val, schedule_array):
    for i, each_array in enumerate(schedule_array):
        if sample_val >= each_array[0] and sample_val <= each_array[1]:
            return each_array
        else:
            continue


def get_pairwise_save_path(num_schedules, type_nn,  which_type='baseline'):
    return '/home/rohanpaleja/PycharmProjects/scheduling_environment/pairwise/' + str(which_type) + '/state_dicts/' \
    + 'num_sched' + str(num_schedules) + str(type_nn) + '_state_dict.tar'


def add_noise_pairwise(label, noise_percentage=0):
    if label == 1:
        choice = np.random.choice(2, p=[noise_percentage * .01, 1 - noise_percentage * .01])
        return choice
    elif label == 0:
        choice = np.random.choice(2, p=[1 - noise_percentage * .01, noise_percentage * .01])
        return choice
    else:
        return label


def find_closest_cluster(centroids, feature_input):
    dist1 = np.linalg.norm(centroids[0], feature_input)
    dist2 = np.linalg.norm(centroids[1], feature_input)
    dist3 = np.linalg.norm(centroids[2], feature_input)
    dist = [dist1, dist2, dist3]
    return np.argmax(dist)

def load_in_all_parameters(load_file_dest, NN_being_edited, dont_load_embedding = False):
    pre_train_NN = torch.load(load_file_dest)
    NN_dict = NN_being_edited.state_dict()
    dict_copy = pre_train_NN.copy()
    if dont_load_embedding:
        for name, param in pre_train_NN.items():
            if 'Embedding' in name:
                del dict_copy[name]
    NN_dict.update(dict_copy)
    NN_being_edited.load_state_dict(NN_dict)
