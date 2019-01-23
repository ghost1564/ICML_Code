import numpy as np
from task import Task



def randomly_initialize_tasks(m):
    tasks = []
    for i in range(0,m):
        tasks.append(Task(name='task'+str(i+1)))
    return tasks


def euclid_dist(location1, location2):
    return np.sqrt((location1[0] - location2[0]) ** 2 + (location1[1] - location2[1]) ** 2)


def compute_angle_in_rad(location1, location2):
    return np.arctan2(location1[0] - location2[0],location1[1] - location2[1])


def compute_dist(location1, location2):
    alpha0 = 1
    alpha1 = 0
    alpha2 = 0
    norm = euclid_dist(location1, location2)
    angle = compute_angle_in_rad(location1, location2)
    return alpha0 * norm + alpha1 * angle + alpha2 * norm * angle


def find_nearest_unoccupied_task(cur_agent, tasks, agents):
    current_location = cur_agent.getz()
    closest_task_distance = np.inf
    allowable_distance_to_task = .1
    closest_task = None
    for task in tasks:
        location_occupied = False
        if task.isTaskScheduled == False:
            task_loc=task.getloc()
            # check if task is occupied
            for agent in agents: # check if any agent is at the task
                if cur_agent == agent: # don't check yourself, cuz that's fine
                    continue
                if compute_dist(agent.getz(), task_loc) < allowable_distance_to_task:
                    location_occupied = True
            if location_occupied == True: # Now you know there is an agent too near to that task, thus, look at next task
                continue

            dist = euclid_dist(task_loc,current_location)
            if dist < closest_task_distance:
                closest_task_distance = dist
                closest_task = task
        else:
            continue
    return closest_task


def compute_start_and_finish_times(a, n_t, current_time):
    duration = n_t.getc()
    speed = a.getv()
    current_location = a.getz()
    task_loc = n_t.getloc()
    dist = np.sqrt((task_loc[0] - current_location[0]) ** 2 + (task_loc[1] - current_location[1]) ** 2)
    travel_time = dist/speed
    start_time = current_time + travel_time
    finish_time = start_time + duration
    return (start_time,finish_time)


def tasks_are_available(tasks):
    task_not_finished_not_scheduled_count = len(tasks)
    for task in tasks:
        if task.getisTaskFinished() == True:
            continue
        if task.getisTaskScheduled() == True:
            continue
        else:
            task_not_finished_not_scheduled_count -= 1
            # TODO: preety sure, you can just return true here
    if task_not_finished_not_scheduled_count < len(tasks):
        return True
    else:
        return False




