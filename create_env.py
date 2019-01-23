from agent import Agent
from timeline import Timeline


from helper_functions import *


# TODO: Create some test cases to see what works

def main():
    agent1 = Agent(1, (1, 1), name='agent1')
    agent2 = Agent(2, (2, 2), name='agent2')

    task1 = Task(3, (4, 3), name='task1')
    task2 = Task(5, (4, 5), name='task2')
    task3 = Task(10, (2, 4), name='task3')
    task4 = Task(4, (6, 4), name='task4')
    task5 = Task(12, (3, 4), name='task5')


    timeline = Timeline()

    t = 0
    agents = [agent1, agent2]
    tasks = [task1, task2, task3, task4, task5]
    # tasks = randomly_initialize_tasks(5)
    while tasks_are_available(tasks):
        # check for completion statuses
        for agent in agents:
            if t >= agent.getFinishTime():
                task = agent.getCurrTask()
                task.changeTaskCompletionStatus() # mark task completed
                agent.changebusy(False) # make agent free again

        for agent in agents:
            print(agent.getName(), 'is busy: ', agent.getisBusy()) # check if agent is busy
            if agent.getisBusy() == False:
                nearest_task = find_nearest_unoccupied_task(agent, tasks, agents)  # find nearest task
                nearest_task.changeTaskScheduleStatus() # change isTaskScheduled to True here
                agent.changebusy(True) # agent is now busy
                (start_time, finish_time) = compute_start_and_finish_times(agent, nearest_task, t)
                agent.updateAgentLocation(nearest_task.getloc())
                timeline.add_to_timeline(start_time,finish_time,agent, nearest_task)
                agent.setFinishTime(finish_time)
                agent.setCurrTask(nearest_task)
        t += .1
        print('time is ', t)

    print(timeline.sort_timeline())
    print(timeline.list_elements)
    # timeline.create_gant_chart()
    timeline.plot_path(agents)

    print('finished')


if __name__ == "__main__":
    main()
