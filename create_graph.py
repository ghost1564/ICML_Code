from agent import *
from timeline import Timeline
from graph import *

from helper_functions import *


# TODO: Create some test cases to see what works

def main():
    agent1 = Agent(1, (1, 1), name='agent1')

    task1 = Task(4, (4, 3), name='task1')
    task2 = Task(1, (4, 5), name='task2')
    task3 = Task(10, (2, 4), name='task3')
    tasks = [task1, task2, task3]

    # tasks = randomly_initialize_tasks(5)
    graph = Graph() # num tasks + 1
    # Manually define constraints
    graph.add_vertex('start') # start
    graph.add_task_vertex_and_edges(task1)
    graph.add_task_vertex_and_edges(task2)
    graph.add_task_vertex_and_edges(task3)
    # graph.add_tasks_vertex_and_edges(tasks)
    graph.add_vertex('end')

    # Other Constraints
    graph.add_edge_by_name('start','end',80)
    graph.initialize_all_start_and_end_nodes(tasks)
    graph.get_random_wait_constraints(tasks)
    graph.get_random_deadline_constraints(tasks)

    graph.print_checking()

    graph.build_M_matrix()
    graph.compute_floyd_warshal()
    print(graph.is_feasible())

    # TODO: randomly initialize constraints and see if it works


if __name__ == main():
    main()
