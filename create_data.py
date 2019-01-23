
from world import *

def main():
    world = World()

    # 2 cases this breaks, either t > max duration, or 19 tasks are scheduled.
    while not world.data_done_generating:
        is_feasible = world.update_floyd_warshall_and_all_vectors()
        world.print_all_features()
        if is_feasible == False:
            print('constraint was infeasible')
            break
        for counter, agent in enumerate(world.agents):
            world.compute_task_to_schedule(counter)
            if world.pairwise:
                world.write_csv_pairwise(counter)
            else:
                world.write_csv_pairwise(counter)
                world.write_csv(counter)
            is_feasible = world.update_floyd_warshall_and_all_vectors()

        world.add_constraints_based_on_task()
        world.check_if_schedule_finished()

        # Next Time Step
        world.t += 1
        print('current time is ',world.t)
        world.update_based_on_time()


if __name__ == main():
    main()
