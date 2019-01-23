from Features_for_task import *



def main():

    heuristic_num = 3

    feature_set = TaskFeatures()
    feature_set.find_subset_of_feasible_tasks()
    feature_set.output_task(heuristic_num)
    feature_set.print_important_elements(heuristic_num)
    feature_set.write_csv(heuristic_num)

if __name__ == '__main__':
    main()
