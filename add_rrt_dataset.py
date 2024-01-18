from utilite_cross_dataset import save_dataset
import numpy as np

def erase_tasks(lst_starts, lst_goals, max_tasks=300):
    if len(lst_starts) > max_tasks:
        while len(lst_starts) > max_tasks:
            indx = np.random.choice(range(len(lst_starts)))
            del lst_starts[indx]
            del lst_goals[indx]
        assert len(lst_starts) == len(lst_goals)
    return lst_starts, lst_goals

def get_tasks(name):
    train_lst_starts, train_lst_goals = [], []
    with open("cross_dataset_balanced/" + name, 'r') as file:
        for idx, line in enumerate(file):
            if idx != 0:
                assert line[-1] == "\n"
                task = line.split("\t")[:-1]
                start = task[:5]
                goal = task[5:]
                assert len(start) == len(goal) == 5, f"start: {start}, goal: {goal}"
                train_lst_starts.append(start)
                train_lst_goals.append(goal)

    assert len(train_lst_starts) == len(train_lst_goals)
    return train_lst_starts, train_lst_goals

# form train dataset
name = "train_map0.txt"
old_lst_starts, old_lst_goals = get_tasks(name)
name = "rrt_map_train.txt"
new_lst_starts, new_lst_goals = get_tasks(name)
# delete tasks
new_lst_starts, new_lst_goals = erase_tasks(new_lst_starts, new_lst_goals, max_tasks=300)
old_lst_starts.extend(new_lst_starts)
old_lst_goals.extend(new_lst_goals)
save_dataset(old_lst_starts, old_lst_goals, "train_map0.txt")


# form validate dataset
#name = "rrt_map_val.txt"
#new_lst_starts, new_lst_goals = get_tasks(name)
# delete tasks
#new_lst_starts, new_lst_goals = erase_tasks(new_lst_starts, new_lst_goals, max_tasks=300)
#save_dataset(new_lst_starts, new_lst_goals, "val_map0.txt")
name = "val_map0.txt"
old_lst_starts, old_lst_goals = get_tasks(name)
name = "rrt_map_val.txt"
new_lst_starts, new_lst_goals = get_tasks(name)
# delete tasks
new_lst_starts, new_lst_goals = erase_tasks(new_lst_starts, new_lst_goals, max_tasks=300)
old_lst_starts.extend(new_lst_starts)
old_lst_goals.extend(new_lst_goals)
save_dataset(old_lst_starts, old_lst_goals, "val_map0.txt")
