import numpy as np
from polamp_env.lib.utils_operations import normalizeAngle

def save_dataset(lst_starts, lst_goals, name):
    with open("hard_dataset_simplified_test/" + name, 'w') as output:
        output.write(str(len(lst_starts)) + '\n')
        for i in range(len(lst_starts)):
            # print(f"i : {i}")
            start = lst_starts[i]
            goal = lst_goals[i]
            for element in start:
                output.write(str(element) + '\t')
            for element in goal:
                output.write(str(element) + '\t')
            output.write("\n")

def save_map(obstacle, name):
    with open("hard_dataset_simplified_test/" + name, 'w') as output:
    # for obstacle in obstacles:
        output.write(str(len(obstacle)) + '\n')
        for polygon in obstacle:
            for element in polygon:
                output.write(str(element) + '\t')
            output.write("\n")

# number_of_tasks = 10
def task_generator(number_of_tasks=1, obst_num=1):
    assert obst_num == 1 or obst_num == 2
    lst_starts = []
    lst_goals = []
    num_task_types = 5

    if obst_num == 1:
        # 1 obstacle
        task_1_up = [4.7, 18, 0.2, 0, 0]
        task_1_down = [4.7, 15.8, -0.2, 0, 0]
        task_2_up = [14, 18, 0.2, 0, 0]
        task_2_down = [14, 15.8, -0.2, 0, 0]
        task_10_up = [10.5, 18, 0.2, 0, 0]
        task_10_down = [10.5, 15.8, -0.2, 0, 0]
        task_11_up = [18, 18, 0.2, 0, 0]
        task_11_down = [18, 15.8, -0.2, 0, 0]

        task_3_left = [17, 15.7, -1.8, 0, 0]
        task_3_right = [19, 15.7, -1.3, 0, 0]
        task_4_left = [17, 5.5, -1.8, 0, 0]
        task_4_right = [19, 5.5, -1.3, 0, 0]
        task_6_left = [17, 8, -1.8, 0, 0]
        task_6_right = [19, 8, -1.3, 0, 0]
        task_7_left = [17, 1, -1.8, 0, 0]
        task_7_right = [19, 1, -1.3, 0, 0]

        task_12_up = [17, 2, 2.8, 0, 0]
        task_12_down = [17, 0, -2.8, 0, 0]
        task_13_up = [9, 2, 2.8, 0, 0]
        task_13_down = [8, 0, -2.8, 0, 0]
        task_5_up = [12, 2, 2.8, 0, 0]
        task_5_down = [12, 0, -2.8, 0, 0]
        task_9_up = [4.3, 2, 2.8, 0, 0]
        task_9_down = [4.3, 0, -2.8, 0, 0]
    else:
        # 2 obstacle
        task_1_up = [17, 2, 2.8, 0, 0]
        task_1_down = [17, 0, -2.8, 0, 0]
        task_2_up = [9, 2, 2.8, 0, 0]
        task_2_down = [8, 0, -2.8, 0, 0]
        task_3_up = [12, 2, 2.8, 0, 0]
        task_3_down = [12, 0, -2.8, 0, 0]
        task_4_up = [4.3, 2, 2.8, 0, 0]
        task_4_down = [4.3, 0, -2.8, 0, 0]

        task_8_left = [3, 15.7, 1.8, 0, 0]
        task_8_right = [5.3, 15.7, 1.3, 0, 0]
        task_7_left = [3, 5.5, 1.8, 0, 0]
        task_7_right = [5.3, 5.5, 1.3, 0, 0]
        task_6_left = [3, 8, 1.8, 0, 0]
        task_6_right = [5.3, 8, 1.3, 0, 0]
        task_5_left = [3, 1, 1.8, 0, 0]
        task_5_right = [5.3, 1, 1.3, 0, 0]

        task_9_up = [4.7, 18, 0.2, 0, 0]
        task_9_down = [4.7, 15.8, -0.2, 0, 0]
        task_10_up = [14, 18, 0.2, 0, 0]
        task_10_down = [14, 15.8, -0.2, 0, 0]
        task_11_up = [10.5, 18, 0.2, 0, 0]
        task_11_down = [10.5, 15.8, -0.2, 0, 0]
        task_12_up = [18, 18, 0.2, 0, 0]
        task_12_down = [18, 15.8, -0.2, 0, 0]
    
    for k in ["up-down", "down-up"]:
        for i in range(1, num_task_types+1):
            for j in range(15):
                if k == "up-down" and obst_num == 1:
                    if i == 1:
                        start_x_min, start_x_max  = task_1_up[0], task_10_up[0]
                        start_y_min, start_y_max  = task_1_down[1], task_10_up[1]
                        start_theta_min, start_theta_max  = task_1_down[2], task_1_up[2]
                        goal_x_min, goal_x_max  = task_2_up[0], task_11_up[0]
                        goal_y_min, goal_y_max  = task_2_down[1], task_11_up[1]
                        goal_theta_min, goal_theta_max  = task_2_down[2], task_2_up[2]
                    elif i == 2:
                        start_x_min, start_x_max  = task_3_left[0], task_3_right[0]
                        start_y_min, start_y_max  = task_6_left[1], task_3_left[1]
                        start_theta_min, start_theta_max  = task_3_left[2], task_3_right[2]
                        goal_x_min, goal_x_max  = task_4_left[0], task_4_right[0]
                        goal_y_min, goal_y_max  = task_7_left[1], task_4_left[1]
                        goal_theta_min, goal_theta_max  = task_4_left[2], task_4_right[2]
                    elif i == 3:
                        start_x_min, start_x_max  = task_5_down[0], task_12_down[0]
                        start_y_min, start_y_max  = task_5_down[1], task_5_up[1]
                        start_theta_min, start_theta_max  = -np.pi  + (task_5_up[2] - np.pi), task_5_down[2] 
                        goal_x_min, goal_x_max  = task_9_down[0], task_13_down[0]
                        goal_y_min, goal_y_max  = task_9_down[1], task_9_up[1]
                        goal_theta_min, goal_theta_max  = -np.pi  + (task_13_up[2] - np.pi), task_13_down[2]
                    elif i == 4:
                        start_x_min, start_x_max  = task_1_up[0], task_2_up[0]
                        start_y_min, start_y_max  = task_1_down[1], task_1_up[1]
                        start_theta_min, start_theta_max  = task_1_down[2], task_1_up[2]
                        goal_x_min, goal_x_max  = task_6_left[0], task_6_right[0]
                        goal_y_min, goal_y_max  = task_7_left[1], task_6_left[1]
                        goal_theta_min, goal_theta_max  = task_6_left[2], task_6_right[2]
                    elif i == 5:
                        start_x_min, start_x_max  = task_3_left[0], task_3_right[0]
                        start_y_min, start_y_max  = task_4_left[1], task_3_left[1]
                        start_theta_min, start_theta_max  = task_3_left[2], task_3_right[2]
                        goal_x_min, goal_x_max  = task_9_up[0], task_5_up[0]
                        goal_y_min, goal_y_max  = task_9_down[1], task_5_up[1]
                        goal_theta_min, goal_theta_max  = -np.pi  + (task_9_up[2] - np.pi), task_9_down[2]
                elif k == "down-up" and obst_num == 1:
                    if i == 1:
                        start_x_min, start_x_max  = task_2_up[0] - 1.5, task_11_up[0] - 1.5
                        start_y_min, start_y_max  = task_2_down[1], task_11_up[1]
                        start_theta_min, start_theta_max  = task_2_down[2] + np.pi, task_2_up[2] + np.pi
                        goal_x_min, goal_x_max  = task_1_up[0] - 1.5, task_10_up[0] - 1.5
                        goal_y_min, goal_y_max  = task_1_down[1], task_10_up[1]
                        goal_theta_min, goal_theta_max  = task_1_down[2] + np.pi, task_1_up[2] + np.pi
                    elif i == 2:
                        start_x_min, start_x_max  = task_4_left[0], task_4_right[0]
                        start_y_min, start_y_max  = task_7_left[1] + 1.5, task_4_left[1] + 1.5
                        start_theta_min, start_theta_max  = task_4_left[2] + np.pi, task_4_right[2] + np.pi
                        goal_x_min, goal_x_max  = task_3_left[0], task_3_right[0]
                        goal_y_min, goal_y_max  = task_6_left[1] + 1.5, task_3_left[1] + 1.5
                        goal_theta_min, goal_theta_max  = task_3_left[2] + np.pi, task_3_right[2] + np.pi
                    elif i == 3:
                        start_x_min, start_x_max  = task_9_down[0] + 1.5, task_13_down[0] + 1.5
                        start_y_min, start_y_max  = task_9_down[1], task_9_up[1]
                        start_theta_min, start_theta_max  = (task_13_up[2] - np.pi), task_13_down[2] + np.pi
                        goal_x_min, goal_x_max  = task_5_down[0] + 1.5, task_12_down[0] + 1.5
                        goal_y_min, goal_y_max  = task_5_down[1], task_5_up[1]
                        goal_theta_min, goal_theta_max  = (task_5_up[2] - np.pi), task_5_down[2] + np.pi
                    elif i == 4:
                        start_x_min, start_x_max  = task_6_left[0], task_6_right[0]
                        start_y_min, start_y_max  = task_7_left[1] + 1.5, task_6_left[1] + 1.5
                        start_theta_min, start_theta_max  = task_6_left[2] + np.pi, task_6_right[2] + np.pi
                        goal_x_min, goal_x_max  = task_1_up[0] - 1.5, task_2_up[0] - 1.5 
                        goal_y_min, goal_y_max  = task_1_down[1], task_1_up[1]
                        goal_theta_min, goal_theta_max  = task_1_down[2] + np.pi, task_1_up[2] + np.pi
                    elif i == 5:
                        start_x_min, start_x_max  = task_9_up[0] + 1.5, task_5_up[0] + 1.5
                        start_y_min, start_y_max  = task_9_down[1], task_5_up[1]
                        start_theta_min, start_theta_max  = (task_9_up[2] - np.pi), task_9_down[2] + np.pi
                        goal_x_min, goal_x_max  = task_3_left[0], task_3_right[0]
                        goal_y_min, goal_y_max  = task_4_left[1] + 1.5, task_3_left[1] + 1.5
                        goal_theta_min, goal_theta_max  = task_3_left[2] + np.pi, task_3_right[2] + np.pi
                elif k == "down-up" and obst_num == 2:
                    if i == 1:
                        start_x_min, start_x_max  = task_3_down[0], task_1_down[0]
                        start_y_min, start_y_max  = task_3_down[1], task_3_up[1]
                        start_theta_min, start_theta_max  = -np.pi  + (task_3_up[2] - np.pi), task_3_down[2]
                        goal_x_min, goal_x_max  = task_4_down[0], task_2_down[0]
                        goal_y_min, goal_y_max  = task_4_down[1], task_4_up[1]
                        goal_theta_min, goal_theta_max  = -np.pi  + (task_2_up[2] - np.pi), task_2_down[2]
                    if i == 2:
                        start_x_min, start_x_max  = task_5_left[0], task_5_right[0]
                        start_y_min, start_y_max  = task_5_left[1], task_7_left[1]
                        start_theta_min, start_theta_max  = task_5_right[2], task_5_left[2]
                        goal_x_min, goal_x_max  = task_6_left[0], task_6_right[0]
                        goal_y_min, goal_y_max  = task_6_left[1], task_8_left[1]
                        goal_theta_min, goal_theta_max  = task_6_right[2], task_6_left[2]
                    if i == 3:
                        start_x_min, start_x_max  = task_9_up[0], task_11_up[0]
                        start_y_min, start_y_max  = task_9_down[1], task_9_up[1]
                        start_theta_min, start_theta_max  = task_9_down[2], task_9_up[2]
                        goal_x_min, goal_x_max  = task_10_up[0], task_12_up[0]
                        goal_y_min, goal_y_max  = task_10_down[1], task_10_up[1]
                        goal_theta_min, goal_theta_max  = task_10_down[2], task_10_up[2]
                    if i == 4:
                        start_x_min, start_x_max  = task_2_down[0], task_1_down[0]
                        start_y_min, start_y_max  = task_2_down[1], task_1_up[1]
                        start_theta_min, start_theta_max  = -np.pi  + (task_2_up[2] - np.pi), task_2_down[2]
                        goal_x_min, goal_x_max  = task_7_left[0], task_7_right[0]
                        goal_y_min, goal_y_max  = task_7_left[1], task_8_left[1]
                        goal_theta_min, goal_theta_max  = task_7_right[2], task_7_left[2]
                    if i == 5:
                        start_x_min, start_x_max  = task_5_left[0], task_5_right[0]
                        start_y_min, start_y_max  = task_5_left[1], task_6_left[1]
                        start_theta_min, start_theta_max  = task_5_right[2], task_5_left[2]
                        goal_x_min, goal_x_max  = task_11_up[0], task_12_up[0]
                        goal_y_min, goal_y_max  = task_11_down[1], task_11_up[1]
                        goal_theta_min, goal_theta_max  = task_11_down[2], task_11_up[2]
                elif k == "up-down" and obst_num == 2:
                    if i == 1:
                        start_x_min, start_x_max  = task_4_down[0] + 1.5, task_2_down[0] + 1.5
                        start_y_min, start_y_max  = task_4_down[1], task_4_up[1]
                        start_theta_min, start_theta_max  = (task_2_up[2] - np.pi), task_2_down[2] + np.pi
                        goal_x_min, goal_x_max  = task_3_down[0] + 1.5, task_1_down[0] + 1.5
                        goal_y_min, goal_y_max  = task_3_down[1], task_3_up[1]
                        goal_theta_min, goal_theta_max  = (task_3_up[2] - np.pi), task_3_down[2] + np.pi
                    if i == 2:
                        start_x_min, start_x_max  = task_6_left[0], task_6_right[0]
                        start_y_min, start_y_max  = task_6_left[1], task_8_left[1]
                        start_theta_min, start_theta_max  = task_6_right[2] + np.pi, task_6_left[2] + np.pi
                        goal_x_min, goal_x_max  = task_5_left[0], task_5_right[0]
                        goal_y_min, goal_y_max  = task_5_left[1], task_7_left[1]
                        goal_theta_min, goal_theta_max  = task_5_right[2] + np.pi, task_5_left[2] + np.pi
                    if i == 3:
                        start_x_min, start_x_max  = task_10_up[0] - 1.5, task_12_up[0] - 1.5
                        start_y_min, start_y_max  = task_10_down[1], task_10_up[1]
                        start_theta_min, start_theta_max  = task_10_down[2] + np.pi, task_10_up[2] + np.pi
                        goal_x_min, goal_x_max  = task_9_up[0] - 1.5, task_11_up[0] - 1.5
                        goal_y_min, goal_y_max  = task_9_down[1], task_9_up[1]
                        goal_theta_min, goal_theta_max  = task_9_down[2] + np.pi, task_9_up[2] + np.pi
                    if i == 4:
                        start_x_min, start_x_max  = task_7_left[0], task_7_right[0]
                        start_y_min, start_y_max  = task_7_left[1], task_8_left[1]
                        start_theta_min, start_theta_max  = task_7_right[2] + np.pi, task_7_left[2] + np.pi
                        goal_x_min, goal_x_max  = task_2_down[0], task_1_down[0]
                        goal_y_min, goal_y_max  = task_2_down[1], task_1_up[1]
                        goal_theta_min, goal_theta_max  = (task_2_up[2] - np.pi), task_2_down[2] + np.pi
                    if i == 5:
                        start_x_min, start_x_max  = task_11_up[0], task_12_up[0]
                        start_y_min, start_y_max  = task_11_down[1], task_11_up[1]
                        start_theta_min, start_theta_max  = task_11_down[2] + np.pi, task_11_up[2] + np.pi
                        goal_x_min, goal_x_max  = task_5_left[0], task_5_right[0]
                        goal_y_min, goal_y_max  = task_5_left[1], task_6_left[1]
                        goal_theta_min, goal_theta_max  = task_5_right[2] + np.pi, task_5_left[2] + np.pi
                state = [np.random.uniform(start_x_min, start_x_max), 
                        np.random.uniform(start_y_min, start_y_max),
                        normalizeAngle(np.random.uniform(start_theta_min, start_theta_max)),
                        0,
                        0]
                goal = [np.random.uniform(goal_x_min, goal_x_max), 
                        np.random.uniform(goal_y_min, goal_y_max),
                        normalizeAngle(np.random.uniform(goal_theta_min, goal_theta_max)),
                        0,
                        0]
                lst_starts.append(state)
                lst_goals.append(goal)

    assert len(lst_starts) == len(lst_goals)
    return lst_starts, lst_goals

map_1 = [[8, 9, 0.0, 5, 7],
        [28, 9, 0.0, 11, 7],
        [11, 25, 0.0, 5, 10],
        [-1, 9, 0.0, 11, 2],
        [11, -4, 0.0, 2, 10]]


map_2 = [[14, 9, 0.0, 5, 7],
        [28, 9, 0.0, 11, 7],
        [11, 25, 0.0, 5, 10],
        [-1, 9, 0.0, 11, 2],
        [11, -4, 0.0, 2, 10]]

print(" Saving dataset!!! ")
save_map(map_1, "obstacle_map0.txt")
lst_starts, lst_goals = task_generator(obst_num=1)
save_dataset(lst_starts, lst_goals, "train_map0.txt")

save_map(map_2, "obstacle_map1.txt")
lst_starts, lst_goals = task_generator(obst_num=2)
save_dataset(lst_starts, lst_goals, "train_map1.txt")