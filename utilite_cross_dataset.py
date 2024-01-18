import numpy as np
from polamp_env.lib.utils_operations import normalizeAngle

def save_dataset(lst_starts, lst_goals, name, file_key='w'):
    with open("cross_dataset_for_test/" + name, file_key) as output:
        if file_key == "a":
            pass
        else:
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
    with open("without_obst_rrt_dataset/" + name, 'w') as output:
    # for obstacle in obstacles:
        output.write(str(len(obstacle)) + '\n')
        for polygon in obstacle:
            for element in polygon:
                output.write(str(element) + '\t')
            output.write("\n")

# number_of_tasks = 10
def task_generator(train=False, tasks_per_patern=15):
    lst_starts = []
    lst_goals = []

    # forward paterns
    task1 = [4, 18, 0.2, 0, 0]
    task2 = [11, 18, 0.2, 0, 0]
    task3 = [8, 16, -0.2, 0, 0]
    task4 = [15, 16, -0.2, 0, 0]
    patern_1 = ([task1, task2, task3, task4], "left_right")

    task1 = [13, 35, 2.8, 0, 0]
    task2 = [4, 35, 2.8, 0, 0]
    task3 = [8, 32, -2.8, 0, 0]
    task4 = [2, 32, -2.8, 0, 0]
    patern_2 = ([task1, task2, task3, task4], "right_left")

    task1 = [23, 2, 0.2, 0, 0]
    task2 = [31, 2, 0.2, 0, 0]
    task3 = [27, -1, -0.2, 0, 0]
    task4 = [35, -1, -0.2, 0, 0]
    patern_5 = ([task1, task2, task3, task4], "left_right")

    task1 = [19.5, 12, -1.3, 0, 0]
    task2 = [19.5, 6, -1.3, 0, 0]
    task3 = [17, 9, -1.8, 0, 0]
    task4 = [17, 4, -1.8, 0, 0]
    patern_8 = ([task1, task2, task3, task4], "up_down")

    task1 = [22, 18, 0.2, 0, 0]
    task2 = [30.5, 18, 0.2, 0, 0]
    task3 = [27.5, 16, -0.2, 0, 0]
    task4 = [35, 16, -0.2, 0, 0]
    patern_4 = ([task1, task2, task3, task4], "left_right")

    task1 = [22, 35, 0.2, 0, 0]
    task2 = [31.5, 35, 0.2, 0, 0]
    task3 = [27.5, 32, -0.2, 0, 0]
    task4 = [35, 32, -0.2, 0, 0]
    patern_3 = ([task1, task2, task3, task4], "left_right")

    task1 = [17, 21, 1.8, 0, 0]
    task2 = [17, 27.5, 1.8, 0, 0]
    task3 = [19, 24.5, 1.3, 0, 0]
    task4 = [19, 30, 1.3, 0, 0]
    patern_7 = ([task1, task2, task3, task4], "down_up")

    task1 = [14, 2, 2.8, 0, 0]
    task2 = [6.5, 2, 2.8, 0, 0]
    task3 = [9.5, -1, -2.8, 0, 0]
    task4 = [2, -1, -2.8, 0, 0]
    patern_6 = ([task1, task2, task3, task4], "right_left")

    def reverse_patern(patern):
        patern_type = patern[1]
        if patern_type == "right_left" or patern_type == "left_right" \
           or patern_type == "down_up" or patern_type == "up_down":
            task1 = patern[0][0]
            task2 = patern[0][1]
            task3 = patern[0][2]
            task4 = patern[0][3]
        if patern_type == "right_left":
            patern_type = "left_right"
            task1_copy = [task4[0], task1[1], normalizeAngle(task4[2] + np.pi), 0, 0]
            task2_copy = [task3[0], task1[1], normalizeAngle(task4[2] + np.pi), 0, 0]
            task3_copy = [task2[0], task3[1], normalizeAngle(task2[2] + np.pi), 0, 0]
            task4_copy = [task1[0], task4[1], normalizeAngle(task1[2] + np.pi), 0, 0]
        elif patern_type == "left_right":
            task1_copy = [task4[0], task1[1], normalizeAngle(task4[2] + np.pi), 0, 0]
            task2_copy = [task3[0], task1[1], normalizeAngle(task4[2] + np.pi), 0, 0]
            task3_copy = [task2[0], task3[1], normalizeAngle(task2[2] + np.pi), 0, 0]
            task4_copy = [task1[0], task4[1], normalizeAngle(task1[2] + np.pi), 0, 0]
            patern_type = "right_left"
        elif patern_type == "down_up":
            task1_copy = [task4[0], task4[1], normalizeAngle(task1[2] + np.pi), 0, 0]
            task2_copy = [task3[0], task3[1], normalizeAngle(task2[2] + np.pi), 0, 0]
            task3_copy = [task2[0], task2[1], normalizeAngle(task3[2] + np.pi), 0, 0]
            task4_copy = [task1[0], task1[1], normalizeAngle(task4[2] + np.pi), 0, 0] 
            patern_type = "up_down"
        elif patern_type == "up_down":
            task1_copy = [task4[0], task4[1], normalizeAngle(task1[2] + np.pi), 0, 0]
            task2_copy = [task3[0], task3[1], normalizeAngle(task2[2] + np.pi), 0, 0]
            task3_copy = [task2[0], task2[1], normalizeAngle(task3[2] + np.pi), 0, 0]
            task4_copy = [task1[0], task1[1], normalizeAngle(task4[2] + np.pi), 0, 0] 
            patern_type = "down_up"
        else:
            temp_patern_1 = patern[0][0]
            temp_patern_2 = patern[0][1]
            temp_patern_1 = reverse_patern(temp_patern_1)
            temp_patern_2 = reverse_patern(temp_patern_2)
            if patern_type == "turn_down_up_left":
                patern_type = "turn_left_right_down"
            elif patern_type == "turn_down_up_right":
                patern_type = "turn_right_left_down"
            elif patern_type == "turn_up_down_left":
                patern_type = "turn_left_right_up"
            elif patern_type == "turn_up_down_right":
                patern_type = "turn_right_left_up"
            elif patern_type == "turn_left_right_up":
                patern_type = "turn_up_down_left"
            elif patern_type == "turn_left_right_down":
                patern_type = "turn_down_up_left"
            elif patern_type == "turn_right_left_up":
                patern_type = "turn_up_down_right"
            elif patern_type == "turn_right_left_down":
                patern_type = "turn_down_up_right"
            else:
                assert 1 == 0
            return ((temp_patern_2, temp_patern_1), patern_type)
        return ([task1_copy, task2_copy, task3_copy, task4_copy], patern_type)

    # reverse paterns
    patern_1_reverse = reverse_patern(patern_1)
    patern_2_reverse = reverse_patern(patern_2)
    patern_3_reverse = reverse_patern(patern_3)
    patern_4_reverse = reverse_patern(patern_4)
    patern_5_reverse = reverse_patern(patern_5)
    patern_6_reverse = reverse_patern(patern_6)
    patern_7_reverse = reverse_patern(patern_7)
    patern_8_reverse = reverse_patern(patern_8)

    # turn paterns
    patern_13 = ((patern_7, patern_2), "turn_down_up_left")
    patern_14 = ((patern_7, patern_3), "turn_down_up_right")
    patern_15 = ((patern_8, patern_6), "turn_up_down_left")
    patern_16 = ((patern_8, patern_5), "turn_up_down_right")
    patern_9 = ((patern_1, patern_7), "turn_left_right_up")
    patern_12 = ((patern_1, patern_8), "turn_left_right_down")
    patern_10 = ((patern_4_reverse, patern_7), "turn_right_left_up")
    patern_11 = ((patern_4_reverse, patern_8), "turn_right_left_down")

    # reverse paterns
    patern_13_reverse = reverse_patern(patern_13)
    patern_14_reverse = reverse_patern(patern_14)
    patern_9_reverse = reverse_patern(patern_9)
    patern_10_reverse = reverse_patern(patern_10)
    patern_12_reverse = reverse_patern(patern_12)
    patern_11_reverse = reverse_patern(patern_11)
    patern_16_reverse = reverse_patern(patern_16)
    patern_15_reverse = reverse_patern(patern_15)

    def patern_task(patern, lst_starts, lst_goals, num_tasks=1):
        for i in range(num_tasks):
            patern_type = patern[1]
            if patern_type == "left_right" or patern_type == "right_left" \
                   or patern_type == "down_up" or patern_type == "up_down":
                task1 = patern[0][0]
                task2 = patern[0][1]
                task3 = patern[0][2]
                task4 = patern[0][3]
            else:
                temp_patern_1 = patern[0][0]
                temp_patern_2 = patern[0][1]
                task1_p1 = temp_patern_1[0][0]
                task2_p1 = temp_patern_1[0][1]
                task3_p1 = temp_patern_1[0][2]
                task4_p1 = temp_patern_1[0][3]
                task1_p2 = temp_patern_2[0][0]
                task2_p2 = temp_patern_2[0][1]
                task3_p2 = temp_patern_2[0][2]
                task4_p2 = temp_patern_2[0][3]
            assert patern_type == "left_right" or patern_type == "right_left" \
                   or patern_type == "down_up" or patern_type == "up_down" \
                   or patern_type == "turn_down_up_left" or patern_type == "turn_down_up_right" \
                   or patern_type == "turn_up_down_left" or patern_type == "turn_up_down_right" \
                   or patern_type == "turn_left_right_up" or patern_type == "turn_left_right_down" \
                   or patern_type == "turn_right_left_up" or patern_type == "turn_right_left_down"
            # x, y, theta bounds
            if patern_type == "left_right":
                start_x_min, start_x_max  = task1[0], task3[0]
                start_y_min, start_y_max  = task3[1], task1[1]
                start_theta_min, start_theta_max = task3[2], task1[2]
                goal_x_min, goal_x_max  = task2[0], task4[0]
                goal_y_min, goal_y_max  = task4[1], task2[1]
                goal_theta_min, goal_theta_max  = task4[2], task2[2]
            elif patern_type == "right_left":
                start_x_min, start_x_max  = task3[0], task1[0]
                start_y_min, start_y_max  = task3[1], task1[1]
                start_theta_min, start_theta_max = -np.pi  + (task1[2] - np.pi), task3[2]
                goal_x_min, goal_x_max  = task4[0], task2[0]
                goal_y_min, goal_y_max  = task4[1], task2[1]
                goal_theta_min, goal_theta_max  = -np.pi  + (task2[2] - np.pi), task4[2]
            elif patern_type == "up_down":
                start_x_min, start_x_max  = task3[0], task1[0]
                start_y_min, start_y_max  = task3[1], task1[1]
                start_theta_min, start_theta_max = task3[2], task1[2]
                goal_x_min, goal_x_max  = task4[0], task2[0]
                goal_y_min, goal_y_max  = task4[1], task2[1]
                goal_theta_min, goal_theta_max  = task4[2], task2[2]
            elif patern_type == "down_up":
                start_x_min, start_x_max  = task1[0], task3[0]
                start_y_min, start_y_max  = task1[1], task3[1]
                start_theta_min, start_theta_max = task3[2], task1[2]
                goal_x_min, goal_x_max  = task2[0], task4[0]
                goal_y_min, goal_y_max  = task2[1], task4[1]
                goal_theta_min, goal_theta_max  = task4[2], task2[2]
            elif patern_type == "turn_down_up_left":
                assert temp_patern_1[1] == "down_up"
                assert temp_patern_2[1] == "right_left"
                start_x_min, start_x_max  = task4_p1[0], task2_p1[0]
                start_y_min, start_y_max  = task2_p1[1], task4_p1[1]
                start_theta_min, start_theta_max = task2_p1[2], task4_p1[2]
                goal_x_min, goal_x_max  = task3_p2[0], task1_p2[0]
                goal_y_min, goal_y_max  = task3_p2[1], task1_p2[1]
                goal_theta_min, goal_theta_max  = -np.pi  + (task1_p2[2] - np.pi), task3_p2[2]
            elif patern_type == "turn_down_up_right":
                assert temp_patern_1[1] == "down_up"
                assert temp_patern_2[1] == "left_right"
                start_x_min, start_x_max  = task4_p1[0], task2_p1[0]
                start_y_min, start_y_max  = task2_p1[1], task4_p1[1]
                start_theta_min, start_theta_max = task2_p1[2], task4_p1[2]
                goal_x_min, goal_x_max  = task1_p2[0], task3_p2[0]
                goal_y_min, goal_y_max  = task3_p2[1], task1_p2[1]
                goal_theta_min, goal_theta_max  = task3_p2[2], task1_p2[2]
            elif patern_type == "turn_up_down_left":
                assert temp_patern_1[1] == "up_down"
                assert temp_patern_2[1] == "right_left"
                start_x_min, start_x_max  = task4_p1[0], task2_p1[0]
                start_y_min, start_y_max  = task4_p1[1], task2_p1[1]
                start_theta_min, start_theta_max = task4_p1[2], task2_p1[2]
                goal_x_min, goal_x_max  = task3_p2[0], task1_p2[0]
                goal_y_min, goal_y_max  = task3_p2[1], task1_p2[1]
                goal_theta_min, goal_theta_max = -np.pi  + (task1_p2[2] - np.pi), task3_p2[2]
            elif patern_type == "turn_up_down_right":
                assert temp_patern_1[1] == "up_down"
                assert temp_patern_2[1] == "left_right"
                start_x_min, start_x_max  = task4_p1[0], task2_p1[0]
                start_y_min, start_y_max  = task4_p1[1], task2_p1[1]
                start_theta_min, start_theta_max = task4_p1[2], task2_p1[2]
                goal_x_min, goal_x_max  = task1_p2[0], task3_p2[0]
                goal_y_min, goal_y_max  = task3_p2[1], task1_p2[1]
                goal_theta_min, goal_theta_max  = task3_p2[2], task1_p2[2]
            elif patern_type == "turn_left_right_up":
                assert temp_patern_1[1] == "left_right"
                assert temp_patern_2[1] == "down_up"
                start_x_min, start_x_max  = task2_p1[0], task4_p1[0]
                start_y_min, start_y_max  = task4_p1[1], task2_p1[1]
                start_theta_min, start_theta_max = task4_p1[2], task2_p1[2]
                goal_x_min, goal_x_max  = task1_p2[0], task3_p2[0]
                goal_y_min, goal_y_max  = task1_p2[1], task3_p2[1]
                goal_theta_min, goal_theta_max  = task3_p2[2], task1_p2[2]
            elif patern_type == "turn_left_right_down":
                assert temp_patern_1[1] == "left_right", f"{temp_patern_1[1]}" + " patern type: turn_left_right_down"
                assert temp_patern_2[1] == "up_down"
                start_x_min, start_x_max  = task2_p1[0], task4_p1[0]
                start_y_min, start_y_max  = task4_p1[1], task2_p1[1]
                start_theta_min, start_theta_max = task4_p1[2], task2_p1[2]
                goal_x_min, goal_x_max  = task3_p2[0], task1_p2[0]
                goal_y_min, goal_y_max  = task3_p2[1], task1_p2[1]
                goal_theta_min, goal_theta_max  = task3_p2[2], task1_p2[2]
            elif patern_type == "turn_right_left_up":
                assert temp_patern_1[1] == "right_left"
                assert temp_patern_2[1] == "down_up"
                start_x_min, start_x_max  = task4_p1[0], task2_p1[0]
                start_y_min, start_y_max  = task4_p1[1], task2_p1[1]
                start_theta_min, start_theta_max = -np.pi  + (task2_p1[2] - np.pi), task4_p1[2]
                goal_x_min, goal_x_max  = task1_p2[0], task3_p2[0]
                goal_y_min, goal_y_max  = task1_p2[1], task3_p2[1]
                goal_theta_min, goal_theta_max  = task3_p2[2], task1_p2[2]
            elif patern_type == "turn_right_left_down":
                assert temp_patern_1[1] == "right_left"
                assert temp_patern_2[1] == "up_down" 
                start_x_min, start_x_max  = task4_p1[0], task2_p1[0]
                start_y_min, start_y_max  = task4_p1[1], task2_p1[1]
                start_theta_min, start_theta_max = -np.pi  + (task2_p1[2] - np.pi), task4_p1[2]
                goal_x_min, goal_x_max  = task3_p2[0], task1_p2[0]
                goal_y_min, goal_y_max  = task3_p2[1], task1_p2[1]
                goal_theta_min, goal_theta_max  = task3_p2[2], task1_p2[2]
            else:
                assert 1 == 0
            # get state, goal
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
    
    all_paterns = [patern_1, patern_2, patern_3, patern_4, patern_5, patern_6, patern_7,
                   patern_8, patern_9, patern_10, patern_11, patern_12, patern_13, patern_14,
                   patern_15, patern_16]
    all_paterns_reverse = [patern_1_reverse, patern_2_reverse, patern_3_reverse, patern_4_reverse, 
                           patern_5_reverse, patern_6_reverse, patern_7_reverse, patern_8_reverse, 
                           patern_9_reverse, patern_10_reverse, patern_11_reverse, patern_12_reverse, 
                           patern_13_reverse, patern_14_reverse, patern_15_reverse, patern_16_reverse]
    def evaluation_task(eval_task, all_paterns, all_paterns_reverse, 
                        lst_starts, lst_goals, num_tasks=15):
        assert len(eval_task) == 4
        patern_1_type = eval_task[0]
        patern_2_type = eval_task[3]
        patern_temp_1_idx = int(eval_task[1]) - 1
        patern_temp_2_idx = int(eval_task[2]) - 1
        assert patern_1_type == 'f' or patern_1_type == 'r'
        assert patern_2_type == 'f' or patern_2_type == 'r'
        if patern_1_type == 'f':
            patern_temp_1 = all_paterns[patern_temp_1_idx] 
        else:
            patern_temp_1 = all_paterns_reverse[patern_temp_1_idx]
        if patern_2_type == 'f':
            patern_temp_2 = all_paterns[patern_temp_2_idx] 
        else:
            patern_temp_2 = all_paterns_reverse[patern_temp_2_idx]

        temp_starts = []
        temp_goals = []
        patern_task(patern_temp_1, temp_starts, temp_goals, num_tasks=num_tasks)
        lst_starts.extend(temp_starts)

        temp_starts = []
        temp_goals = []
        patern_task(patern_temp_2, temp_starts, temp_goals, num_tasks=num_tasks)
        lst_goals.extend(temp_goals)


    if train:
        train_paterns = all_paterns
        train_paterns.extend(all_paterns_reverse)
        assert len(train_paterns) == 32
        for patern in train_paterns:
            patern_task(patern, lst_starts, lst_goals, num_tasks=tasks_per_patern)
    else:
        eval_tasks = ["r23f", "r24f", "r25f", "r26f", "r21r", 
                      "r34f", "r35f", "r36f", "r31r", "r32f",
                      "r45f", "r46f", "r41r", "r42f", "r43f",
                      "r56f", "r51r", "r52f", "r53f", "r54f",
                      "r61r", "r62f", "r63f", "r64f", "r65f",
                      "f12f", "f13f", "f14f", "f15f", "f16f",
                      ]
        for eval_task in eval_tasks:
            evaluation_task(eval_task, all_paterns, all_paterns_reverse, 
                            lst_starts, lst_goals, num_tasks=tasks_per_patern)
    assert len(lst_starts) == len(lst_goals)
    return lst_starts, lst_goals

if __name__ == "__main__":
    print(" Saving dataset!!! ")

    rrt_train = False
    form_last_dataset = True
    eval_task_per_patern = 1
    get_test_dataset = True
    if get_test_dataset:
        eval_lst_starts, eval_lst_goals = task_generator(train=False, tasks_per_patern=2)        
        save_dataset(eval_lst_starts, eval_lst_goals, "val_map0.txt")
    else:
        if not form_last_dataset:
            if rrt_train:
                train_lst_starts, train_lst_goals = task_generator(train=True, tasks_per_patern=5)
                eval_lst_starts, eval_lst_goals = task_generator(train=False, tasks_per_patern=eval_task_per_patern)

                save_dataset(train_lst_starts, train_lst_goals, "train_map0.txt")
                save_dataset(eval_lst_starts, eval_lst_goals, "val_map0.txt")
            else:
                train_lst_starts, train_lst_goals = task_generator(train=True, tasks_per_patern=5)
                task_generator(train=False, tasks_per_patern=5)
                eval_lst_starts, eval_lst_goals = task_generator(train=False, tasks_per_patern=eval_task_per_patern)

                save_dataset(train_lst_starts, train_lst_goals, "train_map0.txt")
                save_dataset(eval_lst_starts, eval_lst_goals, "val_map0.txt")
        else:
            task_generator(train=False, tasks_per_patern=5)
            task_generator(train=False, tasks_per_patern=5)
            task_generator(train=False, tasks_per_patern=5)
            train_lst_starts, train_lst_goals = task_generator(train=True, tasks_per_patern=5)

            save_dataset(train_lst_starts, train_lst_goals, "train_map0.txt")

            train_lst_starts, train_lst_goals = task_generator(train=True, tasks_per_patern=3)
            save_dataset(train_lst_starts, train_lst_goals, "val_map0.txt")



