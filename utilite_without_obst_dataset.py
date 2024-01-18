import numpy as np
from polamp_env.lib.utils_operations import normalizeAngle
from MFNLC_for_polamp_env.mfnlc.plan import Planner

def save_dataset(lst_starts, lst_goals, name):
    with open("without_obst_rrt_dataset/" + name, 'w') as output:
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
def task_generator(train=False, use_rrt=False):
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

    def shift_patern(patern, left=False, down=False, up=False, right=False, dist=0):
        assert left or down or up or right
        patern_type = patern[1]
        task1 = patern[0][0]
        task2 = patern[0][1]
        task3 = patern[0][2]
        task4 = patern[0][3]
        task1_copy = task1.copy()
        task2_copy = task2.copy()
        task3_copy = task3.copy()
        task4_copy = task4.copy()
        if left:
            for t in [task1_copy, task2_copy, task3_copy, task4_copy]:
                t[0] -= dist
        elif down:
            for t in [task1_copy, task2_copy, task3_copy, task4_copy]:
                t[1] -= dist
        elif up:
            for t in [task1_copy, task2_copy, task3_copy, task4_copy]:
                t[1] += dist
        elif right:
            for t in [task1_copy, task2_copy, task3_copy, task4_copy]:
                t[0] += dist

        return ([task1_copy, task2_copy, task3_copy, task4_copy], patern_type)

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
    
    # shifted patterns
    patern_17_reverse = shift_patern(patern_1, up=True, dist=7)
    patern_17 = reverse_patern(patern_17_reverse)
    patern_18 = shift_patern(patern_4, up=True, dist=7)
    patern_18_reverse = reverse_patern(patern_18)
    patern_19_reverse = shift_patern(patern_1, down=True, dist=7)
    patern_19 = reverse_patern(patern_19_reverse)
    patern_20 = shift_patern(patern_4, down=True, dist=7)
    patern_20_reverse = reverse_patern(patern_20)
    patern_22 = reverse_patern(shift_patern(patern_7, left=True, dist=14))
    patern_22_reverse = reverse_patern(patern_22)
    patern_23 = reverse_patern(shift_patern(patern_7, right=True, dist=14))
    patern_23_reverse = reverse_patern(patern_23)
    patern_21 = shift_patern(patern_8, left=True, dist=14)
    patern_21_reverse = reverse_patern(patern_21)
    patern_24 = shift_patern(patern_8, right=True, dist=14)
    patern_24_reverse = reverse_patern(patern_24)

    def patern_task(patern, lst_starts, lst_goals, num_tasks=1, use_rrt=False):
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
            if use_rrt:
                env = 
                env.set_agent_state = 
                planner = Planner(env, planning_algo)
                start_plan = time.time()
                path = planner.plan(planner_max_iter, **planning_algo_kwargs)
                end_plan = time.time()
                print("plan time:", end_plan - start_plan)
                lst_starts.append(state)
                lst_goals.append(goal)
            else:
                lst_starts.append(state)
                lst_goals.append(goal)
    
    all_paterns = [patern_1, patern_2, patern_3, patern_4, patern_5, patern_6, patern_7,
                   patern_8, patern_9, patern_10, patern_11, patern_12, patern_13, patern_14,
                   patern_15, patern_16, patern_17, patern_18, patern_19, patern_20, patern_21,
                   patern_22, patern_23, patern_24]
    all_paterns_reverse = [patern_1_reverse, patern_2_reverse, patern_3_reverse, patern_4_reverse, 
                           patern_5_reverse, patern_6_reverse, patern_7_reverse, patern_8_reverse, 
                           patern_9_reverse, patern_10_reverse, patern_11_reverse, patern_12_reverse, 
                           patern_13_reverse, patern_14_reverse, patern_15_reverse, patern_16_reverse,
                           patern_17_reverse, patern_18_reverse, patern_19_reverse, patern_20_reverse,
                           patern_21_reverse, patern_22_reverse, patern_23_reverse, patern_24_reverse]
    def evaluation_task(eval_task, all_paterns, all_paterns_reverse, 
                        lst_starts, lst_goals, num_tasks=15, 
                        first_two_digit=False, second_two_digit=False):
        patern_1_type = eval_task[0]
        patern_2_type = eval_task[-1]
        if not first_two_digit and not second_two_digit:
            assert len(eval_task) == 4, eval_task
            patern_temp_1_idx = int(eval_task[1]) - 1
            patern_temp_2_idx = int(eval_task[2]) - 1
        elif first_two_digit and second_two_digit:
            assert len(eval_task) == 6, eval_task
            patern_temp_1_idx = int(eval_task[1:3]) - 1
            patern_temp_2_idx = int(eval_task[3:5]) - 1
        else:
            assert len(eval_task) == 5, eval_task
            if first_two_digit:
                patern_temp_1_idx = int(eval_task[1:3]) - 1
                patern_temp_2_idx = int(eval_task[3]) - 1
            else:
                patern_temp_1_idx = int(eval_task[1]) - 1
                patern_temp_2_idx = int(eval_task[2:4]) - 1
            
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
        patern_task(patern_temp_1, temp_starts, temp_goals, num_tasks=15)
        lst_starts.extend(temp_starts)

        temp_starts = []
        temp_goals = []
        patern_task(patern_temp_2, temp_starts, temp_goals, num_tasks=15)
        lst_goals.extend(temp_goals)


    if train:
        #train_paterns = all_paterns
        #train_paterns.extend(all_paterns_reverse)
        #assert len(train_paterns) == 32
        if use_rrt:
            train_paterns = [patern_1]
            for patern in train_paterns:
                patern_task(patern, lst_starts, lst_goals, num_tasks=1, use_rrt=use_rrt)
        else:
            train_paterns = [patern_1, patern_2, patern_3, patern_4, 
                            patern_5, patern_6, patern_8, 
                            patern_17, patern_18, patern_19, patern_20, 
                            patern_21, patern_22, patern_23, patern_24]
            train_paterns.extend([patern_1_reverse, patern_2_reverse, 
                                patern_3_reverse, patern_4_reverse, 
                                patern_5_reverse, patern_6_reverse, patern_8_reverse, 
                                patern_17_reverse, patern_18_reverse, 
                                patern_19_reverse, patern_20_reverse, 
                                patern_21_reverse, patern_22_reverse, 
                                patern_23_reverse, patern_24_reverse])
            for patern in train_paterns:
                patern_task(patern, lst_starts, lst_goals, num_tasks=15)
    else:
        if use_rrt:
            pass
        else:
            eval_tasks = ["r23f", "r24f", "r25f", "f13f", "f14f", "f15f", "r63f", "r64f", "r65f", 
                        "r32f", "r31r", "r36f", "r42f", "r41r", "r46f", "r52f", "r51r", "r56f", 
                        ("f2221f", "all"), ("f228f", "first"), ("f2224f", "all"), 
                        ("r721f", "second"), "r78f", ("r724f", "second"), 
                        ("f2321f", "all"), ("f238f", "first"), ("f2324f", "all"),
                        ("r2122r", "all"), ("r217f", "first"), ("r2123r", "all"),
                        ("r822r", "second"), "r87f", ("r823r", "second"),
                        ("r2422r", "all"), ("r247f", "first"), ("r2423r", "all"),
                        ]
            for eval_task in eval_tasks:
                if type(eval_task) == type(tuple()):
                    if eval_task[1] == "first":
                        evaluation_task(eval_task[0], all_paterns, all_paterns_reverse, 
                                    lst_starts, lst_goals, num_tasks=15, 
                                    first_two_digit=True)
                    elif eval_task[1] == "second":
                        evaluation_task(eval_task[0], all_paterns, all_paterns_reverse, 
                                        lst_starts, lst_goals, num_tasks=15, 
                                        second_two_digit=True)
                    else:
                        evaluation_task(eval_task[0], all_paterns, all_paterns_reverse, 
                                        lst_starts, lst_goals, num_tasks=15, 
                                        first_two_digit=True, second_two_digit=True)
                else:
                    evaluation_task(eval_task, all_paterns, all_paterns_reverse, 
                                    lst_starts, lst_goals, num_tasks=15)
            # test
            #train_paterns = [patern_22]
            #for patern in train_paterns:
            #    patern_task(patern, lst_starts, lst_goals, num_tasks=15)
    assert len(lst_starts) == len(lst_goals)
    return lst_starts, lst_goals

print(" Saving dataset!!! ")
train_lst_starts, train_lst_goals = task_generator(train=True, use_rrt=True)
eval_lst_starts, eval_lst_goals = task_generator(train=False, use_rrt=True)
train_lst_starts.extend(eval_lst_starts)
train_lst_goals.extend(eval_lst_goals)

save_dataset(train_lst_starts, train_lst_goals, "train_map0.txt")
save_dataset(eval_lst_starts, eval_lst_goals, "val_map0.txt")
