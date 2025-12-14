import numpy as np


def r(arr, i, v):
    res = arr[:]
    res[i] = v
    return res


def avg_acc(acc_func, blocks_pruning_strategy, start_iteration, end_iteration):
    # acc_sum = 0
    # for i in range(start_iteration, end_iteration + 1):
    #     acc_sum += acc_func(blocks_pruning_strategy, i)
    
    n = len(list(range(start_iteration, end_iteration + 1)))
    accs = acc_func([blocks_pruning_strategy] * n, list(range(start_iteration, end_iteration + 1)), use_batch=True)
    acc_sum = sum(accs)
    return acc_sum / n


def heuristic_solve(acc_func, blocks_retraining_time_per_iteration, retraining_iter_search_quantum, retraining_window, inverse_debug=False):
    num_blocks = len(blocks_retraining_time_per_iteration)
    
    blocks_pruning_strategy = [1] * num_blocks
    num_retraining_iterations = 0
    left_retraining_time = retraining_window
    
    while True:
        acc_gain_per_time_unit = []
        for i in range(num_blocks):
            if blocks_pruning_strategy[i] == 0 or \
                (blocks_retraining_time_per_iteration[i][0] - blocks_retraining_time_per_iteration[i][1]) * num_retraining_iterations > left_retraining_time:
                acc_gain_per_time_unit.append(0)
            else:
                acc_gain_per_time_unit.append(
                    (avg_acc(acc_func, r(blocks_pruning_strategy, i, 0), 0, num_retraining_iterations) - avg_acc(acc_func, r(blocks_pruning_strategy, i, 1), 0, num_retraining_iterations)) / (blocks_retraining_time_per_iteration[i][0] - blocks_retraining_time_per_iteration[i][1])
                )
        
        st1 = [t[b] for t, b in zip(blocks_retraining_time_per_iteration, blocks_pruning_strategy)]
        st = sum(st1) * retraining_iter_search_quantum
        if st <= left_retraining_time:
            acc_gain_per_time_unit.append(
                avg_acc(acc_func, blocks_pruning_strategy, num_retraining_iterations, num_retraining_iterations + retraining_iter_search_quantum) / st
            )
        else:
            acc_gain_per_time_unit.append(0)
            
        # print(f'acc gain per time unit: {acc_gain_per_time_unit}')
            
        if sum(acc_gain_per_time_unit) == 0:
            # print(f'q += {left_retraining_time // st}')
            num_retraining_iterations += left_retraining_time // sum(st1)
            break
        
        # if inverse_debug:
        #     acc_gain_per_time_unit = [-_a for _a in acc_gain_per_time_unit]
        
        if inverse_debug:
            print(acc_gain_per_time_unit)
            acc_gain_per_time_unit[acc_gain_per_time_unit < 1e-7] = 1000
            best_action = np.argmin(acc_gain_per_time_unit)
        else:
            best_action = np.argmax(acc_gain_per_time_unit)
            
        if best_action < num_blocks:
            blocks_pruning_strategy[best_action] = 0
            left_retraining_time -= (blocks_retraining_time_per_iteration[i][0] - blocks_retraining_time_per_iteration[i][1]) * num_retraining_iterations
            # print(f'unprune block {best_action}, time -= {(blocks_retraining_time_per_iteration[i][0] - blocks_retraining_time_per_iteration[i][1]) * num_retraining_iterations}')
        else:
            num_retraining_iterations += retraining_iter_search_quantum
            left_retraining_time -= st
            # print(f'q += {retraining_iter_search_quantum}, time -= {st}')
        
        # print(f'left time: {left_retraining_time}')
    
    # print(f'final solution: {blocks_pruning_strategy}, {num_retraining_iterations}')
    
    return blocks_pruning_strategy, num_retraining_iterations


def mul(arr):
    res = 1.
    for x in arr:
        res *= x
    return res


if __name__ == '__main__':
    accs = np.array([[0.9, 0.91], [0.85, 0.75], [0.88, 0.89]])
    def acc_func(B, q):
        res = 1.
        for i in range(len(B)):
            res *= accs[i][B[i]]
        res *= (q / 1000)
        return res
        
    heuristic_solve(
        acc_func,
        np.array([[10, 7], [11, 7], [9, 6]]),
        10,
        600
    )
    