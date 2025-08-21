import sys
import os
from ortools.sat.python import cp_model

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scheduling_environment.job import Job
from scheduling_environment.operation import Operation
from scheduling_environment.machine import Machine
from scheduling_environment.jobShop import JobShop
from visualization import gantt_with_pm

def schedule_pm_and_plot():
    # ====================================================================
    # 1. 模擬輸入 (來自第一階段 DRL 的生產排程)
    # ====================================================================
    # 格式: (job_id, op_id, machine_id, start_time, end_time)
    # 這些是固定的，CP 模型不能改變它們。
    schedule_from_drl = [
        (0, 0, 0, 0, 10), (0, 1, 1, 10, 25), (0, 2, 0, 30, 45), # 機器0: [25,30] 有空隙
        (1, 0, 1, 0, 10), (1, 1, 2, 10, 20), (1, 2, 1, 30, 40), # 機器1: [25,30] 有空隙
        (2, 0, 2, 0, 10), (2, 1, 0, 10, 25), (2, 2, 2, 25, 35), # 機器2: [20,25] 有空隙
    ]
    num_machines = 3
    # 預計一個較大的可能完工時間，讓模型有足夠的搜索空間
    # 確保這個上限足夠大，以容納所有作業和PM，避免無解。
    upper_bound_makespan = 10000 
    dynamic_params = {
        'utilization': [0.4, 0.35, 0.3],
        'switch_cost_factor': [2, 2, 2]
    }
    cost_params = {
        'Cp_k': [100, 120, 110],  # 每台機器的 PM 固定成本 (C_pk)
        'Cf_k': [1000, 1100, 1050], # 每台機器的故障固定成本 (C_fk)
        'Tp_k': [5, 6, 5],    # 每台機器的 PM 時長
        'w_makespan_cost': 1.0 # makespan成本權重
    }

    # ====================================================================
    # 2. 建立 CP-SAT 模型
    # ====================================================================
    model = cp_model.CpModel()

    # --- 決策變數 ---
    # total_makespan: 整個排程的總完工時間，這是我們希望最小化的目標之一。
    total_makespan = model.NewIntVar(0, upper_bound_makespan, 'total_makespan')
    # pm_starts: 每台機器進行預防性維護的開始時間，這是模型需要決策的關鍵變數。
    pm_starts = {k: model.NewIntVar(0, upper_bound_makespan, f'pm_start_m_{k}') for k in range(num_machines)}
    
    # --- 成本變數 ---
    # 這些變數用於構建總成本函數，它們的上限設定為sys.maxsize，確保不會因為溢出而導致無解。
    large_upper_bound = sys.maxsize
    total_cost_var = model.NewIntVar(0, large_upper_bound, 'total_cost')
    makespan_cost = model.NewIntVar(0, large_upper_bound, 'makespan_cost')
    pm_cost = model.NewIntVar(0, large_upper_bound, 'pm_cost')
    failure_cost = model.NewIntVar(0, large_upper_bound, 'failure_cost')

    # --- 約束條件 ---
    # 1. Makespan 約束：總完工時間必須大於所有生產作業和 PM 活動的結束時間。
    max_op_end_time = max(op[4] for op in schedule_from_drl)
    model.Add(total_makespan >= max_op_end_time)
    for k in range(num_machines):
        model.Add(total_makespan >= pm_starts[k] + cost_params['Tp_k'][k])

    # 2. 無重疊約束：PM 活動不能與任何生產作業重疊。
    for k in range(num_machines):
        # 為 PM 活動創建一個區間變數。
        pm_interval = model.NewIntervalVar(
            pm_starts[k], cost_params['Tp_k'][k], pm_starts[k] + cost_params['Tp_k'][k], f'pm_interval_m_{k}'
        )
        # 收集該機器上所有生產作業的區間變數。
        job_intervals = [
            model.NewFixedSizeIntervalVar(s, e - s, f'job_{j[0]}_{j[1]}_{k}')
            for j in schedule_from_drl if j[2] == k
        ]
        # 添加無重疊約束，確保 PM 和生產作業不會同時進行。
        model.AddNoOverlap(job_intervals + [pm_interval])

    # --- 成本計算 (線性模型) ---
    # 1. PM 固定成本：每台機器進行一次 PM 的固定費用。
    model.Add(pm_cost == sum(cost_params['Cp_k']))

    # 2. Makespan 成本：總完工時間帶來的成本，與makespan成正比。
    # 使用AddMultiplicationEquality來正確處理變數與常數的乘法。
    w_makespan_cost_int = int(cost_params['w_makespan_cost'] * 100) # 放大100倍以保持整數精度
    model.AddMultiplicationEquality(makespan_cost, [total_makespan, model.NewConstant(w_makespan_cost_int)])

    # 3. 預期故障成本 (線性模型)：
    # 故障成本與機器運轉時間成正比，PM 會重置運轉時間。
    machine_failure_costs = []
    for k in range(num_machines):
        # 成本係數，同樣放大以保持整數精度。
        # 這裡使用簡化的固定係數，如果需要，可以根據dynamic_params動態計算
        cost_factor = int(0.01 * cost_params['Cf_k'][k] * 100) # 簡化為常數

        # PM 前的故障成本：與PM開始時間成正比。
        cost_before_pm = model.NewIntVar(0, large_upper_bound, f'cost_before_pm_m{k}')
        model.AddMultiplicationEquality(cost_before_pm, [pm_starts[k], model.NewConstant(cost_factor)])

        # PM 後的故障成本：與PM結束到總完工時間之間的剩餘時間成正比。
        remaining_time = model.NewIntVar(0, upper_bound_makespan, f'remaining_time_m{k}')
        model.Add(remaining_time == total_makespan - pm_starts[k] - cost_params['Tp_k'][k])
        cost_after_pm = model.NewIntVar(0, large_upper_bound, f'cost_after_pm_m{k}')
        model.AddMultiplicationEquality(cost_after_pm, [remaining_time, model.NewConstant(cost_factor)])

        machine_failure_costs.append(cost_before_pm)
        machine_failure_costs.append(cost_after_pm)

    model.Add(failure_cost == sum(machine_failure_costs))

    # 4. 總成本：所有成本項的加總。
    # 注意：pm_cost 和 makespan_cost 已經被放大100倍，所以這裡直接加總。
    scaled_pm_cost = model.NewIntVar(0, large_upper_bound, 'scaled_pm_cost')
    model.AddMultiplicationEquality(scaled_pm_cost, [pm_cost, model.NewConstant(100)]) # 放大100倍以匹配其他成本的尺度
    model.Add(total_cost_var == scaled_pm_cost + failure_cost + makespan_cost)
    
    # 設定優化目標：最小化總成本。
    model.Minimize(total_cost_var)

    # ====================================================================
    # 3. 執行求解器並產生圖表
    # ====================================================================
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print('成功找到最佳解！')
        # 由於成本被放大100倍，這裡除以100以顯示實際成本。
        print(f'最小總成本: {solver.ObjectiveValue() / 100}')
        final_makespan = solver.Value(total_makespan)
        print(f'最佳總完工時間 (Makespan): {final_makespan}')

        # 準備數據用於甘特圖繪製
        job_shop = JobShop(recalculate_makespan=False)
        job_shop.add_machines(num_machines)
        num_jobs = max(op[0] for op in schedule_from_drl) + 1
        for i in range(num_jobs):
            job_shop.add_job(Job(job_id=i))

        for job_info in schedule_from_drl:
            job_id, op_id, machine_id, start_time, end_time = job_info
            op = Operation(job_id, op_id, machine_id=machine_id)
            op.scheduling_information['start_time'] = start_time
            op.scheduling_information['end_time'] = end_time
            job_shop.machines[machine_id]._processed_operations.append(op)

        pm_schedule = []
        for k in range(num_machines):
            pm_start_time = solver.Value(pm_starts[k])
            pm_duration = cost_params['Tp_k'][k]
            print(f'機器 {k} 的最佳 PM 開始時間: {pm_start_time}')
            pm_schedule.append((k, pm_start_time, pm_duration))
        
        plt = gantt_with_pm.plot(job_shop, pm_schedule)
        plt.xlim(0, final_makespan + 5) # 設定X軸範圍，稍微超出makespan
        plt.savefig('gantt_with_pm.png')
        print("已生成甘特圖並儲存為 gantt_with_pm.png")

    else:
        print(f'模型無法求解。狀態: {solver.StatusName(status)}')

if __name__ == '__main__':
    schedule_pm_and_plot()
