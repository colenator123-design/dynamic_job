
import sys
import os
import numpy as np

# 將專案根目錄添加到 Python 路徑中
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class JointOptimizer:
    def __init__(self, initial_schedule, jobs, machines, config):
        """
        初始化第二階段的聯合優化器。

        參數:
            initial_schedule (dict): 第一階段產生的初始排程。
            jobs (list): 工件物件列表。
            machines (list): 機器物件列表。
            config (dict): 包含模型參數的字典。
        """
        self.schedule = initial_schedule
        self.jobs = jobs
        self.machines = machines
        self.config = config

    def solve(self):
        """
        執行第二階段的聯合優化，整合預防性維護。
        """
        print("\n第二階段：開始進行生產與維護的聯合優化...")

        # 這裡將實作論文中的回饋更新策略 (feedback-updated strategy)
        # 演算法將迭代執行，直到生產和維護之間沒有衝突

        final_schedule, final_cost = self._feedback_update_strategy()

        print(f"第二階段完成。最終總成本 (TTC + TBC + TMC): {final_cost}")
        return final_schedule, final_cost

    def _feedback_update_strategy(self):
        """
        實作論文中的核心回饋更新策略。
        """
        iteration = 0
        max_iterations = self.config.get('max_iterations_feedback', 10)

        while iteration < max_iterations:
            print(f"  回饋更新迭代: {iteration + 1}")

            # 1. 根據當前排程，計算每台機器的動態故障率和最佳 PM 計畫
            optimal_pm_plans = self._calculate_optimal_pm()

            # 2. 檢查 PM 計畫是否與生產排程衝突
            conflicts = self._check_for_conflicts(optimal_pm_plans)

            if not conflicts:
                print("  找到無衝突的排程，優化結束。")
                # 將 PM 活動加入最終排程
                self._apply_pm_to_schedule(optimal_pm_plans)
                break

            # 3. 如果有衝突，則根據論文的演算法 2 進行重新排程
            print(f"  發現 {len(conflicts)} 個衝突，進行重新排程...")
            self._reschedule_operations(conflicts)

            iteration += 1
        
        # 計算最終的總成本
        final_cost = self._calculate_total_cost()
        return self.schedule, final_cost

    def _calculate_optimal_pm(self):
        """
        根據論文公式 (16)-(18)，計算每台機器的最佳 PM 計畫。
        使用數值搜索方法找到最小化 Vrk 的最佳 PM 間隔 Trk。
        返回一個字典，key 為 machine_id，value 為 PM 計畫。
        """
        pm_plans = {}
        makespan = max(op['end_time'] for op in self.schedule.values()) if self.schedule else 0

        for machine in self.machines:
            # 1. 計算機器的動態參數
            utilization, switch_count = self._get_dynamic_params(machine.machine_id)
            
            best_interval = -1
            min_vrk = float('inf')

            # 2. 數值搜索最佳 PM 間隔 Trk
            # 搜索範圍從 1 到 makespan
            for t_interval in range(1, int(makespan) + 1):
                vrk = self._calculate_vrk_objective(t_interval, machine, utilization, switch_count)
                if vrk < min_vrk:
                    min_vrk = vrk
                    best_interval = t_interval

            if best_interval > 0:
                # 論文中的 t*rk，即最佳 PM 時間點
                # 簡化處理：假設維護週期從 0 開始，所以最佳時間點就是最佳間隔
                optimal_pm_time_point = best_interval
                pm_plans[machine.machine_id] = {
                    'interval': best_interval,
                    'start_time': optimal_pm_time_point,
                    'duration': machine.maintenance_params['pm_duration']
                }
        return pm_plans

    def _calculate_vrk_objective(self, trk, machine, utilization, switch_count):
        """
        根據論文公式 (16)-(18) 計算 Vrk 的值。
        """
        m_params = machine.maintenance_params
        cp_k = m_params['pm_cost']
        cf_k = m_params['mr_cost']
        tp_k = m_params['pm_duration']
        tf_k = m_params['mr_duration']

        # 計算累積故障率的積分
        integral_lambda = self._get_failure_rate_integral(0, trk, machine, utilization, switch_count)

        # 計算 VC_rk (維護成本率)
        vc_rk_numerator = cp_k + cf_k * integral_lambda
        vc_rk_denominator = trk + tp_k + tf_k * integral_lambda
        vc_rk = vc_rk_numerator / vc_rk_denominator if vc_rk_denominator != 0 else float('inf')

        # 計算 VA_rk (可用性)
        va_rk_numerator = trk
        va_rk_denominator = trk + tp_k + tf_k * integral_lambda
        va_rk = va_rk_numerator / va_rk_denominator if va_rk_denominator != 0 else 0

        # 論文中提到 VC_rk* 和 VA_rk* 是理論上的最優值，這裡我們用一個簡化的方式
        # 我們直接最小化 Vrk = w1 * VC_rk - w2 * VA_rk (w1=w2=0.5)
        # 這等價於最小化 VC_rk - VA_rk
        vrk = vc_rk - va_rk
        return vrk

    def _get_failure_rate_integral(self, start_t, end_t, machine, utilization, switch_count):
        """
        計算動態 Weibull 分布在特定時間區間的累積故障風險。
        """
        m_params = machine.maintenance_params
        theta = m_params['theta'] # Shape parameter (β in paper)
        eta = m_params['eta']   # Scale parameter (η in paper)
        
        # 論文中沒有明確給出 f(βX) 的形式，但從 ga_solver.py 的邏輯推斷
        # 影響因子是作用在尺度參數 eta 上的
        # 這裡我們假設 f(βX) 是對 eta 的一個修正
        # 為了簡化，我們直接使用 machine 的固有參數，忽略 X 的影響
        # 在更完整的實作中，這裡應該加入 beta_vectors 的影響

        if eta <= 0:
            return float('inf')

        # 計算 Weibull 分布的累積故障率積分 H(t) = (t/η)^θ
        integral_end = (end_t / eta)**theta if end_t >= 0 else 0
        integral_start = (start_t / eta)**theta if start_t >= 0 else 0
        
        return integral_end - integral_start

    def _get_dynamic_params(self, machine_id):
        """
        計算指定機器的利用率和作業切換次數。
        """
        machine_ops = [op for op in self.schedule.values() if op['machine'] == machine_id]
        if not machine_ops:
            return 0, 0

        makespan = max(op['end_time'] for op in self.schedule.values())
        if makespan == 0:
            return 0, 0

        # 計算總加工時間
        total_processing_time = sum(op['end_time'] - op['start_time'] for op in machine_ops)
        
        # 計算利用率
        utilization = total_processing_time / makespan

        # 計算切換次數 (簡化為工序數量 - 1)
        switch_count = len(machine_ops) - 1 if len(machine_ops) > 1 else 0

        return utilization, switch_count

    def _check_for_conflicts(self, pm_plans):
        """
        檢查 PM 計畫是否與當前的生產排程衝突。
        返回一個包含衝突資訊的列表。
        """
        conflicts = []
        for machine_id, pm_plan in pm_plans.items():
            pm_start = pm_plan['start_time']
            pm_end = pm_start + pm_plan['duration']

            for op_id, op_info in self.schedule.items():
                if op_info['machine'] == machine_id:
                    op_start = op_info['start_time']
                    op_end = op_info['end_time']
                    
                    # 檢查是否有重疊
                    if max(pm_start, op_start) < min(pm_end, op_end):
                        conflicts.append({
                            'machine_id': machine_id,
                            'pm_plan': pm_plan,
                            'conflicting_op': op_id
                        })
        return conflicts

    def _reschedule_operations(self, conflicts):
        """
        根據論文演算法 2 的思想，重新排程受影響的作業。
        採用貪婪插入的啟發式方法進行重新排程。
        """
        # 1. 確定所有受影響的工序
        affected_op_keys = set()
        pm_windows = {c['machine_id']: [] for c in conflicts}
        for c in conflicts:
            affected_op_keys.add(c['conflicting_op'])
            # 記錄 PM 時間窗口
            pm_plan = c['pm_plan']
            pm_windows[c['machine_id']].append((pm_plan['start_time'], pm_plan['start_time'] + pm_plan['duration']))

        # 找出所有後續工序
        ops_to_reschedule = set()
        for job_id, op_id in affected_op_keys:
            job_ops = self.jobs[job_id].operations
            for i in range(op_id, len(job_ops)):
                ops_to_reschedule.add((job_id, i))

        # 2. 從排程中移除這些工序，並按 job_id, op_id 排序
        reschedule_queue = sorted(list(ops_to_reschedule))
        for key in reschedule_queue:
            if key in self.schedule:
                del self.schedule[key]

        # 3. 貪婪插入重新排程
        # 重新計算 job 和 machine 的可用時間
        machine_end_times = {m.machine_id: 0 for m in self.machines}
        job_end_times = {j.job_id: 0 for j in self.jobs}
        for key, op_info in self.schedule.items():
            machine_end_times[op_info['machine']] = max(machine_end_times[op_info['machine']], op_info['end_time'])
            job_end_times[key[0]] = max(job_end_times[key[0]], op_info['end_time'])

        for job_id, op_id in reschedule_queue:
            operation = self.jobs[job_id].operations[op_id]
            # 假設機器分配不變
            # 在更複雜的實作中，這裡可以重新選擇機器
            assigned_machine = None
            for key, op_info in self.initial_schedule.items(): # 從初始排程中找回機器分配
                if key == (job_id, op_id):
                    assigned_machine = op_info['machine']
                    break
            
            if assigned_machine is None: continue

            processing_time = operation.candidate_machines[assigned_machine]
            
            # 找到最早的可用開始時間
            earliest_start_time = max(machine_end_times[assigned_machine], job_end_times[job_id])

            # 檢查 PM 衝突
            is_valid_slot = False
            while not is_valid_slot:
                is_valid_slot = True
                end_time = earliest_start_time + processing_time
                if assigned_machine in pm_windows:
                    for pm_start, pm_end in pm_windows[assigned_machine]:
                        # 如果工序與 PM 窗口重疊，則將工序推遲到 PM 結束後
                        if max(earliest_start_time, pm_start) < min(end_time, pm_end):
                            earliest_start_time = pm_end
                            is_valid_slot = False
                            break # 重新檢查新的時間點
            
            new_end_time = earliest_start_time + processing_time

            # 更新排程和時間
            self.schedule[(job_id, op_id)] = {
                'machine': assigned_machine,
                'start_time': earliest_start_time,
                'end_time': new_end_time
            }
            machine_end_times[assigned_machine] = new_end_time
            job_end_times[job_id] = new_end_time

        print(f"  重新排程完成，{len(reschedule_queue)} 個工序被重新安排。")

    def _apply_pm_to_schedule(self, pm_plans):
        """
        將無衝突的 PM 活動作為特殊標記應用到最終排程中。
        """
        for machine_id, pm_plan in pm_plans.items():
            pm_key = (f"PM_M{machine_id}", 0)
            self.schedule[pm_key] = {
                'machine': machine_id,
                'start_time': pm_plan['start_time'],
                'end_time': pm_plan['start_time'] + pm_plan['duration']
            }

    def _calculate_total_cost(self):
        """
        計算最終的總成本 (Makespan_Cost + TBC + TMC)。
        """
        # 1. 計算 Makespan 和 TBC
        makespan = max(op['end_time'] for op in self.schedule.values())
        makespan_cost = self.config['w_makespan'] * makespan

        machine_workloads = {m.machine_id: 0 for m in self.machines}
        for op_info in self.schedule.values():
            duration = op_info['end_time'] - op_info['start_time']
            machine_workloads[op_info['machine']] += duration
        
        avg_workload = sum(machine_workloads.values()) / len(self.machines)
        workload_variance = sum((wl - avg_workload)**2 for wl in machine_workloads.values()) / len(self.machines)
        tbc = self.config['workload_balancing_penalty_weight'] * workload_variance

        # 2. 計算總維護成本 (TMC)
        total_maintenance_cost = 0
        for machine in self.machines:
            # a. PM 固定成本
            # 假設每台機器只做一次 PM
            total_maintenance_cost += self.config['pm_cost_fixed']

            # b. 預期故障成本
            # 這裡需要根據 PM 的時間點，分段計算故障風險
            utilization, switch_count = self._get_dynamic_params(machine.machine_id)
            pm_plan = self._calculate_optimal_pm().get(machine.machine_id)
            pm_start_time = pm_plan['start_time'] if pm_plan else float('inf')

            # PM 前的風險
            risk_before_pm = self._get_failure_rate_integral(0, pm_start_time, machine.machine_id, utilization, switch_count)
            # PM 後的風險 (假設 PM 後設備 "as good as new")
            time_after_pm = max(0, makespan - (pm_start_time + self.config['pm_duration']))
            risk_after_pm = self._get_failure_rate_integral(0, time_after_pm, machine.machine_id, utilization, switch_count)

            failure_cost = self.config['failure_cost_fixed'] * (risk_before_pm + risk_after_pm)
            total_maintenance_cost += failure_cost

        return makespan_cost + tbc + total_maintenance_cost


