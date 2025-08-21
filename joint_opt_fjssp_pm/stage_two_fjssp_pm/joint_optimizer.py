
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
        參考 ga_solver.py 的邏輯，計算每台機器的最佳 PM 計畫。
        返回一個字典，key 為 machine_id，value 為 PM 計畫。
        """
        pm_plans = {}
        for machine in self.machines:
            # 1. 計算機器的動態參數 (利用率、切換次數)
            # 這些需要從當前的 schedule 中計算得出
            utilization, switch_count = self._get_dynamic_params(machine.machine_id)

            # 2. 根據論文公式 (16)-(18) 和 ga_solver.py 的邏輯，找到最佳 PM 間隔 Trk
            # 這是一個優化問題，需要找到 Trk 來最小化 Vrk
            # 為了簡化，我們先假設一個固定的 PM 間隔
            optimal_pm_interval = 100 # 這裡應該是一個計算結果

            # 3. 確定 PM 的時間點
            # 這裡需要遍歷機器的排程，找到插入 PM 的最佳位置
            # 暫時先假設在排程中間進行
            pm_start_time = machine.workload / 2

            pm_plans[machine.machine_id] = {
                'interval': optimal_pm_interval,
                'start_time': pm_start_time,
                'duration': self.config['pm_duration']
            }
        return pm_plans

    def _get_dynamic_params(self, machine_id):
        """
        計算指定機器的利用率和作業切換次數。
        """
        # 這裡需要根據 self.schedule 來計算
        # 暫時返回模擬值
        return 0.5, 10 # (utilization, switch_count)

    def _get_failure_rate_integral(self, start_t, end_t, machine_id, utilization, switch_count):
        """
        計算動態 Weibull 分布在特定時間區間的累積故障風險。
        完全參考 ga_solver.py 的實作。
        """
        # 從 config 中獲取參數
        beta = self.config['weibull_beta']
        eta_0 = self.config['weibull_eta_0']
        alpha_uk = self.config['alpha_uk']
        alpha_sk = self.config['alpha_sk']

        # 根據論文公式計算動態尺度參數 η_rk
        eta_dynamic = eta_0 / (1 + alpha_uk * utilization + alpha_sk * switch_count)
        
        if eta_dynamic <= 0:
            return float('inf')

        # 計算 Weibull 分布的累積故障率積分
        integral_end = (end_t / eta_dynamic)**beta if end_t >= 0 else 0
        integral_start = (start_t / eta_dynamic)**beta if start_t >= 0 else 0
        
        return integral_end - integral_start

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
        根據論文演算法 2，重新排程受影響的作業。
        """
        # 1. 識別所有受衝突影響的作業
        affected_ops = set()
        for conflict in conflicts:
            affected_ops.add(conflict['conflicting_op'])
            # 這裡還需要找出所有在衝突作業之後的、有依賴關係的作業

        # 2. 從當前排程中移除這些作業
        reschedule_queue = []
        remaining_schedule = self.schedule.copy()
        for op_id in affected_ops:
            if op_id in remaining_schedule:
                reschedule_queue.append(remaining_schedule.pop(op_id))

        # 3. 使用 VNS 進行局部優化
        # 這裡需要一個修改版的 VNS，只對 reschedule_queue 中的作業進行操作
        # 這是一個非常複雜的步驟，我們先用一個簡化的邏輯代替
        
        # 簡化邏輯：將受影響的作業簡單地推遲
        for op in reschedule_queue:
            pm_duration = self.config['pm_duration']
            op['start_time'] += pm_duration
            op['end_time'] += pm_duration
            remaining_schedule[op['id']] = op # 將作業重新加回排程

        self.schedule = remaining_schedule
        print(f"  簡化版重新排程完成，{len(reschedule_queue)} 個作業被推遲。")

    def _apply_pm_to_schedule(self, pm_plans):
        """
        將無衝突的 PM 活動應用到最終排程中。
        """
        pass

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


