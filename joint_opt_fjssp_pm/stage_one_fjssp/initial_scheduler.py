import sys
import os

# 將專案根目錄添加到 Python 路徑中
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from joint_opt_fjssp_pm.solution_methods.hybrid_dspo_vns import HybridDspoVns

class InitialScheduler:
    def __init__(self, jobs, machines, config):
        """
        初始化第一階段的排程器。
        現在使用 HybridDspoVns 演算法。

        參數:
            jobs (list): 工件物件列表。
            machines (list): 機器物件列表。
            config (object): 包含所有參數的設定物件。
        """
        self.jobs = jobs
        self.machines = machines
        self.config = config

    def solve(self):
        """
        使用 HybridDspoVns 演算法來求解 FJSSP 問題，
        以產生一個初始的作業排程。

        返回:
            tuple: 一個包含排程結果和目標值的元組。
                   (schedule, objective_value)
        """
        print("第一階段：正在使用 Hybrid DPSO-VNS 演算法產生初始排程...")

        # 初始化演算法，傳入完整的設定物件
        dspo_vns_solver = HybridDspoVns(
            jobs=self.jobs,
            machines=self.machines,
            config=self.config
        )

        # 執行第一階段求解
        initial_schedule, objective_value = dspo_vns_solver.run_stage_one()

        print(f"第一階段完成。初始目標值 (TTC+TBC): {objective_value}")

        return initial_schedule, objective_value