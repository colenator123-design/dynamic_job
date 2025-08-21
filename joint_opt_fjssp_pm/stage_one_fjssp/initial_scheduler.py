
import sys
import os

# 將專案根目錄添加到 Python 路徑中
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from solution_methods.FJSP_DRL_Custom.run_FJSP_DRL import run_FJSP_DRL
from solution_methods.helper_functions import load_parameters, load_job_shop_env
from scheduling_environment.jobShop import JobShop

# FJSP_DRL 的配置檔案路徑
FJSP_DRL_PARAM_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs", "FJSP_DRL_Custom.toml")

class InitialScheduler:
    def __init__(self, jobs, machines, config):
        """
        初始化第一階段的排程器。

        參數:
            jobs (list): 工件物件列表。
            machines (list): 機器物件列表。
            config (dict): 包含模型參數的字典。
        """
        self.jobs = jobs
        self.machines = machines
        self.config = config

    def solve(self):
        """
        使用 FJSP_DRL 演算法來求解 FJSSP 問題，
        以產生一個初始的作業排程。

        返回:
            tuple: 一個包含排程結果和目標值的元組。
                   (schedule, objective_value)
        """
        print("第一階段：正在使用 FJSP_DRL 演算法產生初始排程...")

        # 載入 FJSP_DRL 的參數
        fjsp_drl_params = load_parameters(FJSP_DRL_PARAM_FILE)

        # 載入 FJSP_DRL 所需的 jobShopEnv 實例
        # 這裡假設 FJSP_DRL.toml 中的 problem_instance 路徑是正確的
        initial_jobShopEnv = load_job_shop_env(fjsp_drl_params['test_parameters']['problem_instance'])

        # 執行 FJSP_DRL 演算法
        makespan, scheduled_jobShopEnv = run_FJSP_DRL(initial_jobShopEnv, **fjsp_drl_params)

        # 從 scheduled_jobShopEnv 中提取排程資訊，轉換為 JointOptimizer 所需的格式
        initial_schedule = {}
        for machine in scheduled_jobShopEnv.machines:
            for operation in machine.scheduled_operations:
                op_id = f"J{operation.job_id}_O{operation.operation_id}"
                initial_schedule[op_id] = {
                    'id': op_id,
                    'machine': operation.scheduled_machine,
                    'start_time': operation.scheduled_start_time,
                    'end_time': operation.scheduled_end_time
                }
        
        objective_value = makespan # FJSP_DRL 主要優化 makespan

        print(f"第一階段完成。初始目標值 (Makespan): {objective_value}")

        return initial_schedule, objective_value

