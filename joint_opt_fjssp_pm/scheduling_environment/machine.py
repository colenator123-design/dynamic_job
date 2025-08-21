
import numpy as np

class Machine:
    def __init__(self, machine_id, maintenance_params=None):
        """
        初始化一個機器 (Machine) 物件。

        參數:
            machine_id (int): 機器的唯一標識符。
            maintenance_params (dict, optional): 維護相關的參數，例如：
                'weibull_beta' (float): Weibull 分布的形狀參數 (β)。
                'weibull_eta' (float): Weibull 分布的尺度參數 (η)。
                'pm_cost' (float): 預防性維護的成本。
                'pm_duration' (int): 預防性維護的持續時間。
                預設為 None。
        """
        self.machine_id = machine_id
        self.maintenance_params = maintenance_params if maintenance_params is not None else {}
        self.scheduled_operations = []
        self.pm_activities = []
        self.workload = 0

    def __repr__(self):
        return f"Machine(ID={self.machine_id}, Workload={self.workload})"

    def add_operation(self, operation):
        """
        向機器排程中添加一個工序。
        """
        self.scheduled_operations.append(operation)
        self.workload += (operation.end_time - operation.start_time)
        # 保持工序按開始時間排序
        self.scheduled_operations.sort(key=lambda op: op.start_time)

    def get_failure_rate(self, t, operating_conditions=None):
        """
        根據 Weibull 分布計算在時間 t 的故障率。
        可選地，可以考慮動態操作條件。
        """
        beta = self.maintenance_params.get('weibull_beta', 2.0) # 預設為 2.0
        eta = self.maintenance_params.get('weibull_eta', 100.0) # 預設為 100.0

        # 根據論文，這裡可以擴展以包含操作條件的影響
        # 例如，使用 Proportional Hazard Model (PHM)
        # lambda(t|X) = lambda_0(t) * exp(gamma * X)
        # 此處為簡化版，僅使用基本的 Weibull 故障率

        return (beta / eta) * (t / eta)**(beta - 1)

    def schedule_pm(self, start_time, duration):
        """
        安排一次預防性維護活動。
        """
        self.pm_activities.append({'start_time': start_time, 'end_time': start_time + duration})
