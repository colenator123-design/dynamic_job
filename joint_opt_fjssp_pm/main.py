
import config
from data_loader import DataLoader
from scheduling_environment.job import Job
from scheduling_environment.machine import Machine
from scheduling_environment.operation import Operation
from stage_one_fjssp.initial_scheduler import InitialScheduler
from stage_two_fjssp_pm.joint_optimizer import JointOptimizer

def main():
    """
    專案主執行函數。
    """
    print("開始執行 FJSSP-PM 聯合優化...")

    # 1. 載入數據
    # 假設論文的數據表格已經轉換為 CSV 檔案
    loader = DataLoader(table_a1_path='data/table_a1.csv', table_a2_path='data/table_a2.csv')
    maintenance_data = loader.load_maintenance_data()
    production_data, beta_vectors = loader.load_production_data()

    # 2. 初始化環境
    jobs = []
    for job_id, job_info in production_data.items():
        # 根據您的指示，我們使用 makespan 作為目標，因此忽略 due_date 和 tardiness_penalty
        job = Job(job_id=job_id)
        for op_data in job_info['operations']:
            op = Operation(job_id=job_id, op_id=op_data['op_id'], candidate_machines=op_data['candidate_machines'])
            job.add_operation(op)
        jobs.append(job)

    machines = []
    for machine_id, maint_info in maintenance_data.items():
        machine = Machine(machine_id=machine_id, maintenance_params=maint_info)
        machines.append(machine)

    # 3. 執行第一階段：初始排程
    initial_scheduler = InitialScheduler(jobs, machines, config.COST_PARAMS)
    initial_schedule, initial_objective = initial_scheduler.solve()

    # 4. 執行第二階段：聯合優化
    joint_optimizer = JointOptimizer(initial_schedule, jobs, machines, config.COST_PARAMS)
    final_schedule, final_objective = joint_optimizer.solve()

    # 5. 輸出最終結果
    print("\n優化完成！")
    print(f"最終排程: {final_schedule}")
    print(f"最終目標函數值: {final_objective}")

if __name__ == "__main__":
    main()

