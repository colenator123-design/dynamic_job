
# DPSO-VNS 演算法參數
POPULATION_SIZE = 50
MAX_ITERATIONS_DPSO = 100
MAX_ITERATIONS_VNS = 50
MAX_ITERATIONS_FEEDBACK = 10 # 第二階段回饋更新的最大次數

# 成本參數 (以 Makespan 為主要目標)
COST_PARAMS = {
    'w_makespan': 1.0,                      # Makespan 的權重
    'workload_balancing_penalty_weight': 500, # 負載平衡懲罰權重 (Cb)
    'pm_cost_fixed': 100,                     # PM 固定成本 (Cp_k)
    'failure_cost_fixed': 1000,                   # 故障成本 (Cf_k)
    'pm_duration': 5,                         # PM 維護時間 (Tp_k)
}

# 維護參數 (用於動態 Weibull 模型)
MAINTENANCE_PARAMS = {
    'weibull_beta': 2.0,     # Weibull 分布的形狀參數 (β)
    'weibull_eta_0': 50.0, # Weibull 分布的初始尺度參數 (η₀)
    'alpha_uk': 0.1,     # 利用率影響因子 (α_uk)
    'alpha_sk': 0.05,     # 切換影響因子 (α_sk)
}

# 模擬的生產數據 (用於測試)
# 在實際應用中，這些數據應該從外部檔案讀取
NUM_JOBS = 3
NUM_MACHINES = 3

# 每個工件的工序數量
OPERATIONS_PER_JOB = {0: 3, 1: 3, 2: 3}

# 每個工序的候選機器和加工時間
# 格式: (job_id, op_id): {machine_id: processing_time, ...}
CANDIDATE_MACHINES = {
    (0, 0): {0: 10},
    (0, 1): {1: 15},
    (0, 2): {0: 15},
    (1, 0): {1: 10},
    (1, 1): {2: 10},
    (1, 2): {1: 10},
    (2, 0): {2: 10},
    (2, 1): {0: 15},
    (2, 2): {2: 10},
}
