# -*- coding: utf-8 -*-
"""
檔名: stage2_ga_pm_pareto_plus.py

你要的重點全都整合：
  1) 兩目標分離：NSGA-II 最小化 (total_cost, makespan)，且 total_cost 內不再含 makespan 權重。
  2) PHM/Weibull + 作業條件 f(X)：利用率、切換率，且可選每道工序特徵 (op_features) 與 alpha_feat 權重；採分段積分，PM 後 as-good-as-new。
  3) PM 與作業衝突：advance/postpone 以「PM 前期望失敗數」比較風險（比 t* 偏差更合理）。
  4) 多次 PM：每機器 0～K 次（固定長度 + -1 mask）。
  5) 故障（MR）停機：期望失敗數 × 平均修復時長 Td_k 直接插入時間線，會推遲後續作業並影響 makespan/風險。
  6) TBC：提供 std / cv / maxdev 三種定義可選。

使用：
  pip install deap numpy
  python stage2_ga_pm_pareto_plus.py

接上 Stage-I：
  - 把 generate_random_job_shop_schedule(...) 換成你自己的 production_schedule：
      production_schedule = [(job_id, op_id, machine_id, start, end), ...]
  - 若每道工序有特徵，填 op_features[(job, op, m)] = {"load":0.7, "temp":0.4, ...}
    並在 dynamic_params["alpha_feat"] 指定各特徵權重（名稱需一致）。
"""

import os
import sys
import random
import numpy as np
from typing import List, Tuple, Dict, Optional

# 若需用你的 JobShop/gantt 模組作圖，將專案根目錄（上一層）加入 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HAS_ENV_MODULES = True
try:
    from scheduling_environment.job import Job
    from scheduling_environment.operation import Operation
    from scheduling_environment.machine import Machine
    from scheduling_environment.jobShop import JobShop
    from visualization import gantt_with_pm
except Exception:
    HAS_ENV_MODULES = False

from deap import base, creator, tools, algorithms

# ========================== 可調參數 ==========================
KMAX_PM_PER_MACHINE = 2      # 每台機器最多 PM 次數（0~K；-1 代表該槽停用）
P_GENE_OFF = 0.25            # 初始化/突變時把某 PM 槽關掉 (-1) 的機率
TBC_MODE = "std"             # "std" | "cv" | "maxdev"
RANDOM_SEED = 42
# ============================================================


# ====================================================================
# 1) Stage-I 替身：生成示範用初始排程與參數（實作時請換成你的輸入）
# ====================================================================
def generate_random_job_shop_schedule(num_jobs: int, num_machines: int,
                                      max_duration: int = 15,
                                      seed: Optional[int] = None) -> List[Tuple[int,int,int,int,int]]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    production_schedule = []
    machine_end_times = [0] * num_machines
    job_end_times = [0] * num_jobs

    for j in range(num_jobs):
        op_sequence = list(range(num_machines))
        random.shuffle(op_sequence)
        for o, m in enumerate(op_sequence):
            op_duration = random.randint(5, max_duration)
            start_time = max(machine_end_times[m], job_end_times[j])
            end_time = start_time + op_duration
            production_schedule.append((j, o, m, start_time, end_time))
            machine_end_times[m] = end_time
            job_end_times[j] = end_time

    production_schedule.sort(key=lambda x: x[3])
    return production_schedule


def generate_random_params(num_machines: int, seed: Optional[int] = None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    dynamic_params = {
        "utilization": [random.uniform(0.3, 0.6) for _ in range(num_machines)],
        "alpha_uk": [random.uniform(0.1, 0.2) for _ in range(num_machines)],
        "alpha_sk": [random.uniform(0.05, 0.1) for _ in range(num_machines)],
        "alpha_feat": {},  # 例如 {"load":0.6, "temp":0.4}；若無可留空
    }

    cost_params = {
        "Cp_k": [random.randint(80, 150) for _ in range(num_machines)],      # PM 成本
        "Cf_k": [random.randint(800, 1500) for _ in range(num_machines)],     # 故障成本
        "Tp_k": [random.randint(4, 8) for _ in range(num_machines)],          # PM 時長
        "Td_k": [random.uniform(2.0, 6.0) for _ in range(num_machines)],      # MR 平均修復停機時長（用於插段）
        "beta": [random.uniform(1.6, 2.8) for _ in range(num_machines)],      # Weibull 形狀
        "eta_0": [random.uniform(40.0, 70.0) for _ in range(num_machines)],   # Weibull 尺度（基準）
    }
    return dynamic_params, cost_params


# ====================================================================
# 2) 工具：TBC（3 模式）、f(X) 與 PHM 分段積分、advance/postpone 決策
# ====================================================================
def compute_TBC(machine_ops: Dict[int, List[Tuple[int,int,int,int,int]]], mode: str = "std", Cb: float = 500.0) -> float:
    workloads = [sum(e - s for (_, _, _, s, e) in ops) for ops in machine_ops.values()]
    K = len(workloads)
    if K == 0:
        return 0.0
    mean = sum(workloads) / K
    if mode == "std":
        variance = sum((w - mean) ** 2 for w in workloads) / K
        return Cb * (variance ** 0.5)
    elif mode == "cv":
        variance = sum((w - mean) ** 2 for w in workloads) / K
        std = variance ** 0.5
        return Cb * (std / max(1e-6, mean))
    elif mode == "maxdev":
        return Cb * max(abs(w - mean) for w in workloads)
    else:
        variance = sum((w - mean) ** 2 for w in workloads) / K
        return Cb * (variance ** 0.5)


def window_switch_rate(ops_on_machine: List[Tuple[int,int,int,int,int]], a: float, b: float) -> float:
    ops = [(j,o,m,s,e) for (j,o,m,s,e) in ops_on_machine if not (e <= a or s >= b)]
    ops.sort(key=lambda x: x[3])
    if len(ops) <= 1:
        return 0.0
    switches = 0
    last_job = ops[0][0]
    for (j,_,_,_,_) in ops[1:]:
        if j != last_job:
            switches += 1
        last_job = j
    return switches / max(1.0, b - a)


def mean_features(ops_on_machine: List[Tuple[int,int,int,int,int]], a: float, b: float,
                  op_features: Dict[Tuple[int,int,int], Dict[str,float]]) -> Dict[str, float]:
    accum = {}; total = 0.0
    for (j,o,m,s,e) in ops_on_machine:
        if e <= a or s >= b: 
            continue
        dur = max(0.0, min(e, b) - max(s, a))
        if dur <= 0: 
            continue
        feats = op_features.get((j,o,m), {})
        for k, v in feats.items():
            accum[k] = accum.get(k, 0.0) + v * dur
        total += dur
    if total <= 0: 
        return {}
    return {k: v/total for k, v in accum.items()}


def f_cov_factor(a: float, b: float, ops_on_machine: List[Tuple[int,int,int,int,int]],
                 util: float, a_u: float, a_s: float,
                 alpha_feat: Dict[str, float],
                 op_features: Optional[Dict[Tuple[int,int,int], Dict[str,float]]] = None) -> float:
    sr = window_switch_rate(ops_on_machine, a, b)
    f = 1.0 + a_u * util + a_s * sr
    if op_features and alpha_feat:
        feats_mean = mean_features(ops_on_machine, a, b, op_features)
        for name, w in alpha_feat.items():
            if name in feats_mean:
                f += w * feats_mean[name]
    return max(0.0, f)


def weibull_cum_hazard(delta_t: float, beta: float, eta: float) -> float:
    delta_t = max(0.0, float(delta_t))
    if delta_t <= 0.0:
        return 0.0
    return (delta_t / float(eta)) ** float(beta)


def expected_failures_in_window(a: float, b: float, reset_origin: float,
                                ops_on_machine: List[Tuple[int,int,int,int,int]],
                                params: dict,
                                alpha_feat: Dict[str, float],
                                op_features: Optional[Dict[Tuple[int,int,int], Dict[str,float]]] = None) -> float:
    if b <= a:
        return 0.0
    f = f_cov_factor(a, b, ops_on_machine,
                     util=params["utilization"],
                     a_u=params["alpha_uk"], a_s=params["alpha_sk"],
                     alpha_feat=alpha_feat, op_features=op_features)
    H = weibull_cum_hazard
    beta, eta = params["beta"], params["eta_0"]
    return f * max(0.0, H(b - reset_origin, beta, eta) - H(a - reset_origin, beta, eta))


def choose_pm_slot_by_risk(pm_start: float, pm_dur: float,
                           ops_on_machine: List[Tuple[int,int,int,int,int]],
                           params: dict, alpha_feat: Dict[str, float],
                           op_features: Optional[Dict[Tuple[int,int,int], Dict[str,float]]] = None) -> float:
    """
    與作業重疊時，移到左/右空檔，選「PM 前期望失敗數」較小者。
    """
    pm_end = pm_start + pm_dur
    left_end = 0.0
    right_start = float("inf")
    for (_, _, _, s, e) in ops_on_machine:
        if e <= pm_start:
            left_end = max(left_end, float(e))
        if s >= pm_end:
            right_start = min(right_start, float(s))

    ta = max(0.0, left_end - pm_dur)   # 左案
    td = right_start                   # 右案

    base = {
        "beta": params["beta"], "eta_0": params["eta_0"],
        "utilization": params["utilization"],
        "alpha_uk": params["alpha_uk"], "alpha_sk": params["alpha_sk"]
    }
    alpha_feat = alpha_feat or {}

    risk_a = expected_failures_in_window(0.0, ta, 0.0, ops_on_machine, base, alpha_feat, op_features)
    risk_d = expected_failures_in_window(0.0, td, 0.0, ops_on_machine, base, alpha_feat, op_features)

    left_fit = (ta + pm_dur) <= pm_start
    right_fit = (td < float("inf"))
    if left_fit and right_fit:
        return ta if risk_a <= risk_d else td
    elif left_fit:
        return ta
    elif right_fit:
        return td
    else:
        return pm_start


# ====================================================================
# 3) 多次 PM 的基因編碼（固定長度 + -1 mask）
# ====================================================================
def init_individual_multi_pm(num_machines: int, max_pm_start: int) -> List[int]:
    genes = []
    for _ in range(num_machines * KMAX_PM_PER_MACHINE):
        if random.random() < P_GENE_OFF:
            genes.append(-1)
        else:
            genes.append(random.randint(0, max_pm_start))
    return genes


def mutate_multi_pm(individual: List[int], max_pm_start: int, indpb: float = 0.1):
    for i in range(len(individual)):
        if random.random() < indpb:
            if random.random() < P_GENE_OFF:
                individual[i] = -1
            else:
                individual[i] = random.randint(0, max_pm_start)
    return (individual,)


def decode_pm_times_for_machine(individual: List[int], machine_id: int) -> List[float]:
    start = machine_id * KMAX_PM_PER_MACHINE
    end = start + KMAX_PM_PER_MACHINE
    times = [t for t in individual[start:end] if t is not None and t >= 0]
    times.sort()
    return [float(t) for t in times]


# ====================================================================
# 4) 評估：回傳 (total_cost, makespan)；插入期望 MR 停機；多 PM 分段
# ====================================================================
def evaluate_two_objectives(individual: List[int],
                            production_schedule: List[Tuple[int,int,int,int,int]],
                            num_machines: int,
                            dynamic_params: dict,
                            cost_params: dict,
                            op_features: Optional[Dict[Tuple[int,int,int], Dict[str,float]]] = None) -> tuple:

    # 依機台彙整作業
    machine_ops: Dict[int, List[Tuple[int,int,int,int,int]]] = {k: [] for k in range(num_machines)}
    for j, o, m, s, e in production_schedule:
        machine_ops[m].append((j, o, m, s, e))
    for k in range(num_machines):
        machine_ops[k].sort(key=lambda x: x[3])

    total_cost = 0.0
    machine_end_times = [0.0] * num_machines

    for k in range(num_machines):
        ops = machine_ops[k]
        if not ops:
            continue

        pm_dur = float(cost_params["Tp_k"][k])
        pm_times = decode_pm_times_for_machine(individual, k)

        # 與作業重疊的 PM 先做 advance/postpone 決策
        if pm_times:
            adjusted = []
            for t in pm_times:
                overlap = any(max(t, s) < min(t + pm_dur, e) for (_,_,_,s,e) in ops)
                if overlap:
                    t = choose_pm_slot_by_risk(
                        t, pm_dur, ops,
                        params={
                            "beta": cost_params["beta"][k],
                            "eta_0": cost_params["eta_0"][k],
                            "utilization": dynamic_params["utilization"][k],
                            "alpha_uk": dynamic_params["alpha_uk"][k],
                            "alpha_sk": dynamic_params["alpha_sk"][k],
                        },
                        alpha_feat=dynamic_params.get("alpha_feat", {}),
                        op_features=op_features
                    )
                adjusted.append(t)
            pm_times = sorted(adjusted)

            # 去除 PM 彼此重疊：若相鄰重疊，把後者推到前者之後
            cleaned = []
            last_end = -1e18
            for t in pm_times:
                t_eff = max(t, last_end)
                cleaned.append(t_eff)
                last_end = t_eff + pm_dur
            pm_times = cleaned

        # 模擬時間線（插入 PM 與 MR 停機）
        current_time = 0.0
        segment_reset = 0.0
        pm_idx = 0
        Kpm = len(pm_times)

        for (j, o, m, s, e) in ops:
            dur = float(e - s)

            # 先處理所有「排在此刻之前」的 PM
            while pm_idx < Kpm and pm_times[pm_idx] < current_time + 1e-9:
                # 計算上一段（segment_reset -> current_time）期望故障，插入停機
                params_k = {
                    "beta": float(cost_params["beta"][k]), "eta_0": float(cost_params["eta_0"][k]),
                    "utilization": float(dynamic_params["utilization"][k]),
                    "alpha_uk": float(dynamic_params["alpha_uk"][k]), "alpha_sk": float(dynamic_params["alpha_sk"][k]),
                }
                alpha_feat = dynamic_params.get("alpha_feat", {})
                Ef = expected_failures_in_window(segment_reset, current_time, segment_reset,
                                                 ops_on_machine=ops, params=params_k,
                                                 alpha_feat=alpha_feat, op_features=op_features)
                total_cost += float(cost_params["Cf_k"][k]) * Ef
                current_time += Ef * float(cost_params["Td_k"][k])  # 插入 MR 停機
                # 插入 PM
                current_time += pm_dur
                total_cost += float(cost_params["Cp_k"][k])
                segment_reset = current_time
                pm_idx += 1

            next_pm = pm_times[pm_idx] if pm_idx < Kpm else float("inf")

            # 作業是否會跨越下一個 PM？（不允許 preempt）
            if current_time + dur <= next_pm + 1e-9:
                current_time += dur
            else:
                # 先結算到 PM 之前的風險與 MR 停機
                params_k = {
                    "beta": float(cost_params["beta"][k]), "eta_0": float(cost_params["eta_0"][k]),
                    "utilization": float(dynamic_params["utilization"][k]),
                    "alpha_uk": float(dynamic_params["alpha_uk"][k]), "alpha_sk": float(dynamic_params["alpha_sk"][k]),
                }
                alpha_feat = dynamic_params.get("alpha_feat", {})
                Ef = expected_failures_in_window(segment_reset, next_pm, segment_reset,
                                                 ops_on_machine=ops, params=params_k,
                                                 alpha_feat=alpha_feat, op_features=op_features)
                total_cost += float(cost_params["Cf_k"][k]) * Ef
                current_time = max(current_time, next_pm)
                current_time += pm_dur
                current_time += Ef * float(cost_params["Td_k"][k])  # 插入 MR 停機
                total_cost += float(cost_params["Cp_k"][k])
                segment_reset = current_time

                # 作業整個搬到 PM 後
                current_time += dur
                pm_idx += 1

        # 作業全做完後，若還有 PM，逐一插入
        while pm_idx < Kpm:
            params_k = {
                "beta": float(cost_params["beta"][k]), "eta_0": float(cost_params["eta_0"][k]),
                "utilization": float(dynamic_params["utilization"][k]),
                "alpha_uk": float(dynamic_params["alpha_uk"][k]), "alpha_sk": float(dynamic_params["alpha_sk"][k]),
            }
            alpha_feat = dynamic_params.get("alpha_feat", {})
            Ef = expected_failures_in_window(segment_reset, current_time, segment_reset,
                                             ops_on_machine=ops, params=params_k,
                                             alpha_feat=alpha_feat, op_features=op_features)
            total_cost += float(cost_params["Cf_k"][k]) * Ef
            current_time += Ef * float(cost_params["Td_k"][k])
            current_time += pm_dur
            total_cost += float(cost_params["Cp_k"][k])
            segment_reset = current_time
            pm_idx += 1

        # 最後一段（最後 PM 後到完工）
        params_k = {
            "beta": float(cost_params["beta"][k]), "eta_0": float(cost_params["eta_0"][k]),
            "utilization": float(dynamic_params["utilization"][k]),
            "alpha_uk": float(dynamic_params["alpha_uk"][k]), "alpha_sk": float(dynamic_params["alpha_sk"][k]),
        }
        alpha_feat = dynamic_params.get("alpha_feat", {})
        Ef = expected_failures_in_window(segment_reset, current_time, segment_reset,
                                         ops_on_machine=ops, params=params_k,
                                         alpha_feat=alpha_feat, op_features=op_features)
        total_cost += float(cost_params["Cf_k"][k]) * Ef
        current_time += Ef * float(cost_params["Td_k"][k])

        machine_end_times[k] = current_time

    makespan = max(machine_end_times) if machine_end_times else 0.0
    total_cost += compute_TBC(machine_ops, mode=TBC_MODE, Cb=500.0)  # 不含 makespan 權重
    return (total_cost, makespan)


# ====================================================================
# 5) DEAP：NSGA-II with multi-PM genotype
# ====================================================================
def safe_init_creators():
    if "FitnessMin2" not in creator.__dict__:
        creator.create("FitnessMin2", base.Fitness, weights=(-1.0, -1.0))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMin2)


def create_toolbox(num_machines: int, max_pm_start: int, evaluate_fn):
    safe_init_creators()
    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     lambda: init_individual_multi_pm(num_machines, max_pm_start))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate_multi_pm, max_pm_start=max_pm_start, indpb=0.1)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("evaluate", evaluate_fn)
    return toolbox


# ====================================================================
# 6) 主程式
# ====================================================================
def main():
    # ===== 可調規模 =====
    NUM_JOBS = 10
    NUM_MACHINES = 5
    POPULATION_SIZE = 120
    NGEN = 250
    # ====================

    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    print(f"[Stage-II NSGA-II - multi-PM + MR停機 + TBC={TBC_MODE}] jobs={NUM_JOBS}, machines={NUM_MACHINES}, Kmax={KMAX_PM_PER_MACHINE}")

    # 這裡用示範排程；實務上改為你的 Stage-I 輸出
    production_schedule = generate_random_job_shop_schedule(NUM_JOBS, NUM_MACHINES, seed=RANDOM_SEED)
    dynamic_params, cost_params = generate_random_params(NUM_MACHINES, seed=RANDOM_SEED)

    # 若有每道工序特徵可填（鍵： (job_id, op_id, machine_id) ）
    op_features: Dict[Tuple[int,int,int], Dict[str,float]] = {}
    # 例如： op_features[(0,0,1)] = {"load":0.7, "temp":0.4}
    # dynamic_params["alpha_feat"] = {"load":0.6, "temp":0.4}

    # 搜尋上界：最後完工 + 2*max(Tp_k)
    last_end_per_machine = [0] * NUM_MACHINES
    for (_, _, m, _, e) in production_schedule:
        last_end_per_machine[m] = max(last_end_per_machine[m], e)
    global_upper = max(last_end_per_machine) + 2 * max(cost_params["Tp_k"])
    max_pm_start = int(global_upper)

    def eval_with_context(individual):
        return evaluate_two_objectives(
            individual=individual,
            production_schedule=production_schedule,
            num_machines=NUM_MACHINES,
            dynamic_params=dynamic_params,
            cost_params=cost_params,
            op_features=op_features
        )

    toolbox = create_toolbox(NUM_MACHINES, max_pm_start, eval_with_context)

    population = toolbox.population(n=POPULATION_SIZE)
    # 先做一次 NSGA-II 選擇以初始化 crowding distance
    population = toolbox.select(population, len(population))

    hof = tools.ParetoFront()

    algorithms.eaMuPlusLambda(
        population, toolbox,
        mu=POPULATION_SIZE, lambda_=POPULATION_SIZE,
        cxpb=0.75, mutpb=0.25,
        ngen=NGEN, stats=None, halloffame=hof, verbose=True
    )

    # 印出部分 Pareto 解
    print("\n========== Pareto 前緣（部分） ==========")
    print(f"非支配解數量：{len(hof)}（以下最多列 5 筆）")
    for i, ind in enumerate(hof[:5]):
        c, m = eval_with_context(ind)
        print(f"[{i}] cost={c:.3f}, makespan={m:.3f}, pm={list(ind)}")

    # 代表解：以 total_cost 最小者
    best_by_cost = min(hof, key=lambda ind: eval_with_context(ind)[0])
    best_cost, best_ms = eval_with_context(best_by_cost)
    print("\n========== 代表解（total_cost 最小） ==========")
    print(f"PM 基因：{list(best_by_cost)}")
    print(f"最低總成本: {best_cost:.3f}，對應 makespan: {best_ms:.3f}")

    # 視覺化（若環境具備）
    if HAS_ENV_MODULES and NUM_JOBS <= 12 and NUM_MACHINES <= 12:
        try:
            job_shop = JobShop()
            for i in range(NUM_MACHINES):
                job_shop.add_machine(Machine(machine_id=i))

            num_jobs_vis = max(op[0] for op in production_schedule) + 1
            jobs_vis = [Job(job_id=i) for i in range(num_jobs_vis)]
            for j in jobs_vis:
                job_shop.add_job(j)

            for (job_id, op_id, machine_id, start_time, end_time) in production_schedule:
                op = Operation(jobs_vis[job_id], job_id, op_id)
                op.add_operation_scheduling_information(machine_id, start_time, 0, end_time - start_time)
                job_shop.machines[machine_id]._processed_operations.append(op)

            pm_schedule = []
            for k in range(NUM_MACHINES):
                times = decode_pm_times_for_machine(best_by_cost, k)
                for t in times:
                    pm_schedule.append((k, float(t), float(cost_params["Tp_k"][k])))

            final_makespan_vis = max(
                max(e for (*_, e) in production_schedule),
                max((pm[1] + pm[2]) for pm in pm_schedule) if pm_schedule else 0.0
            )

            plt = gantt_with_pm.plot(job_shop, pm_schedule)
            plt.xlim(0, final_makespan_vis + 5)
            out_png = "gantt_with_pm_ga_pareto_plus.png"
            plt.savefig(out_png, bbox_inches="tight", dpi=150)
            print(f"生成甘特圖: {out_png}")
        except Exception as e:
            print(f"[視覺化略過] 例外：{e}")
    else:
        print("（視覺化略過：規模過大或缺模組）")


if __name__ == "__main__":
    main()
