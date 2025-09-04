# -*- coding: utf-8 -*-
"""
safety_nsga3.py
----------------
NSGA-III for Stage-2 (FJSP + PM) with Safety settings + MR expected downtime insertion + Figure suite:
- 目標： [Makespan, Maintenance Cost, Safety Risk (ΔH), Load Imbalance]
- High-Frequency Switching：λ(t|X)=λ_Weibull(age;η,β) * α * exp(β·X)
- 視窗偵測：失效率門檻 + MRL gate
- Advance/Postpone（式(19)–(25)）+ 右移重排（維持先後）
- Crew limit：對 PM 與 MR 事件「聯合」限制
- MR 期望停機（式(32)精神）：每個週期以 H(T') 近似 E[N]，在週期起點後的下一個空檔插入 E[N]*T_f 的「虛擬停機」
- 視覺化：Baseline/Optimized 甘特（PM=綠、MR=橘）、失效率面板、最佳視窗 vs 實際 PM、Pareto、收斂、成本拆解、h_thr 靈敏度
- I/O：支援 CSV/JSON 載入 Stage-1（CSV 欄：machine_id,start,end,job_id,load_factor[,cond_*]）
"""

import os, csv, json, math, random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================
# 資料結構
# =========================

@dataclass
class Operation:
    start: float
    end: float
    job_id: int = -1
    load_factor: float = 1.0
    cond: List[float] = field(default_factory=list)

@dataclass
class MachineTimeline:
    ops: List[Operation]

@dataclass
class Instance:
    machines: List[MachineTimeline]
    pm_duration: float                # Tp
    cost_pm_fixed: float              # Cp
    cost_pm_time: float               # PM 時間成本（每分鐘；保留顯示用）
    weibull_beta: float               # β
    weibull_eta: float                # η
    h_threshold: float = 1.0e-3
    delay_cost_rate: float = 1.0

    # 視窗掃描 & MRL gate
    window_scan_points: int = 400
    use_mrl_gate: bool = True
    mrl_threshold: float = 20.0

    # 可行性
    enforce_idle_pm: bool = True
    maint_crew_limit: int = 1

    # 高頻切換參數
    alpha_switch: float = 1.0
    cond_coeffs: List[float] = field(default_factory=list)
    cond_names: List[str] = field(default_factory=list)

    # MR 參數（式(32)的 T_f）
    mr_time: float = 8.0             # 每次修復的預期停機時間（分鐘，可改）

# =========================
# 載入器
# =========================

def load_instance_from_csv(csv_path: str,
                           pm_duration=12.0, cost_pm_fixed=60.0, cost_pm_time=4.0,
                           weibull_beta=2.0, weibull_eta=140.0, h_threshold=1e-3,
                           delay_cost_rate=1.0, alpha_switch=1.0,
                           cond_coeffs: Optional[List[float]] = None,
                           cond_names: Optional[List[str]] = None,
                           mr_time: float = 8.0) -> Instance:
    rows = []
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    if not rows:
        raise ValueError("CSV 無資料")

    fieldnames = rows[0].keys()
    cond_cols = [c for c in fieldnames if str(c).startswith("cond_")]
    if cond_names is None:
        cond_names = cond_cols[:]
    if cond_coeffs is None:
        cond_coeffs = [0.0] * len(cond_cols)

    by_m: Dict[int, List[Operation]] = {}
    for row in rows:
        mid = int(row["machine_id"])
        s = float(row["start"]); e = float(row["end"])
        jid = int(row.get("job_id", -1)) if row.get("job_id", "") != "" else -1
        lf = float(row.get("load_factor", 1.0)) if row.get("load_factor", "") != "" else 1.0
        cond = []
        for c in cond_cols:
            try:
                cond.append(float(row[c]))
            except:
                cond.append(0.0)
        by_m.setdefault(mid, []).append(Operation(s, e, jid, lf, cond))

    machines = []
    for mid in sorted(by_m.keys()):
        ops = sorted(by_m[mid], key=lambda o: o.start)
        machines.append(MachineTimeline(ops=ops))

    return Instance(
        machines=machines,
        pm_duration=pm_duration,
        cost_pm_fixed=cost_pm_fixed,
        cost_pm_time=cost_pm_time,
        weibull_beta=weibull_beta,
        weibull_eta=weibull_eta,
        h_threshold=h_threshold,
        delay_cost_rate=delay_cost_rate,
        alpha_switch=alpha_switch,
        cond_coeffs=cond_coeffs,
        cond_names=cond_names,
        mr_time=mr_time
    )

def load_instance_from_json(json_path: str) -> Instance:
    with open(json_path, "r") as f:
        data = json.load(f)
    machines = []
    for m in data["machines"]:
        ops = []
        for op in m["ops"]:
            ops.append(Operation(
                start=op["start"], end=op["end"], job_id=op.get("job_id", -1),
                load_factor=op.get("load_factor", 1.0),
                cond=op.get("cond", [])
            ))
        ops = sorted(ops, key=lambda o: o.start)
        machines.append(MachineTimeline(ops=ops))
    return Instance(
        machines=machines,
        pm_duration=data["pm_duration"],
        cost_pm_fixed=data["cost_pm_fixed"],
        cost_pm_time=data["cost_pm_time"],
        weibull_beta=data["weibull_beta"],
        weibull_eta=data["weibull_eta"],
        h_threshold=data.get("h_threshold", 1e-3),
        delay_cost_rate=data.get("delay_cost_rate", 1.0),
        window_scan_points=data.get("window_scan_points", 400),
        use_mrl_gate=data.get("use_mrl_gate", True),
        mrl_threshold=data.get("mrl_threshold", 20.0),
        enforce_idle_pm=data.get("enforce_idle_pm", True),
        maint_crew_limit=data.get("maint_crew_limit", 1),
        alpha_switch=data.get("alpha_switch", 1.0),
        cond_coeffs=data.get("cond_coeffs", []),
        cond_names=data.get("cond_names", []),
        mr_time=data.get("mr_time", 8.0)
    )

def load_sample_instance() -> Instance:
    rng = random.Random(42)
    machines = []
    for _ in range(3):
        t = 0.0
        ops = []
        for _ in range(rng.randint(8, 10)):
            dur = rng.uniform(8, 18)
            lf = rng.uniform(0.8, 1.4)
            jid = rng.randint(1, 6)
            cond = [rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0)]
            ops.append(Operation(start=t, end=t+dur, job_id=jid, load_factor=lf, cond=cond))
            t += dur + rng.uniform(0.0, 4.0)
        machines.append(MachineTimeline(ops=ops))
    return Instance(
        machines=machines,
        pm_duration=12.0,
        cost_pm_fixed=60.0,
        cost_pm_time=4.0,
        weibull_beta=2.0,
        weibull_eta=140.0,
        h_threshold=1.0e-3,
        delay_cost_rate=1.0,
        alpha_switch=1.0,
        cond_coeffs=[0.35, 0.25],
        cond_names=["cut_speed", "feed"],
        mr_time=8.0
    )

# =========================
# 可靠度/切換模型
# =========================

def weibull_cum_hazard(t: float, eta: float, beta: float) -> float:
    return (t / eta) ** beta if t > 0 else 0.0

def weibull_hazard_rate(t: float, eta: float, beta: float) -> float:
    if t <= 0: return 0.0
    return (beta / eta) * (t / eta) ** (beta - 1)

def weibull_reliability(t: float, eta: float, beta: float) -> float:
    if t < 0: t = 0.0
    return math.exp(- (t/eta)**beta)

def mean_residual_life_weibull(t: float, eta: float, beta: float) -> float:
    if t < 0: t = 0.0
    R_t = weibull_reliability(t, eta, beta)
    if R_t < 1e-15: return 0.0
    u_max = t + 10.0 * eta
    n = 400
    h = (u_max - t) / n
    s = 0.0
    for i in range(n + 1):
        u = t + i*h
        w = 4 if i % 2 == 1 else 2
        if i == 0 or i == n: w = 1
        s += w * weibull_reliability(u, eta, beta)
    integral = s * h / 3.0
    return max(0.0, integral / R_t)

def scale_eta_by_load(eta_base: float, load_factor: float) -> float:
    return eta_base / max(0.5, min(1.5, load_factor))

def cond_factor(alpha: float, coeffs: List[float], cond_vec: List[float]) -> float:
    if not coeffs or not cond_vec:
        return alpha
    n = min(len(coeffs), len(cond_vec))
    return alpha * math.exp(sum(coeffs[i]*cond_vec[i] for i in range(n)))

def hazard_rate_with_switch(age: float, eta_base: float, beta: float,
                            load_factor: float, cond_vec: List[float],
                            alpha: float, coeffs: List[float]) -> float:
    eta_dyn = scale_eta_by_load(eta_base, load_factor)
    base = weibull_hazard_rate(age, eta_dyn, beta)
    return base * cond_factor(alpha, coeffs, cond_vec)

def cum_hazard_increment(a1: float, a2: float, eta_base: float, beta: float,
                         load_factor: float, cond_vec: List[float],
                         alpha: float, coeffs: List[float]) -> float:
    if a2 <= a1: return 0.0
    eta_dyn = scale_eta_by_load(eta_base, load_factor)
    Hdiff = weibull_cum_hazard(a2, eta_dyn, beta) - weibull_cum_hazard(a1, eta_dyn, beta)
    return max(0.0, cond_factor(alpha, coeffs, cond_vec) * Hdiff)

# piecewise H(T)：把 [last_pm, last_pm+T] 映到各工序段積分
def piecewise_H_over_duration(
    m_ops: List[Tuple[float,float,int,float,List[float]]],
    last_pm: float,
    T: float,
    eta_base: float,
    beta: float,
    alpha: float,
    coeffs: List[float]
) -> float:
    if T <= 0: return 0.0
    t0, t1 = last_pm, last_pm + T
    H = 0.0
    for (s,e,jid,lf,cond) in m_ops:
        if e <= t0 or s >= t1: continue
        ss, ee = max(s, t0), min(e, t1)
        a1, a2 = ss - last_pm, ee - last_pm
        H += cum_hazard_increment(a1, a2, eta_base, beta, lf, cond, alpha, coeffs)
    return max(0.0, H)

# =========================
# Advance / Postpone（式(19)–(25)）
# =========================

def decide_advance_postpone(t_star: float,
                            prev_end: float,
                            curr_start: float,
                            curr_end: float,
                            Tp: float,
                            eta: float, beta: float,
                            load_factor: float, cond_vec: List[float],
                            alpha: float, coeffs: List[float],
                            Cp: float, Cf: float,
                            T_star: float) -> Tuple[float, str, Dict[str, float]]:
    if curr_start - prev_end > Tp:
        ta = curr_start - Tp
    else:
        ta = prev_end
    td = curr_end

    def H_abs(a, b):
        # 這裡以「年齡」為自變數近似 (因爲 a,b 是絕對時間；若需要更精確可帶 last_pm)
        # 對 advance/postpone 的相對比較夠用
        return abs(b - a)  # 僅用比例近似；如需嚴謹可改為 piecewise_H_between_abs

    CMa = Cf * H_abs(t_star, ta)
    delta_a = abs(t_star - ta)
    denom_a = max(1e-8, T_star - delta_a)
    CPa = (delta_a / denom_a) * Cp

    CMd = Cf * H_abs(t_star, td)
    delta_d = abs(td - t_star)
    denom_d = max(1e-8, T_star + delta_d)
    CPd = (delta_d / denom_d) * Cp

    if (CMa - CPa) > (CPd - CMd):
        return ta, "advance", dict(CMa=CMa, CPa=CPa, CMd=CMd, CPd=CPd)
    else:
        return td, "postpone", dict(CMa=CMd, CPa=CPd, CMd=CMd, CPd=CPd)

# =========================
# Idle/Overlap/Crew 限制
# =========================

def check_overlap(a: Tuple[float,float], b: Tuple[float,float]) -> bool:
    return not (a[1] <= b[0] or a[0] >= b[1])

def idle_windows_from_ops_list(ops: List[Tuple[float,float,int,float,List[float]]]) -> List[Tuple[float,float]]:
    if not ops: return [(0.0, float("inf"))]
    X = sorted(ops, key=lambda z:z[0])
    wins = []
    if X[0][0] > 0: wins.append((0.0, X[0][0]))
    for a,b in zip(X[:-1], X[1:]):
        if b[0] > a[1]:
            wins.append((a[1], b[0]))
    return wins

def enforce_crew_limit_events(events_by_m: Dict[int, List[Tuple[float,float,str]]], K: int)\
        -> Dict[int, List[Tuple[float,float,str]]]:
    """
    聯合 crew 限制（PM + MR），若同時進行數 > K，將較晚者右移到當前最晚事件結束後。
    events_by_m[m] = [(s,e,'PM'/'MR'), ...]
    """
    all_events = []
    for m, segs in events_by_m.items():
        for (s,e,tag) in segs:
            all_events.append([s,e,m,tag])
    all_events.sort(key=lambda x: x[0])

    active = []  # list of (end, idx_in_all)
    for i in range(len(all_events)):
        s,e,m,tag = all_events[i]
        active = [a for a in active if all_events[a[1]][1] > s]
        if len(active) >= K:
            new_start = max(all_events[a[1]][1] for a in active)
            shift = new_start - s
            all_events[i][0] += shift
            all_events[i][1] += shift
        active.append((all_events[i][1], i))
        active.sort(key=lambda x: x[0])

    out: Dict[int, List[Tuple[float,float,str]]] = {m:[] for m in events_by_m.keys()}
    for s,e,m,tag in sorted(all_events, key=lambda x:(x[2], x[0])):
        out[m].append((s,e,tag))
    return out

# =========================
# 問題定義 & 模擬（含 MR 期望停機插入）
# =========================

class Stage2PMProblem:
    def __init__(self, inst: Instance):
        self.inst = inst
        self.machine_slot_sizes = [len(m.ops) for m in inst.machines]
        self.D = sum(self.machine_slot_sizes)

    def split_genes(self, x: np.ndarray) -> List[np.ndarray]:
        res, idx = [], 0
        for sz in self.machine_slot_sizes:
            res.append(x[idx:idx+sz]); idx += sz
        return res

    def _find_T_star(self, alpha_f: float, eta_dyn: float, beta_shape: float) -> float:
        Cp = self.inst.cost_pm_fixed
        Cf, Tf = 2.0*Cp, 0.0
        Tp = self.inst.pm_duration
        Ts = np.linspace(8.0, 240.0, 120)
        VC_list, VA_list = [], []
        for T in Ts:
            H = alpha_f * ((T / eta_dyn) ** beta_shape)
            denom = (T + Tp + Tf * H + 1e-9)
            VC = (Cp + Cf * H) / denom
            VA = T / denom
            VC_list.append(VC); VA_list.append(VA)
        VC_arr = np.array(VC_list); VA_arr = np.array(VA_list)
        VC_star = VC_arr.min(); VA_star = VA_arr.max()
        V = 0.5*(VC_arr/(VC_star+1e-9)) - 0.5*(VA_arr/(VA_star+1e-9))
        return float(Ts[int(np.argmin(V))])

    def simulate_with_pm(self, x: np.ndarray):
        inst = self.inst
        per_m_genes = self.split_genes(x)

        # 可修改的甘特（當前）
        local_ops: Dict[int, List[Tuple[float,float,int,float,List[float]]]] = {}
        for m_idx, tl in enumerate(inst.machines):
            local_ops[m_idx] = [(op.start, op.end, op.job_id, op.load_factor, op.cond) for op in tl.ops]

        # 1) 由基因觸發 → 求 T* 與 t'，建立 PM 事件；同時保存各 cycle 的 T' 與起點
        pm_segments: Dict[int, List[Tuple[float,float]]] = {m:[] for m in range(len(inst.machines))}
        cycle_Tprime: Dict[int, List[float]] = {m:[] for m in range(len(inst.machines))}
        cycle_start : Dict[int, List[float]] = {m:[] for m in range(len(inst.machines))}
        last_pm_end: Dict[int, float] = {m:0.0 for m in range(len(inst.machines))}

        for m_idx, genes in enumerate(per_m_genes):
            for i, g in enumerate(genes):
                s, e, jid, lf, cond = local_ops[m_idx][i]
                alpha_f = cond_factor(inst.alpha_switch, inst.cond_coeffs, cond)
                eta_dyn = scale_eta_by_load(inst.weibull_eta, lf)

                T_star = self._find_T_star(alpha_f, eta_dyn, inst.weibull_beta)
                t_star = last_pm_end[m_idx] + T_star

                if int(g) == 1:
                    prev_end = local_ops[m_idx][i-1][1] if i>0 else 0.0
                    if s <= t_star < e:
                        t_prime, _, _ = decide_advance_postpone(
                            t_star, prev_end, s, e, inst.pm_duration,
                            inst.weibull_eta, inst.weibull_beta, lf, cond,
                            inst.alpha_switch, inst.cond_coeffs,
                            inst.cost_pm_fixed, 2.0*inst.cost_pm_fixed,
                            T_star
                        )
                    else:
                        if t_star < s:
                            t_prime = max(prev_end, min(s - inst.pm_duration, t_star))
                            t_prime = max(t_prime, 0.0)
                        else:
                            t_prime = e

                    pm_segments[m_idx].append((t_prime, t_prime + inst.pm_duration))
                    # 記錄此 cycle 的起點與 T'
                    T_prime = max(1e-6, T_star + (t_prime - t_star))   # 式(28)：T' = T* + (t' - t*)
                    cycle_start[m_idx].append(last_pm_end[m_idx])      # 上一個 PM 結束即 cycle 起點
                    cycle_Tprime[m_idx].append(T_prime)

                    last_pm_end[m_idx] = t_prime + inst.pm_duration   # 下一 cycle 的起點

        # 2) 依據每個 cycle 的 H(T') 期望式插入 MR 虛擬停機（E[N]≈H；時長=E[N]*T_f）
        mr_segments: Dict[int, List[Tuple[float,float]]] = {m:[] for m in range(len(inst.machines))}
        for m_idx in range(len(inst.machines)):
            ops = local_ops[m_idx]
            for cs, Tprime in zip(cycle_start[m_idx], cycle_Tprime[m_idx]):
                H_cycle = piecewise_H_over_duration(
                    ops, cs, Tprime,
                    inst.weibull_eta, inst.weibull_beta,
                    inst.alpha_switch, inst.cond_coeffs
                )
                mr_dur = H_cycle * max(inst.mr_time, 0.0)
                if mr_dur <= 1e-6:
                    continue
                # 嘗試放在「cs 之後」的第一個 idle 視窗；若沒有合適視窗，就從 cs 放下去，後續右移解決
                wins = idle_windows_from_ops_list(ops)
                placed = False
                for (ws,we) in wins:
                    if we <= cs: 
                        continue
                    start = max(ws, cs)
                    if (we - start) >= mr_dur:
                        mr_segments[m_idx].append((start, start + mr_dur))
                        placed = True
                        break
                if not placed:
                    mr_segments[m_idx].append((cs, cs + mr_dur))

        # 3) Crew limit（PM+MR 聯合）
        if inst.maint_crew_limit > 0:
            events_by_m = {m: [(s,e,'PM') for (s,e) in pm_segments[m]] + [(s,e,'MR') for (s,e) in mr_segments[m]]
                           for m in range(len(inst.machines))}
            limited = enforce_crew_limit_events(events_by_m, max(1, inst.maint_crew_limit))
            # 拆回 PM/MR
            pm_segments = {m:[(s,e) for (s,e,tag) in limited[m] if tag=='PM'] for m in limited}
            mr_segments = {m:[(s,e) for (s,e,tag) in limited[m] if tag=='MR'] for m in limited}

        # 4) 右移重排（遇到 PM/MR 區段都視為維護停機）
        for m_idx in range(len(inst.machines)):
            ops = local_ops[m_idx]
            blocks = sorted(pm_segments[m_idx] + mr_segments[m_idx], key=lambda z:z[0])
            for (bs,be) in blocks:
                new_ops = []
                shift = 0.0
                for (s,e,jid,lf,cond) in ops:
                    s += shift; e += shift
                    if check_overlap((s,e), (bs,be)):
                        delta = be - s
                        s += delta; e += delta
                        shift += delta
                    new_ops.append((s,e,jid,lf,cond))
                ops = new_ops
            local_ops[m_idx] = ops

        # 5) 目標
        makespan = 0.0
        for ops in local_ops.values():
            if ops:
                makespan = max(makespan, max(e for (s,e,_,_,_) in ops))

        total_pm = sum(len(v) for v in pm_segments.values())
        total_pm_time = sum((pe-ps) for segs in pm_segments.values() for (ps,pe) in segs)
        maintenance_cost = inst.cost_pm_fixed * total_pm + inst.cost_pm_time * total_pm_time

        safety_risk = 0.0
        for m_idx, ops in local_ops.items():
            last_pm = 0.0
            # 將 PM 視為 reset；MR 只是停機不重置 age
            pms = sorted(pm_segments[m_idx], key=lambda z:z[0])
            pm_i = 0
            for (s,e,jid,lf,cond) in ops:
                while pm_i < len(pms) and pms[pm_i][1] <= s:
                    last_pm = pms[pm_i][1]; pm_i += 1
                t1 = max(last_pm, s); t2 = e
                safety_risk += piecewise_H_over_duration(
                    ops=[(s,e,jid,lf,cond)],  # 單段
                    last_pm=last_pm, T=t2-t1,
                    eta_base=inst.weibull_eta, beta=inst.weibull_beta,
                    alpha=inst.alpha_switch, coeffs=inst.cond_coeffs
                )
                while pm_i < len(pms) and check_overlap((s,e), pms[pm_i]):
                    last_pm = pms[pm_i][1]; pm_i += 1

        finishes = []
        for ops in local_ops.values():
            finishes.append(max([e for (s,e,_,_,_) in ops]) if ops else 0.0)
        finishes = np.array(finishes, dtype=float)
        mu = finishes.mean() if len(finishes)>0 else 0.0
        load_imbalance = float(((finishes - mu)**2).sum())

        feasible = True
        vio = 0
        if self.inst.enforce_idle_pm:
            # 只檢 PM 是否落在 baseline idle（MR 是隨機停機，通常不強制）
            for m_idx, tl in enumerate(self.inst.machines):
                idles = idle_windows_from_ops_list([(op.start,op.end,op.job_id,op.load_factor,op.cond) for op in tl.ops])
                for (ps,pe) in pm_segments[m_idx]:
                    inside = any((ps >= s and pe <= e) for (s,e) in idles)
                    if not inside:
                        feasible = False; vio += 1

        details = dict(
            op_segments={m:[(s,e,j) for (s,e,j,_,_) in local_ops[m]] for m in local_ops},
            pm_segments=pm_segments,
            mr_segments=mr_segments
        )
        return makespan, maintenance_cost, safety_risk, load_imbalance, details, feasible, vio

# =========================
# NSGA-III（約束）
# =========================

def constraint_dominates(a_feas, a_vio, b_feas, b_vio, aF, bF):
    if a_feas and not b_feas: return True
    if b_feas and not a_feas: return False
    if not a_feas and not b_feas:
        if a_vio < b_vio: return True
        if b_vio < a_vio: return False
    return (np.all(aF <= bF) and np.any(aF < bF))

def fast_non_dominated_sort_with_constraints(F: np.ndarray, feas: np.ndarray, vio: np.ndarray) -> List[List[int]]:
    N = F.shape[0]
    S = [[] for _ in range(N)]
    n = np.zeros(N, dtype=int)
    fronts = [[]]
    for p in range(N):
        for q in range(N):
            if p == q: continue
            if constraint_dominates(feas[p], vio[p], feas[q], vio[q], F[p], F[q]):
                S[p].append(q)
            elif constraint_dominates(feas[q], vio[q], feas[p], vio[p], F[q], F[p]):
                n[p] += 1
        if n[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        Q = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    Q.append(q)
        i += 1
        fronts.append(Q)
    fronts.pop()
    return fronts

def das_dennis_reference_points(M: int, p: int) -> np.ndarray:
    out = []
    def rec(prefix, left, d):
        if d == M-1:
            out.append(prefix + [left]); return
        for i in range(left+1):
            rec(prefix + [i], left - i, d+1)
    rec([], p, 0)
    arr = np.array(out, dtype=float) / float(p)
    arr = np.where(arr==0.0, 1e-12, arr)
    return arr

def normalize_objectives(F: np.ndarray):
    zmin = F.min(axis=0)
    zmax = F.max(axis=0)
    denom = np.where(zmax - zmin < 1e-12, 1.0, zmax - zmin)
    return (F - zmin) / denom, zmin, zmax

def associate_to_refs(Fn: np.ndarray, refs: np.ndarray):
    X = Fn + 1e-12
    U = X / np.linalg.norm(X, axis=1, keepdims=True)
    refs_u = refs / np.linalg.norm(refs, axis=1, keepdims=True)
    cos = np.clip(U @ refs_u.T, -1.0, 1.0)
    d_perp = np.sqrt(1.0 - cos**2)
    nearest = np.argmin(d_perp, axis=1)
    dmin = d_perp[np.arange(Fn.shape[0]), nearest]
    return nearest, dmin

def tournament_select(popF: np.ndarray, k: int) -> List[int]:
    N = popF.shape[0]
    idxs = []
    for _ in range(k):
        a, b = random.randrange(N), random.randrange(N)
        fa = popF[a].sum() + random.random()*1e-6
        fb = popF[b].sum() + random.random()*1e-6
        idxs.append(a if fa < fb else b)
    return idxs

def sbx_crossover(p1: np.ndarray, p2: np.ndarray, eta: float, prob: float):
    n = len(p1)
    c1, c2 = p1.copy().astype(float), p2.copy().astype(float)
    if random.random() < prob:
        for i in range(n):
            u = random.random()
            beta = (2*u)**(1/(eta+1)) if u <= 0.5 else (1/(2*(1-u)))**(1/(eta+1))
            x1, x2 = c1[i], c2[i]
            c1[i] = 0.5*((1+beta)*x1 + (1-beta)*x2)
            c2[i] = 0.5*((1-beta)*x1 + (1+beta)*x2)
    return (c1 >= 0.5).astype(int), (c2 >= 0.5).astype(int)

def polynomial_mutation(x: np.ndarray, eta: float, prob: float):
    y = x.astype(float).copy()
    for i in range(len(y)):
        if random.random() < prob:
            u = random.random()
            delta = (2*u)**(1/(eta+1)) - 1 if u < 0.5 else 1 - (2*(1-u))**(1/(eta+1))
            y[i] = np.clip(y[i] + 0.5*delta, 0.0, 1.0)
    return (y >= 0.5).astype(int)

def environmental_selection_with_constraints(popX, popF, popFeas, popVio, K, refs):
    fronts = fast_non_dominated_sort_with_constraints(popF, popFeas, popVio)
    newX, newF, newFeas, newVio = [], [], [], []
    count = 0
    for front in fronts:
        if count + len(front) <= K:
            newX.extend(popX[front]); newF.extend(popF[front])
            newFeas.extend(popFeas[front]); newVio.extend(popVio[front])
            count += len(front)
        else:
            remaining = K - count
            F_front, X_front = popF[front], popX[front]
            feas_front, vio_front = popFeas[front], popVio[front]

            Fn, _, _ = normalize_objectives(F_front)
            nearest, dmin = associate_to_refs(Fn, refs)

            Krefs = refs.shape[0]
            niche_count = [0]*Krefs
            pools = [[] for _ in range(Krefs)]
            for idx_local, r in enumerate(nearest):
                key = (0 if feas_front[idx_local] else 1, vio_front[idx_local], dmin[idx_local])
                pools[r].append((idx_local, key))

            selected_local = set()
            while len(selected_local) < remaining:
                minc = min(niche_count)
                candidate_refs = [ri for ri,c in enumerate(niche_count) if c==minc]
                random.shuffle(candidate_refs)
                picked = False
                for ri in candidate_refs:
                    pool_sorted = sorted(pools[ri], key=lambda t: t[1])
                    for li,_ in pool_sorted:
                        if li not in selected_local:
                            selected_local.add(li)
                            niche_count[ri] += 1
                            picked = True
                            break
                    if picked: break
                if not picked:
                    all_ids = list(range(len(F_front))); random.shuffle(all_ids)
                    for a in all_ids:
                        if a not in selected_local:
                            selected_local.add(a)
                            if len(selected_local) >= remaining: break
                    break

            for li in sorted(list(selected_local)):
                newX.append(X_front[li]); newF.append(F_front[li])
                newFeas.append(feas_front[li]); newVio.append(vio_front[li])
            break

    return np.array(newX), np.array(newF), np.array(newFeas), np.array(newVio)

def nsga3_run(problem: Stage2PMProblem,
              pop_size=72, generations=40, ref_divisions=12,
              sbx_eta=15.0, sbx_prob=0.9,
              pm_eta=20.0, pm_prob=0.06,
              seed=7):
    random.seed(seed); np.random.seed(seed)
    N, D = pop_size, problem.D

    popX = np.random.randint(0, 2, size=(N, D), dtype=int)
    pop_eval = [problem.simulate_with_pm(ind) for ind in popX]
    popF = np.array([e[:4] for e in pop_eval], dtype=float)
    popFeas = np.array([1 if e[5] else 0 for e in pop_eval], dtype=int)
    popVio  = np.array([e[6] for e in pop_eval], dtype=float)

    refs = das_dennis_reference_points(popF.shape[1], ref_divisions)

    ideal_hist = []
    for _ in range(generations):
        parent_idx = tournament_select(popF, N)
        parents = popX[parent_idx]
        offsprings = []
        for i in range(0, N, 2):
            p1 = parents[i]; p2 = parents[(i+1)%N]
            c1, c2 = sbx_crossover(p1, p2, eta=sbx_eta, prob=sbx_prob)
            c1 = polynomial_mutation(c1, eta=pm_eta, prob=pm_prob)
            c2 = polynomial_mutation(c2, eta=pm_eta, prob=pm_prob)
            offsprings.append(c1); offsprings.append(c2)
        offsprings = np.array(offsprings[:N], dtype=int)

        off_eval = [problem.simulate_with_pm(ind) for ind in offsprings]
        offF = np.array([e[:4] for e in off_eval], dtype=float)
        offFeas = np.array([1 if e[5] else 0 for e in off_eval], dtype=int)
        offVio  = np.array([e[6] for e in off_eval], dtype=float)

        unionX  = np.vstack([popX, offsprings])
        unionF  = np.vstack([popF, offF])
        unionFe = np.hstack([popFeas, offFeas])
        unionVi = np.hstack([popVio , offVio ])

        popX, popF, popFeas, popVio = environmental_selection_with_constraints(
            unionX, unionF, unionFe, unionVi, K=N, refs=refs
        )
        ideal_hist.append(popF.min(axis=0).tolist())

    fronts = fast_non_dominated_sort_with_constraints(popF, popFeas, popVio)
    f0 = fronts[0]
    paretoX = popX[f0]
    paretoF = popF[f0]

    ideal = paretoF.min(axis=0)
    dist = np.linalg.norm((paretoF - ideal), axis=1)
    rep_local = np.argmin(dist)
    rep_idx = f0[rep_local]
    rep_mk, rep_mc, rep_risk, rep_lb, rep_details, _, _ = problem.simulate_with_pm(popX[rep_idx])

    return dict(
        popX=popX, popF=popF,
        paretoX=paretoX, paretoF=paretoF,
        representative=dict(idx=int(rep_idx), F=(rep_mk, rep_mc, rep_risk, rep_lb), details=rep_details),
        ideal_hist=np.array(ideal_hist)
    )

# =========================
# 視窗偵測（hazard + MRL）
# =========================

def merge_overlapping_windows(wins: List[Tuple[float,float]]) -> List[Tuple[float,float]]:
    if not wins: return []
    wins = sorted(wins, key=lambda x: x[0])
    merged = [wins[0]]
    for s, e in wins[1:]:
        ps, pe = merged[-1]
        if s <= pe: merged[-1] = (ps, max(pe, e))
        else: merged.append((s, e))
    return merged

def compute_optimal_pm_windows_by_hazard_rate(inst: Instance) -> Dict[int, List[Tuple[float,float]]]:
    windows: Dict[int, List[Tuple[float,float]]] = {}
    beta = inst.weibull_beta
    for m_idx, tl in enumerate(inst.machines):
        if not tl.ops:
            windows[m_idx] = []; continue
        t_start = min(op.start for op in tl.ops)
        t_end   = max(op.end   for op in tl.ops)
        if t_end <= t_start:
            windows[m_idx] = []; continue

        P = max(50, inst.window_scan_points)
        Ts = np.linspace(t_start, t_end, P)
        flags = []
        for t in Ts:
            lf, cond = 1.0, []
            for op in tl.ops:
                if op.start <= t <= op.end:
                    lf, cond = op.load_factor, op.cond
                    break
            # 以 age≈t 近似（視窗偵測僅提供粗定位）
            h_t = hazard_rate_with_switch(t, inst.weibull_eta, beta, lf, cond,
                                          inst.alpha_switch, inst.cond_coeffs)
            ok = (h_t >= inst.h_threshold)
            if inst.use_mrl_gate:
                eta_dyn = scale_eta_by_load(inst.weibull_eta, lf)
                c = cond_factor(inst.alpha_switch, inst.cond_coeffs, cond)
                # 等效 η' 近似：eta' = eta_dyn / c^(1/β)
                eta_eff = eta_dyn / (max(c,1e-12)**(1.0/beta))
                mrl_t = mean_residual_life_weibull(t, eta_eff, beta)
                ok = ok and (mrl_t <= inst.mrl_threshold)
            flags.append(ok)

        wins = []
        in_seg = False; s0 = None
        for i, ok in enumerate(flags):
            if ok and not in_seg:
                in_seg = True; s0 = Ts[i]
            if (not ok or i == len(flags)-1) and in_seg:
                e0 = Ts[i] if not ok else Ts[i]
                if e0 > s0: wins.append((s0, e0))
                in_seg = False
        windows[m_idx] = merge_overlapping_windows(wins)
    return windows

# =========================
# 視覺化（PM=綠、MR=橘）
# =========================

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def baseline_makespan(inst: Instance) -> float:
    tmax = 0.0
    for m in inst.machines:
        if m.ops: tmax = max(tmax, max(op.end for op in m.ops))
    return tmax

def plot_gantt_baseline(inst: Instance, out_path: str, title: str = "Baseline (Stage-1 without PM)"):
    _ensure_dir(os.path.dirname(out_path))
    M = len(inst.machines)
    fig = plt.figure(figsize=(8, 4 + 0.2*M))
    ax = plt.gca(); cmap = plt.get_cmap("tab10")
    y_ticks = []; y_labels = []
    for m in range(M-1, -1, -1):
        y = M-1-m; yc = y*10 + 5
        y_ticks.append(yc); y_labels.append(f"M{m+1}")
        for op in inst.machines[m].ops:
            ax.barh(yc, op.end-op.start, left=op.start, height=6, align='center',
                    edgecolor='black', linewidth=0.6, color=cmap((op.job_id-1) % 10))
            ax.text(op.start+(op.end-op.start)/2, yc, f"{op.job_id}", ha='center', va='center', fontsize=8)
    ax.set_yticks(y_ticks); ax.set_yticklabels(y_labels)
    ax.set_xlabel("time/min"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_path, dpi=170); plt.close(fig)

def plot_failure_rate_panels(inst: Instance, out_path: str, with_ops_blocks: bool = True,
                             pm_segments: Optional[Dict[int, List[Tuple[float,float]]]] = None,
                             mr_segments: Optional[Dict[int, List[Tuple[float,float]]]] = None):
    _ensure_dir(os.path.dirname(out_path))
    M = len(inst.machines)
    cols = 3; rows = int(math.ceil(M / cols))
    fig = plt.figure(figsize=(cols*4.6, rows*3.2))
    cmap = plt.get_cmap("tab10")

    tmax = 0.0
    for m in range(M):
        if inst.machines[m].ops:
            tmax = max(tmax, max(op.end for op in inst.machines[m].ops))
    T = np.linspace(0, tmax, 400)

    for m in range(M):
        ax = fig.add_subplot(rows, cols, m+1)
        h_var = []
        for t in T:
            lf, cond = 1.0, []
            for op in inst.machines[m].ops:
                if op.start <= t <= op.end:
                    lf, cond = op.load_factor, op.cond
                    break
            h_var.append(hazard_rate_with_switch(
                t, inst.weibull_eta, inst.weibull_beta, lf, cond, inst.alpha_switch, inst.cond_coeffs
            ))
        ax.plot(T, h_var, linewidth=1.8, label="h(t) dynamic (switching)")

        h_base = [weibull_hazard_rate(t, inst.weibull_eta, inst.weibull_beta) for t in T]
        ax.plot(T, h_base, linestyle='--', linewidth=1.4, label="h(t) baseline")

        if pm_segments and m in pm_segments:
            ymin, ymax = ax.get_ylim()
            for (ps,pe) in pm_segments[m]:
                ax.add_patch(plt.Rectangle((ps, ymin), pe-ps, ymax-ymin,
                                           facecolor='green', alpha=0.18, edgecolor=None))
        if mr_segments and m in mr_segments:
            ymin, ymax = ax.get_ylim()
            for (ps,pe) in mr_segments[m]:
                ax.add_patch(plt.Rectangle((ps, ymin), pe-ps, ymax-ymin,
                                           facecolor='orange', alpha=0.18, edgecolor=None))

        ax.set_title(f"(M{m+1})")
        ax.set_xlabel("time/min"); ax.set_ylabel("Failure rate")
        if with_ops_blocks:
            ymin, ymax = ax.get_ylim()
            y0 = ymin + 0.05*(ymax - ymin); h = 0.08*(ymax - ymin)
            for op in inst.machines[m].ops:
                color = cmap((op.job_id-1) % 10)
                ax.add_patch(plt.Rectangle((op.start, y0), op.end-op.start, h,
                                           edgecolor='black', linewidth=0.4, facecolor=color, alpha=0.9))
        if m == 0: ax.legend(fontsize=8, loc='upper left')

    fig.tight_layout(); fig.savefig(out_path, dpi=160); plt.close(fig)

def plot_optimal_vs_actual_pm(inst: Instance,
                              actual_pm: Dict[int, List[Tuple[float,float]]],
                              out_path: str):
    _ensure_dir(os.path.dirname(out_path))
    M = len(inst.machines)
    fig = plt.figure(figsize=(9, 4 + 0.22*M))
    ax = plt.gca(); cmap = plt.get_cmap("tab10")
    y_ticks = []; y_labels = []
    opt_windows = compute_optimal_pm_windows_by_hazard_rate(inst)

    for m in range(M-1, -1, -1):
        y = M-1-m; yc = y*10 + 5
        y_ticks.append(yc); y_labels.append(f"M{m+1}")
        for op in inst.machines[m].ops:
            ax.barh(yc, op.end-op.start, left=op.start, height=6, align='center',
                    edgecolor='black', linewidth=0.6, color=cmap((op.job_id-1) % 10))
            ax.text(op.start+(op.end-op.start)/2, yc, f"{op.job_id}", ha='center', va='center', fontsize=8)
        for (s,e) in opt_windows[m]:
            ax.add_patch(plt.Rectangle((s, yc-3), e-s, 6, fill=False, linestyle='--',
                                       linewidth=1.2, edgecolor='blue'))
        for (ps,pe) in actual_pm.get(m, []):
            ax.add_patch(plt.Rectangle((ps, yc-3), pe-ps, 6, fill=False, linestyle='--',
                                       linewidth=1.4, edgecolor='red'))

    ax.set_yticks(y_ticks); ax.set_yticklabels(y_labels)
    ax.set_xlabel("time/min")
    ax.set_title("Optimal PM windows (blue, hazard+MRL) vs Actual PM (red)")
    fig.tight_layout(); fig.savefig(out_path, dpi=170); plt.close(fig)

def plot_gantt(op_segments: Dict[int, List[Tuple[float,float,int]]],
               pm_segments: Dict[int, List[Tuple[float,float]]],
               out_path: str,
               title: str = "Gantt (with PM/MR)",
               mr_segments: Optional[Dict[int, List[Tuple[float,float]]]] = None):
    _ensure_dir(os.path.dirname(out_path))
    M = len(op_segments)
    fig = plt.figure(figsize=(8, 4 + 0.2*M))
    ax = plt.gca(); cmap = plt.get_cmap("tab10")
    y_ticks = []; y_labels = []
    for m in range(M-1, -1, -1):
        y = M-1-m; yc = y*10 + 5
        y_ticks.append(yc); y_labels.append(f"M{m+1}")
        for (s,e,jid) in op_segments[m]:
            ax.barh(yc, e-s, left=s, height=6, align='center',
                    edgecolor='black', linewidth=0.6, color=cmap((jid-1) % 10))
            ax.text(s+(e-s)/2, yc, f"{jid}", ha='center', va='center', fontsize=8)
        for (ps,pe) in pm_segments[m]:
            ax.barh(yc, pe-ps, left=ps, height=6, align='center',
                    edgecolor='black', linewidth=0.6, color='green')
        if mr_segments and m in mr_segments:
            for (ps,pe) in mr_segments[m]:
                ax.barh(yc, pe-ps, left=ps, height=6, align='center',
                        edgecolor='black', linewidth=0.6, color='orange')
    ax.set_yticks(y_ticks); ax.set_yticklabels(y_labels)
    ax.set_xlabel("time/min"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(out_path, dpi=170); plt.close(fig)

def plot_gantt_side_by_side(inst: Instance,
                            opt_ops: Dict[int, List[Tuple[float,float,int]]],
                            opt_pms: Dict[int, List[Tuple[float,float]]],
                            out_path: str,
                            opt_mrs: Optional[Dict[int, List[Tuple[float,float]]]] = None,
                            title_left="Baseline (no PM)",
                            title_right="Optimized (with PM+MR)"):
    _ensure_dir(os.path.dirname(out_path))
    M = len(inst.machines)
    fig = plt.figure(figsize=(14, 4 + 0.26*M))
    cmap = plt.get_cmap("tab10")

    ax1 = fig.add_subplot(1,2,1)
    y_ticks = []; y_labels = []
    for m in range(M-1, -1, -1):
        y = M-1-m; yc = y*10 + 5
        y_ticks.append(yc); y_labels.append(f"M{m+1}")
        for op in inst.machines[m].ops:
            ax1.barh(yc, op.end-op.start, left=op.start, height=6, align='center',
                     edgecolor='black', linewidth=0.6, color=cmap((op.job_id-1) % 10))
            ax1.text(op.start+(op.end-op.start)/2, yc, f"{op.job_id}", ha='center', va='center', fontsize=8)
    ax1.set_yticks(y_ticks); ax1.set_yticklabels(y_labels)
    ax1.set_xlabel("time/min"); ax1.set_title(title_left)

    ax2 = fig.add_subplot(1,2,2)
    y_ticks2 = []; y_labels2 = []
    for m in range(M-1, -1, -1):
        y = M-1-m; yc = y*10 + 5
        y_ticks2.append(yc); y_labels2.append(f"M{m+1}")
        for (s,e,jid) in opt_ops[m]:
            ax2.barh(yc, e-s, left=s, height=6, align='center',
                     edgecolor='black', linewidth=0.6, color=cmap((jid-1) % 10))
            ax2.text(s+(e-s)/2, yc, f"{jid}", ha='center', va='center', fontsize=8)
        for (ps,pe) in opt_pms[m]:
            ax2.barh(yc, pe-ps, left=ps, height=6, align='center',
                     edgecolor='black', linewidth=0.6, color='green')
        if opt_mrs and m in opt_mrs:
            for (ps,pe) in opt_mrs[m]:
                ax2.barh(yc, pe-ps, left=ps, height=6, align='center',
                         edgecolor='black', linewidth=0.6, color='orange')
    ax2.set_yticks(y_ticks2); ax2.set_yticklabels(y_labels2)
    ax2.set_xlabel("time/min"); ax2.set_title(title_right)

    fig.tight_layout(); fig.savefig(out_path, dpi=170); plt.close(fig)

def plot_pareto_2d_pairs(popF, paretoF, out_dir):
    _ensure_dir(out_dir)
    names = ["Makespan","Maintenance cost","Safety risk (ΔH)","Load imbalance"]
    M = popF.shape[1]
    for i in range(M):
        for j in range(i+1, M):
            fig = plt.figure(figsize=(5,4))
            plt.scatter(popF[:,i], popF[:,j], s=10, alpha=0.35, label="Population")
            plt.scatter(paretoF[:,i], paretoF[:,j], s=25, label="Pareto")
            plt.xlabel(names[i]); plt.ylabel(names[j]); plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"pareto_2d_{i}{j}.png"), dpi=160)
            plt.close(fig)

def plot_convergence(ideal_hist, out_dir):
    _ensure_dir(out_dir)
    fig = plt.figure(figsize=(6,4))
    for m in range(ideal_hist.shape[1]):
        plt.plot(ideal_hist[:,m], label=f"Obj{m+1}")
    plt.xlabel("Generation"); plt.ylabel("Ideal value (min)"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "convergence.png"), dpi=160); plt.close(fig)

def plot_cost_breakdown(inst: Instance, baseline_mk: float, rep_mk: float, rep_mc: float, out_dir: str):
    _ensure_dir(out_dir)
    delay = max(0.0, rep_mk - baseline_mk)
    delay_cost = delay * inst.delay_cost_rate
    total_cost = rep_mc + delay_cost

    fig = plt.figure(figsize=(6,4))
    xs = np.arange(3); vals = [rep_mc, delay_cost, total_cost]
    labels = ["Maintenance Cost", "Delay Cost", "Total"]
    plt.bar(xs, vals)
    for i,v in enumerate(vals):
        plt.text(xs[i], v + 0.03*(max(vals)+1e-9), f"{v:.1f}", ha='center', va='bottom', fontsize=9)
    plt.xticks(xs, labels, rotation=10)
    plt.ylabel("cost"); plt.title(f"Cost breakdown (delay_rate={inst.delay_cost_rate}/min)")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "cost_breakdown_bar.png"), dpi=170); plt.close(fig)

    fig = plt.figure(figsize=(5,5))
    parts = [rep_mc, delay_cost]
    lbls = [f"MC ({rep_mc:.1f})", f"Delay ({delay_cost:.1f})"]
    plt.pie(parts, labels=lbls, autopct=lambda p: f"{p:.1f}%", startangle=90)
    plt.title("Total cost composition")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "cost_breakdown_pie.png"), dpi=170); plt.close(fig)

# =========================
# h_threshold 靈敏度
# =========================

def sweep_h_threshold(inst: Instance, thresholds: List[float],
                      pop_size=48, generations=25, seed=11) -> List[Dict]:
    results = []
    base_inst = inst
    for hthr in thresholds:
        inst_local = Instance(
            machines=base_inst.machines,
            pm_duration=base_inst.pm_duration,
            cost_pm_fixed=base_inst.cost_pm_fixed,
            cost_pm_time=base_inst.cost_pm_time,
            weibull_beta=base_inst.weibull_beta,
            weibull_eta=base_inst.weibull_eta,
            h_threshold=hthr,
            delay_cost_rate=base_inst.delay_cost_rate,
            window_scan_points=base_inst.window_scan_points,
            use_mrl_gate=base_inst.use_mrl_gate,
            mrl_threshold=base_inst.mrl_threshold,
            enforce_idle_pm=base_inst.enforce_idle_pm,
            maint_crew_limit=base_inst.maint_crew_limit,
            alpha_switch=base_inst.alpha_switch,
            cond_coeffs=base_inst.cond_coeffs,
            cond_names=base_inst.cond_names,
            mr_time=base_inst.mr_time
        )
        problem = Stage2PMProblem(inst_local)
        result = nsga3_run(problem, pop_size=pop_size, generations=generations, seed=seed)
        rep_F = result["representative"]["F"]
        rep_details = result["representative"]["details"]
        pm_count = sum(len(v) for v in rep_details["pm_segments"].values())
        results.append(dict(
            h_threshold=hthr,
            pm_count=pm_count,
            makespan=rep_F[0],
            maintenance_cost=rep_F[1],
            safety_risk=rep_F[2],
            load_imbalance=rep_F[3]
        ))
    return results

def plot_hthr_sensitivity(results: List[Dict], out_dir: str):
    _ensure_dir(out_dir)
    results = sorted(results, key=lambda d: d["h_threshold"])
    h = [r["h_threshold"] for r in results]
    pmn = [r["pm_count"] for r in results]
    mk = [r["makespan"] for r in results]
    risk = [r["safety_risk"] for r in results]
    imb = [r["load_imbalance"] for r in results]

    fig = plt.figure(figsize=(6,4)); plt.plot(h, pmn, marker='o'); plt.xscale("log")
    plt.xlabel("h_threshold (log)"); plt.ylabel("#PM"); plt.title("Sensitivity: PM count vs h_threshold")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "sens_hthr_pmcount.png"), dpi=170); plt.close(fig)

    fig = plt.figure(figsize=(6,4)); plt.plot(h, risk, marker='o'); plt.xscale("log")
    plt.xlabel("h_threshold (log)"); plt.ylabel("Safety risk"); plt.title("Sensitivity: risk vs h_threshold")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "sens_hthr_risk.png"), dpi=170); plt.close(fig)

    fig = plt.figure(figsize=(6,4)); plt.plot(h, mk, marker='o'); plt.xscale("log")
    plt.xlabel("h_threshold (log)"); plt.ylabel("Makespan"); plt.title("Sensitivity: makespan vs h_threshold")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "sens_hthr_makespan.png"), dpi=170); plt.close(fig)

    fig = plt.figure(figsize=(6,4)); plt.plot(h, imb, marker='o'); plt.xscale("log")
    plt.xlabel("h_threshold (log)"); plt.ylabel("Load imbalance"); plt.title("Sensitivity: load imbalance vs h_threshold")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "sens_hthr_imbalance.png"), dpi=170); plt.close(fig)

    with open(os.path.join(out_dir, "sensitivity_hthr.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["h_threshold","pm_count","makespan","maintenance_cost","safety_risk","load_imbalance"])
        for r in results:
            w.writerow([r["h_threshold"], r["pm_count"], r["makespan"], "", r["safety_risk"], r["load_imbalance"]])

# =========================
# 主程式
# =========================

def main(csv_path: Optional[str] = None, json_path: Optional[str] = None):
    out_dir = "/mnt/data/visualization/safety_stage2_nsga3"
    _ensure_dir(out_dir)

    if csv_path and os.path.isfile(csv_path):
        inst = load_instance_from_csv(csv_path)
    elif json_path and os.path.isfile(json_path):
        inst = load_instance_from_json(json_path)
    else:
        inst = load_sample_instance()

    params_dir = os.path.join(out_dir, "report"); _ensure_dir(params_dir)
    with open(os.path.join(params_dir, "params.json"), "w") as f:
        json.dump({
            "beta": inst.weibull_beta,
            "eta": inst.weibull_eta,
            "pm_duration_min": inst.pm_duration,
            "cost_pm_fixed": inst.cost_pm_fixed,
            "cost_pm_time_per_min": inst.cost_pm_time,
            "delay_cost_per_min": inst.delay_cost_rate,
            "h_threshold_per_min": inst.h_threshold,
            "use_mrl_gate": inst.use_mrl_gate,
            "mrl_threshold_min": inst.mrl_threshold,
            "window_scan_points": inst.window_scan_points,
            "enforce_idle_pm": inst.enforce_idle_pm,
            "maint_crew_limit": inst.maint_crew_limit,
            "alpha_switch": inst.alpha_switch,
            "cond_coeffs": inst.cond_coeffs,
            "cond_names": inst.cond_names,
            "mr_time_min": inst.mr_time
        }, f, indent=2)

    plot_gantt_baseline(inst, os.path.join(out_dir, "fig_baseline_stage1.png"))
    base_mk = baseline_makespan(inst)

    problem = Stage2PMProblem(inst)
    result = nsga3_run(problem, pop_size=72, generations=40,
                       ref_divisions=12, sbx_eta=15.0, sbx_prob=0.9,
                       pm_eta=20.0, pm_prob=0.06, seed=7)

    popF = result["popF"]; paretoF = result["paretoF"]
    rep = result["representative"]; rep_F = rep["F"]; rep_details = rep["details"]
    ideal_hist = result["ideal_hist"]

    plot_pareto_2d_pairs(popF, paretoF, os.path.join(out_dir, "pareto"))
    plot_convergence(ideal_hist, os.path.join(out_dir, "pareto"))

    plot_failure_rate_panels(inst, os.path.join(out_dir, "fig_failure_panels.png"),
                             with_ops_blocks=True,
                             pm_segments=rep_details["pm_segments"],
                             mr_segments=rep_details["mr_segments"])

    plot_optimal_vs_actual_pm(inst, rep_details["pm_segments"], os.path.join(out_dir, "fig_opt_vs_actual_pm.png"))

    plot_gantt(rep_details["op_segments"], rep_details["pm_segments"],
               os.path.join(out_dir, "fig_gantt_optimized.png"),
               title="Optimized (NSGA-III, with PM+MR)",
               mr_segments=rep_details["mr_segments"])

    plot_gantt_side_by_side(inst, rep_details["op_segments"], rep_details["pm_segments"],
                            os.path.join(out_dir, "fig_gantt_side_by_side.png"),
                            opt_mrs=rep_details["mr_segments"])

    plot_cost_breakdown(inst, base_mk, rep_F[0], rep_F[1], os.path.join(out_dir, "cost"))

    thresholds = [5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
    sens = sweep_h_threshold(inst, thresholds, pop_size=48, generations=25, seed=11)
    plot_hthr_sensitivity(sens, os.path.join(out_dir, "sensitivity"))

    with open(os.path.join(out_dir, "final_population.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["makespan","maintenance_cost","safety_risk","load_imbalance"])
        for r in popF: w.writerow(list(r))
    with open(os.path.join(out_dir, "pareto_front.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["makespan","maintenance_cost","safety_risk","load_imbalance"])
        for r in paretoF: w.writerow(list(r))

    print("[代表解 F] (mk, mc, risk, lb) =", rep_F)
    print("Baseline makespan =", base_mk)
    print("輸出資料夾:", os.path.abspath(out_dir))

if __name__ == "__main__":
    # main(csv_path="stage1_baseline.csv")  # CSV 欄位：machine_id,start,end,job_id,load_factor[,cond_0,cond_1,...]
    # main(json_path="stage1_baseline.json")
    main()
