import random

class HybridDspoVns:
    def __init__(self, jobs, machines, config):
        """
        初始化混合 DPSO-VNS 演算法。

        參數:
            jobs (list): 工件物件列表。
            machines (list): 機器物件列表。
            config (object): 包含所有演算法和成本參數的設定物件。
        """
        self.jobs = jobs
        self.machines = machines
        self.config = config
        self.population_size = config.POPULATION_SIZE
        self.max_iterations_dspo = config.MAX_ITERATIONS_DPSO
        self.max_iterations_vns = config.MAX_ITERATIONS_VNS
        self.g_best = None # 全局最佳解

    def run_stage_one(self):
        """
        執行第一階段的求解，目標是最小化 TTC + TBC。
        """
        # 1. 初始化粒子群
        population = self._initialize_population()

        # 2. DPSO 全局搜索
        for i in range(self.max_iterations_dspo):
            for particle in population:
                self._update_particle_dspo(particle)
            
            # 更新全局最佳解
            best_in_pop = min(population, key=lambda p: self.calculate_fitness_stage_one(p))
            if self.g_best is None or self.calculate_fitness_stage_one(best_in_pop) < self.calculate_fitness_stage_one(self.g_best):
                self.g_best = best_in_pop

        # 3. VNS 局部增強
        # 根據論文，對 DPSO 找到的 top 10% 解進行 VNS 搜索
        top_10_percent = sorted(population, key=lambda p: self.calculate_fitness_stage_one(p))[:int(0.1 * self.population_size)]

        for particle in top_10_percent:
            self._enhance_with_vns(particle)

        # 更新全局最佳解
        best_after_vns = min(top_10_percent, key=lambda p: self.calculate_fitness_stage_one(p))
        if self.calculate_fitness_stage_one(best_after_vns) < self.calculate_fitness_stage_one(self.g_best):
            self.g_best = best_after_vns

        # 解碼最佳解以獲得排程
        final_schedule = self._decode_particle(self.g_best)
        objective_value = self.calculate_fitness_stage_one(self.g_best)

        return final_schedule, objective_value

    def _initialize_population(self):
        """
        初始化粒子群，每個粒子代表一個可行的排程方案。
        採用論文中提到的四種方法生成初始 MS。
        """
        population = []
        for _ in range(self.population_size):
            # 1. 生成 OS (操作序列)
            # 簡單地將所有工件的所有工序ID平鋪到一個列表中
            os_part = []
            for job in self.jobs:
                os_part.extend([job.job_id] * len(job.operations))
            random.shuffle(os_part)

            # 2. 生成 MS (機器序列)
            # 根據論文，使用四種方法按比例生成
            # 30% global selection I, 30% global selection II, 20% local selection, 20% random
            # 這裡為了簡化，我們先只用隨機選擇
            ms_part = []
            op_counters = {job.job_id: 0 for job in self.jobs}
            for job_id in os_part:
                op_id = op_counters[job_id]
                operation = self.jobs[job_id].operations[op_id]
                
                # 從候選機器中隨機選一個
                machine_id = random.choice(list(operation.candidate_machines.keys()))
                ms_part.append(machine_id)
                
                op_counters[job_id] += 1

            particle = {'os': os_part, 'ms': ms_part, 'p_best': None, 'fitness': float('inf')}
            particle['p_best'] = particle.copy()
            population.append(particle)
        return population

    def _decode_particle(self, particle):
        """
        將粒子的編碼 (OS, MS) 轉換為一個實際的排程方案。
        返回一個包含所有工序排程資訊的字典和 makespan。
        """
        schedule = {}
        machine_end_times = {m.machine_id: 0 for m in self.machines}
        job_end_times = {j.job_id: 0 for j in self.jobs}
        
        op_counters = {job.job_id: 0 for job in self.jobs}

        for i in range(len(particle['os'])):
            job_id = particle['os'][i]
            machine_id = particle['ms'][i]
            
            op_id = op_counters[job_id]
            # 找到對應的 Job 物件
            current_job = next((j for j in self.jobs if j.job_id == job_id), None)
            if current_job is None:
                raise ValueError(f"Job with ID {job_id} not found during initialization.")
            operation = current_job.operations[op_id]
            processing_time = operation.candidate_machines[machine_id]

            # 計算工序的開始時間
            start_time = max(machine_end_times[machine_id], job_end_times[job_id])
            end_time = start_time + processing_time

            # 更新時間
            machine_end_times[machine_id] = end_time
            job_end_times[job_id] = end_time
            
            # 記錄排程
            schedule[(job_id, op_id)] = {
                'machine': machine_id,
                'start_time': start_time,
                'end_time': end_time
            }
            
            op_counters[job_id] += 1
            
        makespan = max(machine_end_times.values())
        return schedule, makespan

    def _update_particle_dspo(self, particle):
        """
        使用 DPSO 的規則更新單個粒子。
        包含個體搜索、局部學習和全局學習。
        """
        # 為了簡化，這裡的 fitness 計算會被多次調用
        # 在完整的實作中，可以優化這部分，避免重複計算
        current_fitness = self.calculate_fitness_stage_one(particle)

        # 1. 個體搜索 (Mutation)
        # 論文中對應 V_ma 和 V_op 算子
        mutated_particle = self._mutate(particle)
        mutated_fitness = self.calculate_fitness_stage_one(mutated_particle)
        
        new_particle = particle
        new_fitness = current_fitness

        if mutated_fitness < new_fitness:
            new_particle = mutated_particle
            new_fitness = mutated_fitness

        # 2. 局部學習 (Crossover with p_best)
        # 論文中對應 C_job 和 C_op 算子
        crossed_particle_p = self._crossover(new_particle, particle['p_best'])
        crossed_fitness_p = self.calculate_fitness_stage_one(crossed_particle_p)

        if crossed_fitness_p < new_fitness:
            new_particle = crossed_particle_p
            new_fitness = crossed_fitness_p

        # 3. 全局學習 (Crossover with g_best)
        if self.g_best is not None:
            crossed_particle_g = self._crossover(new_particle, self.g_best)
            crossed_fitness_g = self.calculate_fitness_stage_one(crossed_particle_g)
            if crossed_fitness_g < new_fitness:
                new_particle = crossed_particle_g
                new_fitness = crossed_fitness_g
        
        particle = new_particle

        # 更新粒子的 p_best
        if new_fitness < self.calculate_fitness_stage_one(particle['p_best']):
            particle['p_best'] = particle.copy()
            particle['fitness'] = new_fitness

    def _mutate(self, particle):
        """
        對粒子進行變異操作，包含 V_ma 和 V_op 兩種鄰域結構。
        """
        new_particle = particle.copy()
        
        # 論文中是先用 V_ma，再用 V_op
        p_prime = self._v_ma(new_particle)
        p_double_prime = self._v_op(p_prime)
        
        return p_double_prime

    def _v_ma(self, particle):
        """
        V_ma (Machine Re-selection) 操作。
        隨機選擇一個工序，並為其重新選擇一個可用的機器。
        """
        p = particle.copy()
        op_index = random.randrange(len(p['ms']))
        job_id = p['os'][op_index]
        
        op_count = 0
        for i in range(op_index + 1):
            if p['os'][i] == job_id:
                op_count += 1
        op_id = op_count - 1

        # 找到對應的 Job 物件
        current_job = next((j for j in self.jobs if j.job_id == job_id), None)
        if current_job is None:
            raise ValueError(f"Job with ID {job_id} not found during _v_ma.")
        operation = current_job.operations[op_id]
        if len(operation.candidate_machines) > 1:
            # 從候選機器中選擇一個不同於當前的機器
            current_machine = p['ms'][op_index]
            possible_choices = [m for m in operation.candidate_machines.keys() if m != current_machine]
            if possible_choices:
                new_machine = random.choice(possible_choices)
                p['ms'][op_index] = new_machine
        return p

    def _v_op(self, particle):
        """
        V_op (Operation Sequence Change) 操作。
        隨機選擇兩個位置並交換其 OS 和 MS。
        """
        p = particle.copy()
        pos1, pos2 = random.sample(range(len(p['os'])), 2)
        
        # 交換 OS 和 MS 以保持配對
        p['os'][pos1], p['os'][pos2] = p['os'][pos2], p['os'][pos1]
        p['ms'][pos1], p['ms'][pos2] = p['ms'][pos2], p['ms'][pos1]
        return p

    def _crossover(self, p1, p2):
        """
        對兩個粒子進行交叉操作，包含 C_job 和 C_op 兩種方式。
        """
        # 隨機選擇一種交叉方式
        if random.random() < 0.5:
            return self._c_job(p1, p2)
        else:
            return self._c_op(p1, p2)

    def _c_job(self, p1, p2):
        """
        C_job (Job-based Crossover) 操作。
        """
        p1_os, p1_ms = p1['os'], p1['ms']
        p2_os, p2_ms = p2['os'], p2['ms']
        
        num_jobs = len(self.jobs)
        job_indices = list(range(num_jobs))
        random.shuffle(job_indices)
        
        set1_size = random.randint(1, num_jobs - 1)
        job_set1 = set(job_indices[:set1_size])
        job_set2 = set(job_indices[set1_size:])

        # 從 p1 繼承 job_set1 的部分
        new_os1 = [op for op in p1_os if op in job_set1]

        # 從 p2 繼承 job_set2 的部分
        new_os2 = [op for op in p2_os if op in job_set2]

        # 合併產生新的 OS
        new_os = new_os1 + new_os2

        # Re-generate new_ms to ensure validity for the new_os
        new_ms = []
        op_counters = {job.job_id: 0 for job in self.jobs} # Reset op_counters for new_os
        for job_id in new_os:
            op_id = op_counters[job_id]
            current_job = next((j for j in self.jobs if j.job_id == job_id), None)
            if current_job is None:
                raise ValueError(f"Job with ID {job_id} not found during _c_job.")
            operation = current_job.operations[op_id]
            
            # Randomly choose a valid machine for this operation
            machine_id = random.choice(list(operation.candidate_machines.keys()))
            new_ms.append(machine_id)
            
            op_counters[job_id] += 1

        return {'os': new_os, 'ms': new_ms, 'p_best': p1['p_best'], 'fitness': p1['fitness']}

    def _c_op(self, p1, p2):
        """
        C_op (Operation-based Crossover) 操作。
        """
        p1_os, p1_ms = p1['os'], p1['ms']
        p2_os, p2_ms = p2['os'], p2['ms']
        
        new_os = []
        p2_os_copy = list(p2_os)
        crossover_points = sorted(random.sample(range(len(p1_os)), 2))
        start, end = crossover_points[0], crossover_points[1]
        middle_os = p1_os[start:end]
        for op_val in middle_os:
            if op_val in p2_os_copy:
                p2_os_copy.remove(op_val)
        new_os = p2_os_copy[:start] + middle_os + p2_os_copy[start:]

        # Re-generate new_ms to ensure validity for the new_os
        new_ms = []
        op_counters = {job.job_id: 0 for job in self.jobs} # Reset op_counters for new_os
        for job_id in new_os:
            op_id = op_counters[job_id]
            current_job = next((j for j in self.jobs if j.job_id == job_id), None)
            if current_job is None:
                raise ValueError(f"Job with ID {job_id} not found during _c_op.")
            operation = current_job.operations[op_id]
            
            # Randomly choose a valid machine for this operation
            machine_id = random.choice(list(operation.candidate_machines.keys()))
            new_ms.append(machine_id)
            
            op_counters[job_id] += 1

        return {'os': new_os, 'ms': new_ms, 'p_best': p1['p_best'], 'fitness': p1['fitness']}

    def _enhance_with_vns(self, particle):
        """
        使用 VNS 的鄰域搜索來增強單個粒子。
        """
        best_particle = particle
        best_fitness = self.calculate_fitness_stage_one(best_particle)

        for _ in range(self.max_iterations_vns):
            # 隨機選擇一個鄰域結構
            if random.random() < 0.5:
                # S_ma: 改變機器分配
                new_particle = self._vns_change_machine(particle)
            else:
                # V_op: 交換工序順序
                new_particle = self._vns_swap_operations(particle)
            
            new_fitness = self.calculate_fitness_stage_one(new_particle)

            if new_fitness < best_fitness:
                best_particle = new_particle
                best_fitness = new_fitness
        
        return best_particle

    def _vns_change_machine(self, particle):
        """
        VNS 的 S_ma 操作：為隨機一個工序更換機器。
        """
        new_particle = particle.copy()
        op_index = random.randrange(len(new_particle['ms']))
        job_id = new_particle['os'][op_index]
        
        op_count = 0
        for i in range(op_index + 1):
            if new_particle['os'][i] == job_id:
                op_count += 1
        op_id = op_count - 1

        # 找到對應的 Job 物件
        current_job = next((j for j in self.jobs if j.job_id == job_id), None)
        if current_job is None:
            raise ValueError(f"Job with ID {job_id} not found during _vns_change_machine.")
        operation = current_job.operations[op_id]
        if len(operation.candidate_machines) > 1:
            new_machine = random.choice(list(operation.candidate_machines.keys()))
            new_particle['ms'][op_index] = new_machine

        return new_particle

    def _vns_swap_operations(self, particle):
        """
        VNS 的 V_op 操作：隨機交換兩個工序的位置。
        """
        new_particle = particle.copy()
        pos1, pos2 = random.sample(range(len(new_particle['os'])), 2)
        new_particle['os'][pos1], new_particle['os'][pos2] = new_particle['os'][pos2], new_particle['os'][pos1]
        # 注意：交換 OS 後，MS 也需要對應交換，以保持 (OS, MS) 的配對關係
        new_particle['ms'][pos1], new_particle['ms'][pos2] = new_particle['ms'][pos2], new_particle['ms'][pos1]
        return new_particle

    def calculate_fitness_stage_one(self, particle):
        """
        計算第一階段的適應度函數值 (TTC + TBC)。
        論文中的目標是最小化延遲和負載均衡，這裡我們簡化為 Makespan 和負載均衡。
        """
        schedule, makespan = self._decode_particle(particle)

        # 1. 計算 Makespan 成本 (作為 TTC 的代理)
        # 注意：self.config 現在是整個 config 物件，成本參數在 COST_PARAMS 內
        cost_params = self.config.COST_PARAMS
        makespan_cost = cost_params['w_makespan'] * makespan

        # 2. 計算工作負載平衡懲罰 (TBC)
        machine_workloads = {m.machine_id: 0 for m in self.machines}
        for op_info in schedule.values():
            duration = op_info['end_time'] - op_info['start_time']
            machine_workloads[op_info['machine']] += duration
        
        if not self.machines:
            avg_workload = 0
            workload_variance = 0
        else:
            avg_workload = sum(machine_workloads.values()) / len(self.machines)
            workload_variance = sum((wl - avg_workload)**2 for wl in machine_workloads.values()) / len(self.machines)
        
        tbc = cost_params['workload_balancing_penalty_weight'] * workload_variance

        return makespan_cost + tbc

    def _decode_particle(self, particle):
        """
        將粒子的編碼 (OS, MS) 轉換為一個實際的排程方案。
        返回一個包含所有工序排程資訊的字典和 makespan。
        """
        schedule = {}
        machine_end_times = {m.machine_id: 0 for m in self.machines}
        job_end_times = {j.job_id: 0 for j in self.jobs}
        
        op_counters = {j.job_id: 0 for j in self.jobs}

        for i in range(len(particle['os'])):
            job_id = particle['os'][i]
            machine_id = particle['ms'][i]
            
            op_id = op_counters[job_id]
            # 找到對應的 Job 物件
            current_job = next((j for j in self.jobs if j.job_id == job_id), None)
            if current_job is None:
                raise ValueError(f"Job with ID {job_id} not found.")

            operation = current_job.operations[op_id]
            processing_time = operation.candidate_machines[machine_id]

            # 計算工序的開始時間 (需滿足工序限制和機器限制)
            start_time = max(machine_end_times[machine_id], job_end_times[job_id])
            end_time = start_time + processing_time

            # 更新機器的完工時間和工件的完工時間
            machine_end_times[machine_id] = end_time
            job_end_times[job_id] = end_time
            
            # 記錄排程結果
            schedule[(job_id, op_id)] = {
                'machine': machine_id,
                'start_time': start_time,
                'end_time': end_time
            }
            
            op_counters[job_id] += 1
            
        makespan = max(machine_end_times.values()) if machine_end_times else 0
        return schedule, makespan