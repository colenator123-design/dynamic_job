import copy
from dataclasses import dataclass

import numpy as np
import torch

from solution_methods.FJSP_DRL_Dynamic.src.load_data import load_feats_from_case, nums_detec


@dataclass
class EnvState:
    """
    Class for the state of the environment
    """
    # static
    opes_appertain_batch: torch.Tensor = None
    ope_pre_adj_batch: torch.Tensor = None
    ope_sub_adj_batch: torch.Tensor = None
    end_ope_biases_batch: torch.Tensor = None
    nums_opes_batch: torch.Tensor = None

    # dynamic
    batch_idxes: torch.Tensor = None
    feat_opes_batch: torch.Tensor = None
    feat_mas_batch: torch.Tensor = None
    proc_times_batch: torch.Tensor = None
    ope_ma_adj_batch: torch.Tensor = None
    time_batch: torch.Tensor = None

    mask_job_procing_batch: torch.Tensor = None
    mask_job_finish_batch: torch.Tensor = None
    mask_ma_procing_batch: torch.Tensor = None
    ope_step_batch: torch.Tensor = None

    def update(self, batch_idxes, feat_opes_batch, feat_mas_batch, proc_times_batch, ope_ma_adj_batch,
               mask_job_procing_batch, mask_job_finish_batch, mask_ma_procing_batch, ope_step_batch, time):
        self.batch_idxes = batch_idxes
        self.feat_opes_batch = feat_opes_batch
        self.feat_mas_batch = feat_mas_batch
        self.proc_times_batch = proc_times_batch
        self.ope_ma_adj_batch = ope_ma_adj_batch

        self.mask_job_procing_batch = mask_job_procing_batch
        self.mask_job_finish_batch = mask_job_finish_batch
        self.mask_ma_procing_batch = mask_ma_procing_batch
        self.ope_step_batch = ope_step_batch
        self.time_batch = time


def convert_feat_job_2_ope(feat_job_batch, opes_appertain_batch):
    """
    Convert job features into operation features (such as dimension)
    """
    return feat_job_batch.gather(1, opes_appertain_batch)


class FJSPEnv_training():
    """
    FJSP environment
    """

    def __init__(self, case, env_paras, data_source='case'):
        """
        :param case: The instance generator or the addresses of the instances
        :param env_paras: A dictionary of parameters for the environment
        :param data_source: Indicates that the instances came from a generator or files
        """

        # load paras
        # static
        self.batch_size = env_paras["batch_size"]  # Number of parallel instances during training
        self.num_jobs = env_paras["num_jobs"]  # Number of jobs
        self.num_mas = env_paras["num_mas"]  # Number of machines
        self.paras = env_paras  # Parameters
        self.device = env_paras["device"]  # Computing device for PyTorch
        # load instance
        num_data = 8  # The amount of data extracted from instance
        tensors = [[] for _ in range(num_data)]
        self.num_opes = 0
        lines = []
        if data_source == 'case':  # Generate instances through generators
            arrival_times_list = []
            for i in range(self.batch_size):
                instance_lines, arrival_times = case.get_case(i)
                lines.append(instance_lines)  # Generate an instance and save it
                arrival_times_list.append(torch.tensor(arrival_times, dtype=torch.float32))
                num_jobs, num_mas, num_opes = nums_detec(lines[i])
                # Records the maximum number of operations in the parallel instances
                self.num_opes = max(self.num_opes, num_opes)
            self.job_arrival_times_batch = torch.stack(arrival_times_list).to(self.device)
        else:  # Load instances from files
            for i in range(self.batch_size):
                with open(case[i]) as file_object:
                    line = file_object.readlines()
                    lines.append(line)
                num_jobs, num_mas, num_opes = nums_detec(lines[i])
                self.num_opes = max(self.num_opes, num_opes)
            # For static problems from files, arrival times are all 0
            self.job_arrival_times_batch = torch.zeros(self.batch_size, self.num_jobs, dtype=torch.float32).to(self.device)
        # load feats
        for i in range(self.batch_size):
            load_data = load_feats_from_case(lines[i], num_mas, self.num_opes)
            for j in range(num_data):
                tensors[j].append(load_data[j])

        # dynamic feats
        # shape: (batch_size, num_opes, num_mas)
        self.proc_times_batch = torch.stack(tensors[0], dim=0)
        # shape: (batch_size, num_opes, num_mas)
        self.ope_ma_adj_batch = torch.stack(tensors[1], dim=0).long()
        # shape: (batch_size, num_opes, num_opes), for calculating the cumulative amount along the path of each job
        self.cal_cumul_adj_batch = torch.stack(tensors[7], dim=0).float()

        # static feats
        # shape: (batch_size, num_opes, num_opes)
        self.ope_pre_adj_batch = torch.stack(tensors[2], dim=0)
        # shape: (batch_size, num_opes, num_opes)
        self.ope_sub_adj_batch = torch.stack(tensors[3], dim=0)
        # shape: (batch_size, num_opes), represents the mapping between operations and jobs
        self.opes_appertain_batch = torch.stack(tensors[4], dim=0).long()
        # shape: (batch_size, num_jobs), the id of the first operation of each job
        self.num_ope_biases_batch = torch.stack(tensors[5], dim=0).long()
        # shape: (batch_size, num_jobs), the number of operations for each job
        self.nums_ope_batch = torch.stack(tensors[6], dim=0).long()
        # shape: (batch_size, num_jobs), the id of the last operation of each job
        self.end_ope_biases_batch = self.num_ope_biases_batch + self.nums_ope_batch - 1
        # shape: (batch_size), the number of operations for each instance
        self.nums_opes = torch.sum(self.nums_ope_batch, dim=1)

        # Define Reward Weights (hardcoded for now)
        self.w1_makespan = 1.0 # Weight for makespan
        self.w2_workload_imbalance = 0.1 # Weight for workload imbalance
        self.w3_machine_switching = 0.5 # Weight for machine switching

        # dynamic variable
        self.batch_idxes = torch.arange(self.batch_size)  # Uncompleted instances
        self.time = torch.zeros(self.batch_size)  # Current time of the environment
        self.N = torch.zeros(self.batch_size).int()  # Count scheduled operations
        # shape: (batch_size, num_jobs), the id of the current operation (be waiting to be processed) of each job
        self.ope_step_batch = copy.deepcopy(self.num_ope_biases_batch)
        '''
        features, dynamic
            ope:
                Status
                Number of neighboring machines
                Processing time
                Number of unscheduled operations in the job
                Job completion time
                Start time
            ma:
                Number of neighboring operations
                Available time
                Utilization
        '''
        # Generate raw feature vectors
        feat_opes_batch = torch.zeros(size=(self.batch_size, self.paras["ope_feat_dim"], self.num_opes))
        feat_mas_batch = torch.zeros(size=(self.batch_size, self.paras["ma_feat_dim"], num_mas))

        feat_opes_batch[:, 1, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=2)
        feat_opes_batch[:, 2, :] = torch.sum(self.proc_times_batch, dim=2).div(feat_opes_batch[:, 1, :] + 1e-9)
        feat_opes_batch[:, 3, :] = convert_feat_job_2_ope(self.nums_ope_batch, self.opes_appertain_batch)
        feat_opes_batch[:, 5, :] = torch.bmm(feat_opes_batch[:, 2, :].unsqueeze(1),
                                             self.cal_cumul_adj_batch).squeeze()
        end_time_batch = (feat_opes_batch[:, 5, :] + feat_opes_batch[:, 2, :]).gather(1, self.end_ope_biases_batch)
        feat_opes_batch[:, 4, :] = convert_feat_job_2_ope(end_time_batch, self.opes_appertain_batch)
        feat_mas_batch[:, 0, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=1)
        # Initialize new operation features (indices 6 and 7) to default values
        feat_opes_batch[:, 6, :] = -1.0 # Predecessor Machine ID
        feat_opes_batch[:, 7, :] = 0.0 # Average Eligible Machine Load
        self.feat_opes_batch = feat_opes_batch
        self.feat_mas_batch = feat_mas_batch

        # Masks of current status, dynamic
        # shape: (batch_size, num_jobs), True for jobs in process
        self.mask_job_procing_batch = torch.full(size=(self.batch_size, num_jobs), dtype=torch.bool, fill_value=False)
        # shape: (batch_size, num_jobs), True for completed jobs
        self.mask_job_finish_batch = torch.full(size=(self.batch_size, num_jobs), dtype=torch.bool, fill_value=False)
        # shape: (batch_size, num_mas), True for machines in process
        self.mask_ma_procing_batch = torch.full(size=(self.batch_size, num_mas), dtype=torch.bool, fill_value=False)
        # shape: (batch_size, num_jobs), True for jobs that have arrived
        self.mask_job_arrived_batch = self.job_arrival_times_batch <= self.time.unsqueeze(1)
        '''
        Partial Schedule (state) of jobs/operations, dynamic
            Status
            Allocated machines
            Start time
            End time
        '''
        self.schedules_batch = torch.zeros(size=(self.batch_size, self.num_opes, 4))
        self.schedules_batch[:, :, 2] = feat_opes_batch[:, 5, :]
        self.schedules_batch[:, :, 3] = feat_opes_batch[:, 5, :] + feat_opes_batch[:, 2, :]
        '''
        Partial Schedule (state) of machines, dynamic
            idle
            available_time
            utilization_time
            id_ope
        '''
        self.machines_batch = torch.zeros(size=(self.batch_size, self.num_mas, 4))
        self.machines_batch[:, :, 0] = torch.ones(size=(self.batch_size, self.num_mas))

        self.makespan_batch = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]  # shape: (batch_size)
        self.done_batch = self.mask_job_finish_batch.all(dim=1)  # shape: (batch_size)

        self.state = EnvState(batch_idxes=self.batch_idxes,
                              feat_opes_batch=self.feat_opes_batch, feat_mas_batch=self.feat_mas_batch,
                              proc_times_batch=self.proc_times_batch, ope_ma_adj_batch=self.ope_ma_adj_batch,
                              ope_pre_adj_batch=self.ope_pre_adj_batch, ope_sub_adj_batch=self.ope_sub_adj_batch,
                              mask_job_procing_batch=self.mask_job_procing_batch,
                              mask_job_finish_batch=self.mask_job_finish_batch,
                              mask_ma_procing_batch=self.mask_ma_procing_batch,
                              opes_appertain_batch=self.opes_appertain_batch,
                              ope_step_batch=self.ope_step_batch,
                              end_ope_biases_batch=self.end_ope_biases_batch,
                              time_batch=self.time, nums_opes_batch=self.nums_opes)

        # Save initial data for reset
        self.old_proc_times_batch = copy.deepcopy(self.proc_times_batch)
        self.old_ope_ma_adj_batch = copy.deepcopy(self.ope_ma_adj_batch)
        self.old_cal_cumul_adj_batch = copy.deepcopy(self.cal_cumul_adj_batch)
        self.old_feat_opes_batch = copy.deepcopy(self.feat_opes_batch)
        self.old_feat_mas_batch = copy.deepcopy(self.feat_mas_batch)
        self.old_state = copy.deepcopy(self.state)
        self.old_job_arrival_times_batch = copy.deepcopy(self.job_arrival_times_batch)

    def step(self, actions):
        """
        Environment transition function
        """
        opes = actions[0, :]
        mas = actions[1, :]
        jobs = actions[2, :]
        self.N += 1

        # Removed unselected O-M arcs of the scheduled operations
        remain_ope_ma_adj = torch.zeros(size=(self.batch_size, self.num_mas), dtype=torch.int64)
        remain_ope_ma_adj[self.batch_idxes, mas] = 1
        self.ope_ma_adj_batch[self.batch_idxes, opes] = remain_ope_ma_adj[self.batch_idxes, :]
        self.proc_times_batch *= self.ope_ma_adj_batch

        # Update for some O-M arcs are removed, such as 'Status', 'Number of neighboring machines' and 'Processing time'
        proc_times = self.proc_times_batch[self.batch_idxes, opes, mas]
        self.feat_opes_batch[self.batch_idxes, :3, opes] = torch.stack(
            (torch.ones(self.batch_idxes.size(0), dtype=torch.float),
             torch.ones(self.batch_idxes.size(0), dtype=torch.float),
             proc_times), dim=1)
        
        # Calculate Predecessor Machine ID (index 6) - Vectorized
        is_first_op_of_job = (opes == self.num_ope_biases_batch[self.batch_idxes, jobs])
        predecessor_opes_abs_ids = opes - 1
        
        # Get status and machine ID of predecessor operations
        predecessor_status = self.schedules_batch[self.batch_idxes, predecessor_opes_abs_ids, 0]
        predecessor_machines = self.schedules_batch[self.batch_idxes, predecessor_opes_abs_ids, 1]
        
        # Condition: not first op of job AND predecessor is scheduled
        condition = (~is_first_op_of_job) & (predecessor_status == 1)
        
        # Initialize predecessor machine IDs for the *scheduled* operations to -1
        predecessor_machine_ids_for_scheduled_ops = torch.full_like(opes, -1, dtype=torch.float)
        
        # Update where condition is true
        predecessor_machine_ids_for_scheduled_ops = torch.where(
            condition,
            predecessor_machines,
            predecessor_machine_ids_for_scheduled_ops
        )
        
        # Update the feature in feat_opes_batch for the *scheduled* operations
        self.feat_opes_batch[self.batch_idxes, 6, opes] = predecessor_machine_ids_for_scheduled_ops.float()
        last_opes = torch.where(opes - 1 < self.num_ope_biases_batch[self.batch_idxes, jobs], self.num_opes - 1,
                                opes - 1)
        self.cal_cumul_adj_batch[self.batch_idxes, last_opes, :] = 0

        # Update 'Number of unscheduled operations in the job'
        start_ope = self.num_ope_biases_batch[self.batch_idxes, jobs]
        end_ope = self.end_ope_biases_batch[self.batch_idxes, jobs]
        for i in range(self.batch_idxes.size(0)):
            self.feat_opes_batch[self.batch_idxes[i], 3, start_ope[i]:end_ope[i] + 1] -= 1

        # Update 'Start time' and 'Job completion time'
        self.feat_opes_batch[self.batch_idxes, 5, opes] = self.time[self.batch_idxes]
        is_scheduled = self.feat_opes_batch[self.batch_idxes, 0, :]
        mean_proc_time = self.feat_opes_batch[self.batch_idxes, 2, :]
        start_times = self.feat_opes_batch[self.batch_idxes, 5, :] * is_scheduled  # real start time of scheduled opes
        un_scheduled = 1 - is_scheduled  # unscheduled opes
        estimate_times = torch.bmm((start_times + mean_proc_time).unsqueeze(1),
                                   self.cal_cumul_adj_batch[self.batch_idxes, :, :]).squeeze() * un_scheduled  # estimate start time of unscheduled opes
        self.feat_opes_batch[self.batch_idxes, 5, :] = start_times + estimate_times
        end_time_batch = (self.feat_opes_batch[self.batch_idxes, 5, :] + self.feat_opes_batch[self.batch_idxes, 2, :]).gather(1, self.end_ope_biases_batch[
                                                                                  self.batch_idxes, :])
        self.feat_opes_batch[self.batch_idxes, 4, :] = convert_feat_job_2_ope(end_time_batch, self.opes_appertain_batch[
                                                                                              self.batch_idxes, :])

        # Update partial schedule (state)
        self.schedules_batch[self.batch_idxes, opes, :2] = torch.stack((torch.ones(self.batch_idxes.size(0)), mas),
                                                                       dim=1)
        self.schedules_batch[self.batch_idxes, :, 2] = self.feat_opes_batch[self.batch_idxes, 5, :]
        self.schedules_batch[self.batch_idxes, :, 3] = self.feat_opes_batch[self.batch_idxes, 5, :] + \
                                                       self.feat_opes_batch[self.batch_idxes, 2, :]
        self.machines_batch[self.batch_idxes, mas, 0] = torch.zeros(self.batch_idxes.size(0))
        self.machines_batch[self.batch_idxes, mas, 1] = self.time[self.batch_idxes] + proc_times
        self.machines_batch[self.batch_idxes, mas, 2] += proc_times
        self.machines_batch[self.batch_idxes, mas, 3] = jobs.float()

        # Update feature vectors of machines
        self.feat_mas_batch[self.batch_idxes, 0, :] = torch.count_nonzero(self.ope_ma_adj_batch[self.batch_idxes, :, :],
                                                                          dim=1).float()
        self.feat_mas_batch[self.batch_idxes, 1, mas] = self.time[self.batch_idxes] + proc_times
        utiliz = self.machines_batch[self.batch_idxes, :, 2]
        cur_time = self.time[self.batch_idxes, None].expand_as(utiliz)
        utiliz = torch.minimum(utiliz, cur_time)
        utiliz = utiliz.div(self.time[self.batch_idxes, None] + 1e-9)
        self.feat_mas_batch[self.batch_idxes, 2, :] = utiliz

        # --- Calculate Workload Imbalance Penalty ---
        current_machine_workloads = self.feat_mas_batch[self.batch_idxes, 3, :] # Raw Workload
        if self.num_mas > 1:
            workload_mean = torch.mean(current_machine_workloads, dim=1, keepdim=True)
            workload_variance = torch.mean((current_machine_workloads - workload_mean)**2, dim=1)
            workload_imbalance_penalty = workload_variance
        else:
            workload_imbalance_penalty = torch.zeros(self.batch_idxes.size(0)).to(self.device) # No imbalance if only one machine

        # --- Calculate Machine Switching Penalty (Placeholder) ---
        # This requires Predecessor Machine ID feature, which is complex to implement with current tensor-only env
        # For now, a simple placeholder or 0
        machine_switching_penalty = torch.zeros(self.batch_idxes.size(0)).to(self.device)

        # Update other variable according to actions
        self.ope_step_batch[self.batch_idxes, jobs] += 1
        self.mask_job_procing_batch[self.batch_idxes, jobs] = True
        self.mask_ma_procing_batch[self.batch_idxes, mas] = True
        self.mask_job_finish_batch = torch.where(self.ope_step_batch == self.end_ope_biases_batch + 1,
                                                 True, self.mask_job_finish_batch)
        self.done_batch = self.mask_job_finish_batch.all(dim=1)
        self.done = self.done_batch.all()

        # Calculate Average Eligible Machine Load (index 7)
        avg_eligible_machine_loads = torch.zeros_like(self.feat_opes_batch[:, 7, :])
        unscheduled_opes_mask = (self.schedules_batch[:, :, 0] == 0)
        all_machine_workloads = self.feat_mas_batch[:, 3, :] # (batch_size, num_mas)

        for i, batch_id in enumerate(self.batch_idxes):
            instance_unscheduled_opes_mask = unscheduled_opes_mask[i]
            instance_unscheduled_opes_indices = torch.arange(self.num_opes)[instance_unscheduled_opes_mask]

            for op_idx in instance_unscheduled_opes_indices:
                eligible_machines_mask = (self.ope_ma_adj_batch[i, op_idx, :] == 1)
                eligible_machine_ids = torch.arange(self.num_mas)[eligible_machines_mask]

                if eligible_machine_ids.numel() > 0:
                    eligible_machine_workloads = all_machine_workloads[i, eligible_machine_ids]
                    avg_load = torch.mean(eligible_machine_workloads)
                    avg_eligible_machine_loads[i, op_idx] = avg_load

        self.feat_opes_batch[:, 7, :] = avg_eligible_machine_loads

        # Calculate new makespan before using it in reward calculation
        new_makespan = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]

        # Combine into new reward_batch
        makespan_reward_component = self.makespan_batch - new_makespan
        
        total_penalty = (self.w2_workload_imbalance * workload_imbalance_penalty) + \
                        (self.w3_machine_switching * machine_switching_penalty)
        
        self.reward_batch = makespan_reward_component - total_penalty
        self.makespan_batch = new_makespan

        # Check if there are still O-M pairs to be processed, otherwise the environment transits to the next time
        flag_trans_2_next_time = self.if_no_eligible()
        while ~((~((flag_trans_2_next_time == 0) & (~self.done_batch))).all()):
            self.next_time(flag_trans_2_next_time)
            flag_trans_2_next_time = self.if_no_eligible()

        # Update the vector for uncompleted instances
        mask_finish = (self.N + 1) <= self.nums_opes
        if ~(mask_finish.all()):
            self.batch_idxes = torch.arange(self.batch_size)[mask_finish]

        # Update state of the environment
        self.state.update(self.batch_idxes, self.feat_opes_batch, self.feat_mas_batch, self.proc_times_batch,
                          self.ope_ma_adj_batch, self.mask_job_procing_batch, self.mask_job_finish_batch,
                          self.mask_ma_procing_batch,
                          self.ope_step_batch, self.time)
        return self.state, self.reward_batch, self.done_batch, None

    def if_no_eligible(self):
        """
        Check if there are still O-M pairs to be processed
        """
        ope_step_batch = torch.where(self.ope_step_batch > self.end_ope_biases_batch,
                                     self.end_ope_biases_batch, self.ope_step_batch)
        op_proc_time = self.proc_times_batch.gather(1, ope_step_batch.unsqueeze(-1).expand(-1, -1,
                                                                                           self.proc_times_batch.size(
                                                                                               2)))
        ma_eligible = ~self.mask_ma_procing_batch.unsqueeze(1).expand_as(op_proc_time)
        job_arrived = self.mask_job_arrived_batch[:, :, None].expand_as(op_proc_time)
        job_eligible = job_arrived & ~(self.mask_job_procing_batch + self.mask_job_finish_batch)[:, :, None].expand_as(
            op_proc_time)
        flag_trans_2_next_time = torch.sum(
            torch.where(ma_eligible & job_eligible, op_proc_time.double(), 0.0).transpose(1, 2),
            dim=[1, 2])
        # shape: (batch_size)
        # An element value of 0 means that the corresponding instance has no eligible O-M pairs
        # in other words, the environment need to transit to the next time
        return flag_trans_2_next_time

    def next_time(self, flag_trans_2_next_time):
        """
        Transit to the next time, considering both machine completions and job arrivals.
        """
        # Instances that need to transit to the next time step
        flag_need_trans = (flag_trans_2_next_time == 0) & (~self.done_batch)

        # 1. Find the next machine completion time
        ma_available_times = self.machines_batch[:, :, 1]
        # Consider only machines that are currently processing and will finish after the current time
        ma_finish_times = torch.where(
            (ma_available_times > self.time[:, None]) & (self.machines_batch[:, :, 0] == 0),
            ma_available_times,
            torch.full_like(ma_available_times, float('inf'))
        )
        min_ma_finish_times, _ = torch.min(ma_finish_times, dim=1)

        # 2. Find the next job arrival time
        # Consider only jobs that have not yet arrived
        unarrived_job_times = torch.where(
            ~self.mask_job_arrived_batch,
            self.job_arrival_times_batch,
            torch.full_like(self.job_arrival_times_batch, float('inf'))
        )
        min_arrival_times, _ = torch.min(unarrived_job_times, dim=1)

        # 3. Determine the next event time
        next_event_times = torch.min(min_ma_finish_times, min_arrival_times)
        
        # For instances that don't need to transit, keep current time. Otherwise, update to next event time.
        new_time = torch.where(flag_need_trans, next_event_times, self.time)
        self.time = new_time

        # 4. Update masks based on the new time
        
        # Update job arrival mask first
        self.mask_job_arrived_batch = self.job_arrival_times_batch <= self.time.unsqueeze(1)

        # Detect which machines finished at exactly the new time
        # Condition: machine was busy, its available time matches the new time, and the instance needed a transition
        machines_just_finished = (self.machines_batch[:, :, 0] == 0) & \
                                 (self.machines_batch[:, :, 1] == self.time[:, None]) & \
                                 flag_need_trans[:, None]

        # Update machine availability and job processing status for finished jobs
        if torch.any(machines_just_finished):
            # Free up machines
            self.machines_batch[:, :, 0][machines_just_finished] = 1
            self.mask_ma_procing_batch[machines_just_finished] = False

            # Get the jobs that were running on these machines
            finished_jobs_on_machines = self.machines_batch[:, :, 3][machines_just_finished].long()
            
            # Create a mask to update only the relevant jobs in mask_job_procing_batch
            batch_indices = torch.where(machines_just_finished)[0]
            
            # Free up jobs
            self.mask_job_procing_batch[batch_indices, finished_jobs_on_machines] = False

        # Update machine utilization feature
        utiliz = self.machines_batch[:, :, 2]
        cur_time = self.time[:, None].expand_as(utiliz)
        utiliz = torch.minimum(utiliz, cur_time)
        utiliz = utiliz.div(self.time[:, None] + 1e-5)
        self.feat_mas_batch[:, 2, :] = utiliz
        
        # Update job completion mask
        self.mask_job_finish_batch = torch.where(self.ope_step_batch == self.end_ope_biases_batch + 1,
                                                 True, self.mask_job_finish_batch)

    def reset(self):
        """
        Reset the environment to its initial state
        """
        self.proc_times_batch = copy.deepcopy(self.old_proc_times_batch)
        self.ope_ma_adj_batch = copy.deepcopy(self.old_ope_ma_adj_batch)
        self.cal_cumul_adj_batch = copy.deepcopy(self.old_cal_cumul_adj_batch)
        self.feat_opes_batch = copy.deepcopy(self.old_feat_opes_batch)
        self.feat_mas_batch = copy.deepcopy(self.old_feat_mas_batch)
        self.state = copy.deepcopy(self.old_state)
        self.job_arrival_times_batch = copy.deepcopy(self.old_job_arrival_times_batch)

        self.batch_idxes = torch.arange(self.batch_size)
        self.time = torch.zeros(self.batch_size)
        self.N = torch.zeros(self.batch_size)
        self.mask_job_arrived_batch = self.job_arrival_times_batch <= self.time.unsqueeze(1)
        self.ope_step_batch = copy.deepcopy(self.num_ope_biases_batch)
        self.mask_job_procing_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool,
                                                 fill_value=False)
        self.mask_job_finish_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool,
                                                fill_value=False)
        self.mask_ma_procing_batch = torch.full(size=(self.batch_size, self.num_mas), dtype=torch.bool,
                                                fill_value=False)
        self.schedules_batch = torch.zeros(size=(self.batch_size, self.num_opes, 4))
        self.schedules_batch[:, :, 2] = self.feat_opes_batch[:, 5, :]
        self.schedules_batch[:, :, 3] = self.feat_opes_batch[:, 5, :] + self.feat_opes_batch[:, 2, :]
        self.machines_batch = torch.zeros(size=(self.batch_size, self.num_mas, 4))
        self.machines_batch[:, :, 0] = torch.ones(size=(self.batch_size, self.num_mas))

        self.makespan_batch = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]
        self.done_batch = self.mask_job_finish_batch.all(dim=1)
        return self.state

    def get_idx(self, id_ope, batch_id):
        """
        Get job and operation (relative) index based on instance index and operation (absolute) index
        """
        idx_job = max([idx for (idx, val) in enumerate(self.num_ope_biases_batch[batch_id]) if id_ope >= val])
        idx_ope = id_ope - self.num_ope_biases_batch[batch_id][idx_job]
        return idx_job, idx_ope

    def validate_gantt(self):
        """
        Verify whether the schedule is feasible
        """
        ma_gantt_batch = [[[] for _ in range(self.num_mas)] for __ in range(self.batch_size)]
        for batch_id, schedules in enumerate(self.schedules_batch):
            for i in range(int(self.nums_opes[batch_id])):
                step = schedules[i]
                ma_gantt_batch[batch_id][int(step[1])].append([i, step[2].item(), step[3].item()])
        proc_time_batch = self.proc_times_batch

        # Check whether there are overlaps and correct processing times on the machine
        flag_proc_time = 0
        flag_ma_overlap = 0
        flag = 0
        for k in range(self.batch_size):
            ma_gantt = ma_gantt_batch[k]
            proc_time = proc_time_batch[k]
            for i in range(self.num_mas):
                ma_gantt[i].sort(key=lambda s: s[1])
                for j in range(len(ma_gantt[i])):
                    if (len(ma_gantt[i]) <= 1) or (j == len(ma_gantt[i]) - 1):
                        break
                    if ma_gantt[i][j][2] > ma_gantt[i][j + 1][1]:
                        flag_ma_overlap += 1
                    if ma_gantt[i][j][2] - ma_gantt[i][j][1] != proc_time[ma_gantt[i][j][0]][i]:
                        flag_proc_time += 1
                    flag += 1

        # Check job order and overlap
        flag_ope_overlap = 0
        for k in range(self.batch_size):
            schedule = self.schedules_batch[k]
            nums_ope = self.nums_ope_batch[k]
            num_ope_biases = self.num_ope_biases_batch[k]
            for i in range(self.num_jobs):
                if int(nums_ope[i]) <= 1:
                    continue
                for j in range(int(nums_ope[i]) - 1):
                    step = schedule[num_ope_biases[i] + j]
                    step_next = schedule[num_ope_biases[i] + j + 1]
                    if step[3] > step_next[2]:
                        flag_ope_overlap += 1

        # Check whether there are unscheduled operations
        flag_unscheduled = 0
        for batch_id, schedules in enumerate(self.schedules_batch):
            count = 0
            for i in range(schedules.size(0)):
                if schedules[i][0] == 1:
                    count += 1
            add = 0 if (count == self.nums_opes[batch_id]) else 1
            flag_unscheduled += add

        if flag_ma_overlap + flag_ope_overlap + flag_proc_time + flag_unscheduled != 0:
            return False, self.schedules_batch
        else:
            return True, self.schedules_batch

    def close(self):
        pass
