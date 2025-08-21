
class Operation:
    def __init__(self, job_id, op_id, candidate_machines=None):
        """
        初始化一個工序 (Operation) 物件。

        參數:
            job_id (int): 此工序所屬的工件 ID。
            op_id (int): 此工序在工件內的順序 ID。
            candidate_machines (dict, optional): 可加工此工序的候選機器及其加工時間。
                                               格式為 {machine_id: processing_time}。
                                               預設為 None。
        """
        self.job_id = job_id
        self.op_id = op_id
        self.candidate_machines = candidate_machines if candidate_machines is not None else {}
        self.assigned_machine = None
        self.start_time = None
        self.end_time = None

    def __repr__(self):
        return f"Operation(Job={self.job_id}, ID={self.op_id}, Machine={self.assigned_machine})"

    def set_sequencing_info(self, machine_id, start_time, end_time):
        """
        設定工序的排程資訊。
        """
        if machine_id not in self.candidate_machines:
            raise ValueError(f"機器 {machine_id} 不是此工序的候選機器。")
        self.assigned_machine = machine_id
        self.start_time = start_time
        self.end_time = end_time
