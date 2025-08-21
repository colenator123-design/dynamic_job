
class Job:
    def __init__(self, job_id, operations=None, due_date=0, tardiness_penalty=0):
        """
        初始化一個工件 (Job) 物件。

        參數:
            job_id (int): 工件的唯一標識符。
            operations (list, optional): 組成此工件的工序 (Operation) 列表。預設為 None。
            due_date (int, optional): 工件的交貨日期。預設為 0。
            tardiness_penalty (float, optional): 工件的延遲懲罰權重。預設為 0。
        """
        self.job_id = job_id
        self.operations = operations if operations is not None else []
        self.due_date = due_date
        self.tardiness_penalty = tardiness_penalty
        self.completion_time = 0

    def __repr__(self):
        return f"Job(ID={self.job_id}, Operations={len(self.operations)}, DueDate={self.due_date})"

    def add_operation(self, operation):
        """
        向工件中添加一個工序。
        """
        self.operations.append(operation)

    def get_completion_time(self):
        """
        計算並返回工件的完工時間 (所有工序中最晚的結束時間)。
        """
        if not self.operations:
            return 0
        return max(op.end_time for op in self.operations if op.end_time is not None)
