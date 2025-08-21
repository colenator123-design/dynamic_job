from collections import OrderedDict
from typing import Dict, List, Optional


class Operation:
    # 表示「一個工作(Job)中的一道工序(Operation)」的資料結構
    # 會保存：所屬 Job、工序編號、可用機器及加工時間、前置工序，以及排程結果等
    def __init__(self, job, job_id, operation_id):
        self._job = job                      # 所屬的 Job 物件（可從這裡取得整個 Job 的資訊）
        self._job_id = job_id                # 此工序所屬的 Job 編號（純識別用的 ID）
        self._operation_id = operation_id    # 此工序在 Job 內的工序編號/序位
        self._processing_times = OrderedDict()  # 機器ID → 該機器的加工時間（FJSP 會用到）
        self._predecessors: List = []        # 前置工序列表（需先完成的其他 Operation）
        self._scheduling_information = {}    # 排程結果（開始/結束時間、機器、設定時間等）

    def __repr__(self):
        # 便利的字串表示法：用於除錯或列印此物件時顯示 job_id 與 operation_id
        return (
            f"<Operation(job_id={self._job_id}, operation_id={self._operation_id})>"
        )

    def reset(self):
        # 重置排程資訊（將排程結果清空，便於重排）
        self._scheduling_information = {}

    @property
    def job(self):
        """Return the job object of the operation."""
        # 回傳此工序所屬的 Job 物件
        return self._job

    @property
    def job_id(self) -> int:
        """Return the job's id of the operation."""
        # 回傳此工序所屬的 Job 編號（整數）
        return self._job_id

    @property
    def operation_id(self) -> int:
        """Return the operation's id."""
        # 回傳此工序在 Job 內的工序編號（整數）
        return self._operation_id

    @property
    def scheduling_information(self) -> Dict:
        """Return the scheduling information of the operation."""
        # 回傳完整的排程資訊字典（若未排程則可能為空字典）
        return self._scheduling_information

    @property
    def processing_times(self) -> dict:
        """Return a dictionary of machine ids and processing time durations."""
        # 回傳可加工此工序的機器與其加工時間映射（machine_id → duration）
        return self._processing_times

    @property
    def scheduled_start_time(self) -> Optional[int]:
        """Return the scheduled start time of the operation."""
        # 回傳排定的開始時間；未排程時為 None
        return self._scheduling_information.get('start_time', None)

    @property
    def scheduled_end_time(self) -> Optional[int]:
        """Return the scheduled end time of the operation."""
        # 回傳排定的結束時間；未排程時為 None
        return self._scheduling_information.get('end_time', None)

    @property
    def scheduled_duration(self) -> Optional[int]:
        """Return the scheduled duration of the operation."""
        # 回傳排定的加工時間（processing_time）；未排程時為 None
        return self._scheduling_information.get('processing_time', None)

    @property
    def scheduled_machine(self) -> Optional[int]:
        """Return the machine id that the operation is scheduled on."""
        # 回傳此工序被分配的機器ID；未分配時為 None
        return self._scheduling_information.get('machine_id', None)

    @property
    def predecessors(self) -> List:
        """Return the list of predecessor operations."""
        # 回傳前置工序列表（需先完成的工序）
        return self._predecessors

    @property
    def optional_machines_id(self) -> List:
        """Returns the list of machine ids that are eligible for processing this operation."""
        # 回傳所有可加工此工序的機器ID列表（由 processing_times 的 key 組成）
        return list(self._processing_times.keys())

    @property
    def finishing_time_predecessors(self) -> int:
        """Return the finishing time of the latest predecessor."""
        # 回傳所有前置工序中「最晚」的結束時間；若沒有前置工序則回傳 0
        if not self.predecessors:
            return 0
        end_times_predecessors = [operation.scheduled_end_time for operation in self.predecessors]
        return max(end_times_predecessors)

    def update_job_id(self, new_job_id: int) -> None:
        """Update the id of a job (used for assembly scheduling problems, with no pre-given job id)."""
        # 更新此工序所屬的 Job 編號（某些情境，如組裝排程，Job ID 可能後設）
        self._job_id = new_job_id

    def update_job(self, job) -> None:
        """Update job information (edge case for FAJSP)."""
        # 更新所屬的 Job 物件（例如在彈性車間 FAJSP 的特殊情況）
        self._job = job

    def add_predecessors(self, predecessors: List) -> None:
        """Add a list of predecessor operations to the current operation."""
        # 批量新增前置工序（將傳入列表的元素加入到 _predecessors 末尾）
        self.predecessors.extend(predecessors)

    def add_operation_option(self, machine_id, duration) -> None:
        """Add an machine option to the current operation."""
        # 新增一個可用機器的加工選項：指定 machine_id 與其加工時間 duration
        self._processing_times[machine_id] = duration

    def update_scheduled_sequence_dependent_setup_times(self, start_time_setup, setup_duration):
        """Update the sequence dependent setup times of this operation (used for backfilling logic)."""
        # 更新「序列依賴設定時間」(SDST)：記錄設定開始/結束時間與設定時長
        self._scheduling_information['start_setup'] = start_time_setup
        self._scheduling_information['end_setup'] = start_time_setup + setup_duration
        self._scheduling_information['setup_time'] = setup_duration

    def add_operation_scheduling_information(self, machine_id: int, start_time: int, setup_time: int, duration) -> None:
        """Add scheduling information to the current operation."""
        # 一次性寫入完整的排程結果：
        # - 機器ID
        # - 加工開始/結束時間（end_time = start_time + duration）
        # - 加工時間 processing_time
        # - 設定時間區段（start_setup = start_time - setup_time, end_setup = start_time）
        # 注意：此方法會覆蓋整個 _scheduling_information 字典
        self._scheduling_information = {
            'machine_id': machine_id,
            'start_time': start_time,
            'end_time': start_time + duration,
            'processing_time': duration,
            'start_setup': start_time - setup_time,
            'end_setup': start_time,
            'setup_time': setup_time}
