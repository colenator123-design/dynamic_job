import pandas as pd

class DataLoader:
    def __init__(self):
        """
        初始化數據載入器。
        直接從論文附錄硬編碼載入數據。
        """
        pass

    def load_maintenance_data(self):
        """
        從 Table A1 載入機器的維護參數。
        返回一個字典，key 為 machine_id (從 0 開始)。
        """
        maintenance_data = {
            0: {'theta': 2.5, 'eta': 215, 'pm_duration': 8, 'pm_cost': 480, 'mr_duration': 20, 'mr_cost': 1010},
            1: {'theta': 3.3, 'eta': 150, 'pm_duration': 10, 'pm_cost': 550, 'mr_duration': 24, 'mr_cost': 1300},
            2: {'theta': 3.3, 'eta': 200, 'pm_duration': 6, 'pm_cost': 460, 'mr_duration': 16, 'mr_cost': 1200},
            3: {'theta': 2.6, 'eta': 320, 'pm_duration': 13, 'pm_cost': 470, 'mr_duration': 18, 'mr_cost': 1150},
            4: {'theta': 2.5, 'eta': 240, 'pm_duration': 5, 'pm_cost': 450, 'mr_duration': 18, 'mr_cost': 1220},
            5: {'theta': 2.2, 'eta': 220, 'pm_duration': 9, 'pm_cost': 500, 'mr_duration': 22, 'mr_cost': 1240},
            6: {'theta': 3.2, 'eta': 320, 'pm_duration': 7, 'pm_cost': 570, 'mr_duration': 28, 'mr_cost': 1190},
            7: {'theta': 2.8, 'eta': 210, 'pm_duration': 12, 'pm_cost': 520, 'mr_duration': 22, 'mr_cost': 1230},
            8: {'theta': 2.4, 'eta': 200, 'pm_duration': 11, 'pm_cost': 540, 'mr_duration': 28, 'mr_cost': 1140},
            9: {'theta': 2.2, 'eta': 250, 'pm_duration': 15, 'pm_cost': 500, 'mr_duration': 28, 'mr_cost': 1320},
            10: {'theta': 2.6, 'eta': 320, 'pm_duration': 6, 'pm_cost': 510, 'mr_duration': 21, 'mr_cost': 1260},
            11: {'theta': 2.6, 'eta': 160, 'pm_duration': 10, 'pm_cost': 560, 'mr_duration': 25, 'mr_cost': 1050},
            12: {'theta': 2.4, 'eta': 290, 'pm_duration': 12, 'pm_cost': 480, 'mr_duration': 26, 'mr_cost': 1170},
            13: {'theta': 2.7, 'eta': 360, 'pm_duration': 16, 'pm_cost': 490, 'mr_duration': 29, 'mr_cost': 1080},
            14: {'theta': 2.2, 'eta': 220, 'pm_duration': 18, 'pm_cost': 510, 'mr_duration': 30, 'mr_cost': 1130},
        }
        return maintenance_data

    def load_production_data(self):
        """
        從 Table A2 載入工件的生產數據。
        返回 jobs_data (字典) 和 beta_vectors (字典)。
        Job, Operation, Machine ID 均從 0 開始。
        """
        jobs_data = {
            0: {'operations': [
                {'op_id': 0, 'candidate_machines': {0: 15, 1: 17, 2: 19, 3: 22}},
                {'op_id': 1, 'candidate_machines': {0: 10, 1: 8, 3: 9}},
                {'op_id': 2, 'candidate_machines': {4: 12, 5: 17, 6: 15}},
                {'op_id': 3, 'candidate_machines': {4: 8, 5: 12, 6: 14}},
                {'op_id': 4, 'candidate_machines': {7: 23, 8: 19}},
                {'op_id': 5, 'candidate_machines': {7: 11, 8: 15}},
                {'op_id': 6, 'candidate_machines': {7: 22, 8: 16, 10: 18}},
                {'op_id': 7, 'candidate_machines': {1: 13, 2: 12, 4: 17}},
                {'op_id': 8, 'candidate_machines': {3: 16, 6: 12}},
                {'op_id': 9, 'candidate_machines': {9: 9, 12: 12}},
                {'op_id': 10, 'candidate_machines': {10: 19, 11: 18, 14: 21}},
                {'op_id': 11, 'candidate_machines': {9: 15, 11: 14, 13: 17}},
                {'op_id': 12, 'candidate_machines': {12: 15, 13: 12, 14: 15}}
            ]},
            1: {'operations': [
                {'op_id': 0, 'candidate_machines': {0: 18, 2: 16, 3: 20}},
                {'op_id': 1, 'candidate_machines': {1: 26, 4: 24}},
                {'op_id': 2, 'candidate_machines': {2: 16, 3: 13, 5: 20}},
                {'op_id': 3, 'candidate_machines': {4: 9, 6: 10, 8: 8}},
                {'op_id': 4, 'candidate_machines': {0: 19, 5: 22}},
                {'op_id': 5, 'candidate_machines': {1: 15, 7: 13}},
                {'op_id': 6, 'candidate_machines': {7: 14, 8: 16}},
                {'op_id': 7, 'candidate_machines': {10: 18, 11: 15, 14: 22}},
                {'op_id': 8, 'candidate_machines': {9: 11, 12: 13}},
                {'op_id': 9, 'candidate_machines': {12: 15, 13: 12}},
                {'op_id': 10, 'candidate_machines': {13: 14, 14: 13}}
            ]},
            2: {'operations': [
                {'op_id': 0, 'candidate_machines': {0: 25, 1: 28, 3: 22}},
                {'op_id': 1, 'candidate_machines': {2: 14, 4: 17}},
                {'op_id': 2, 'candidate_machines': {5: 18, 6: 15, 7: 16}},
                {'op_id': 3, 'candidate_machines': {7: 20, 8: 19}},
                {'op_id': 4, 'candidate_machines': {10: 10, 12: 15, 13: 13}},
                {'op_id': 5, 'candidate_machines': {9: 15, 11: 18}},
                {'op_id': 6, 'candidate_machines': {13: 19, 14: 17}},
                {'op_id': 7, 'candidate_machines': {2: 13, 4: 12}}
            ]},
            3: {'operations': [
                {'op_id': 0, 'candidate_machines': {0: 12, 1: 10, 3: 13}},
                {'op_id': 1, 'candidate_machines': {1: 14, 2: 11}},
                {'op_id': 2, 'candidate_machines': {4: 8, 5: 9}},
                {'op_id': 3, 'candidate_machines': {7: 9, 8: 7}},
                {'op_id': 4, 'candidate_machines': {7: 10, 8: 9}},
                {'op_id': 5, 'candidate_machines': {6: 15, 9: 13, 10: 11}},
                {'op_id': 6, 'candidate_machines': {11: 11, 13: 12}},
                {'op_id': 7, 'candidate_machines': {4: 12, 6: 13}},
                {'op_id': 8, 'candidate_machines': {11: 9, 12: 10}},
                {'op_id': 9, 'candidate_machines': {12: 7, 14: 8}}
            ]},
            4: {'operations': [
                {'op_id': 0, 'candidate_machines': {1: 14, 2: 13, 3: 12}},
                {'op_id': 1, 'candidate_machines': {0: 10, 4: 12}},
                {'op_id': 2, 'candidate_machines': {2: 12, 4: 10}},
                {'op_id': 3, 'candidate_machines': {5: 9, 6: 12}},
                {'op_id': 4, 'candidate_machines': {7: 13, 9: 11}},
                {'op_id': 5, 'candidate_machines': {8: 15, 11: 14}},
                {'op_id': 6, 'candidate_machines': {6: 14, 7: 13, 10: 11}},
                {'op_id': 7, 'candidate_machines': {11: 8, 14: 10}},
                {'op_id': 8, 'candidate_machines': {12: 18, 13: 15}}
            ]},
            5: {'operations': [
                {'op_id': 0, 'candidate_machines': {0: 16, 1: 14, 3: 12}},
                {'op_id': 1, 'candidate_machines': {2: 20, 4: 18}},
                {'op_id': 2, 'candidate_machines': {3: 15, 5: 17, 6: 14}},
                {'op_id': 3, 'candidate_machines': {7: 12, 8: 13}},
                {'op_id': 4, 'candidate_machines': {9: 13, 10: 10, 11: 14}},
                {'op_id': 5, 'candidate_machines': {12: 17, 14: 19}},
                {'op_id': 6, 'candidate_machines': {13: 15, 14: 12}}
            ]}
        }

        beta_vectors = {
            0: [1.095, 0.479, 0.134, 0.754],
            1: [1.124, 0.658, 0.278, 1.006],
            2: [0.875, 1.232, 0.433, 0.986],
            3: [0.878, 0.956, 0.354, 1.432],
            4: [1.128, 0.452, 0.126, 0.364],
            5: [0.768, 0.982, 0.453, 1.057]
        }
        return jobs_data, beta_vectors