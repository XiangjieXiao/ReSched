import random
import time
import os
import re

import numpy as np


class CaseGenerator:
    """
        the generator of SD1 data (from "https://github.com/songwenas12/fjsp-drl"),
        used for generating training instances

        Remark: the validation and testing intances of SD1 data are
    """

    def __init__(self, num_jobs, num_mas, nums_ope=None, path='./test', flag_same_opes=True, flag_doc=False):
        # n_i
        self.str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
        if nums_ope is None:
            nums_ope = []
        self.flag_doc = flag_doc  # Whether save the instance to a file
        self.flag_same_opes = flag_same_opes
        self.nums_ope = nums_ope
        self.path = path  # Instance save path (relative path)
        self.num_jobs = num_jobs
        self.num_machines = num_mas

        self.mas_per_ope_min = 1  # The minimum number of machines that can process an operation
        self.mas_per_ope_max = num_mas

        self.opes_per_job_min = int(num_mas * 0.8)  # The minimum number of operations for a job
        self.opes_per_job_max = int(num_mas * 1.2)

        self.proctime_per_ope_min = 1  # Minimum average processing time
        self.max_processing_time = 20

        self.proctime_dev = 0.2

        self.seed = None

    def generate_instances(self, instance_num: int):
        """
            Generate multiple instances
        :param instance_num: the number of instances
        """
        job_idx = []
        operation_idx = []
        durations = []
        dependencies_on = []
        dependencies_oj = []

        # set the operation number for each job in this batch
        self.nums_ope = [random.randint(self.opes_per_job_min, self.opes_per_job_max) for _ in range(self.num_jobs)]

        job_id = [job for job, num in enumerate(self.nums_ope) for _ in range(num)]
        operation_id = [op for num in self.nums_ope for op in range(num)]

        # Pre-compute dependencies and connections (same for all instances in the batch)
        num_jo = len(operation_id)
        # Operation-level dependencies
        dependency_forward_o = np.zeros((num_jo, num_jo), dtype=bool)
        # Job-level dependencies (i.e., current operation to later operations of the same job)
        dependency_forward_j = np.zeros((num_jo, num_jo), dtype=bool)

        for idx in range(num_jo):
            op_id = operation_id[idx]
            if op_id == 0:
                job_start = idx
                job_end = job_start + 1
                while job_end < num_jo and operation_id[job_end] != 0:
                    job_end += 1
            else:
                dep_from = idx - 1
                dep_to = idx

                # Operation dependencies
                dependency_forward_o[dep_from, dep_to] = True
                # Job dependencies
                op_range = job_end - dep_to
                dependency_forward_j[dep_from, dep_to:dep_to + op_range] = True

        dependency = dependency_forward_o
        connection = dependency_forward_j

        for i in range(instance_num):
            # get duration
            _, OpPT, _ = self.get_case(i)
            job_idx.append(np.array(job_id))
            operation_idx.append(np.array(operation_id))
            durations.append(OpPT)

            # get dependencies
            dependencies_on.append(dependency)
            dependencies_oj.append(connection)

        job_idx = np.array(job_idx)
        operation_idx = np.array(operation_idx)
        durations = np.array(durations)
        dependencies_on = np.array(dependencies_on)
        dependencies_oj = np.array(dependencies_oj)

        return job_idx, operation_idx, durations, dependencies_on, dependencies_oj

    def get_case(self, idx=0):
        """
        Generate FJSP instance
        :param idx: The instance number
        """
        if not self.flag_same_opes:
            self.nums_ope = [random.randint(self.opes_per_job_min, self.opes_per_job_max) for _ in range(self.num_jobs)]
        self.num_opes = sum(self.nums_ope)
        self.nums_option = [random.randint(self.mas_per_ope_min, self.mas_per_ope_max) for _ in range(self.num_opes)]
        self.num_options = sum(self.nums_option)

        self.ope_ma = []
        for val in self.nums_option:
            self.ope_ma = self.ope_ma + sorted(random.sample(range(1, self.num_machines + 1), val))
        self.proc_time = []

        self.proc_times_mean = [random.randint(self.proctime_per_ope_min, self.max_processing_time) for _ in
                                range(self.num_opes)]
        for i in range(len(self.nums_option)):
            low_bound = max(self.proctime_per_ope_min, round(self.proc_times_mean[i] * (1 - self.proctime_dev)))
            high_bound = min(self.max_processing_time, round(self.proc_times_mean[i] * (1 + self.proctime_dev)))
            proc_time_ope = [random.randint(low_bound, high_bound) for _ in range(self.nums_option[i])]
            self.proc_time = self.proc_time + proc_time_ope

        self.num_ope_biass = [sum(self.nums_ope[0:i]) for i in range(self.num_jobs)]
        self.num_ma_biass = [sum(self.nums_option[0:i]) for i in range(self.num_opes)]
        line0 = '{0}\t{1}\t{2}\n'.format(self.num_jobs, self.num_machines, self.num_options / self.num_opes)
        lines_doc = []
        lines_doc.append('{0}\t{1}\t{2}'.format(self.num_jobs, self.num_machines, self.num_options / self.num_opes))
        for i in range(self.num_jobs):
            flag = 0
            flag_time = 0
            flag_new_ope = 1
            idx_ope = -1
            idx_ma = 0
            line = []
            option_max = sum(self.nums_option[self.num_ope_biass[i]:(self.num_ope_biass[i] + self.nums_ope[i])])
            idx_option = 0
            while True:
                if flag == 0:
                    line.append(self.nums_ope[i])
                    flag += 1
                elif flag == flag_new_ope:
                    idx_ope += 1
                    idx_ma = 0
                    flag_new_ope += self.nums_option[self.num_ope_biass[i] + idx_ope] * 2 + 1
                    line.append(self.nums_option[self.num_ope_biass[i] + idx_ope])
                    flag += 1
                elif flag_time == 0:
                    line.append(self.ope_ma[self.num_ma_biass[self.num_ope_biass[i] + idx_ope] + idx_ma])
                    flag += 1
                    flag_time = 1
                else:
                    line.append(self.proc_time[self.num_ma_biass[self.num_ope_biass[i] + idx_ope] + idx_ma])
                    flag += 1
                    flag_time = 0
                    idx_option += 1
                    idx_ma += 1
                if idx_option == option_max:
                    str_line = " ".join([str(val) for val in line])
                    lines_doc.append(str_line)
                    break
        job_length, op_pt = text_to_matrix(lines_doc)
        if self.flag_doc:
            if not os.path.exists(self.path):
                os.makedirs(self.path)
            # doc = open(
            #     self.path + '/' + '{0}x{1}_{2}.fjs'.format(self.num_jobs, self.num_machines, str.zfill(str(idx + 1), 3)),
            #     'w')
            doc = open(self.path + f'/{self.str_time}.txt', 'a')
            # doc = open(self.path + f'/ours.txt', 'a')
            for i in range(len(lines_doc)):
                print(lines_doc[i], file=doc)
            doc.close()

        return job_length, op_pt, self.num_options / self.num_opes

    def set_seed(self, seed):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)


def text_to_matrix(text):
    """
            Convert text form of the data into matrix form
    :param text: the standard text form of the instance
    :return:  the matrix form of the instance
            job_length: the number of operations in each job (shape [J])
            op_pt: the processing time matrix with shape [N, M],
                where op_pt[i,j] is the processing time of the ith operation
                on the jth machine or 0 if $O_i$ can not process on $M_j$
    """
    n_j = int(re.findall(r'\d+\.?\d*', text[0])[0])
    n_m = int(re.findall(r'\d+\.?\d*', text[0])[1])

    job_length = np.zeros(n_j, dtype='int32')
    op_pt = []

    for i in range(n_j):
        content = np.array([int(s) for s in re.findall(r'\d+\.?\d*', text[i + 1])])
        job_length[i] = content[0]

        idx = 1
        for j in range(content[0]):
            op_pt_row = np.zeros(n_m, dtype='int32')
            mch_num = content[idx]
            next_idx = idx + 2 * mch_num + 1
            for k in range(mch_num):
                mch_idx = content[idx + 2 * k + 1]
                pt = content[idx + 2 * k + 2]
                op_pt_row[mch_idx - 1] = pt

            idx = next_idx
            op_pt.append(op_pt_row)

    op_pt = np.array(op_pt)

    return job_length, op_pt


if __name__ == '__main__':
    env = CaseGenerator(10, 5)
    job_idx, operation_idx, durations, dependencies_on, dependencies_oj = env.generate_instances(20)
    a = 1