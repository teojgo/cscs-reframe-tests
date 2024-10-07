# Copyright 2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import itertools

import reframe as rfm
import reframe.utility.sanity as sn



@rfm.simple_test
class CXIGPULoopbackBW(rfm.RunOnlyRegressionTest):
    valid_systems = ['+nvgpu']
    valid_prog_environs = ['builtin']
    cxi_device_count = 4
    sourcesdir = None
    num_tasks_per_node = 1
    executable = 'bash'
    executable_opts = [
        f"-c",
        f"'for (( j=0; j<{cxi_device_count}; j++ )); "
        f"do cxi_gpu_loopback_bw -d cxi$j; done'"
    ]
    reference = {
        '*': {
            'min_gpu_bw': (187.0, -0.02, None, 'Gb/s')
        }
    }

    @sanity_function
    def assert_cxi_count(self):
        cxi_count = sn.count(
            sn.extractall(rf'CXI Loopback Bandwidth Test', self.stdout))
        return sn.assert_eq(4, cxi_count)

    @performance_function('Gb/s')
    def min_gpu_bw(self):
        regex = r'^\s+\d+(\s+\d+)\s+(?P<bw>\S+)\s+\S+\s*'
        return sn.min(sn.extractall(regex, self.stdout, 'bw', float))


@rfm.simple_test
class CXIGPU2GPULoopbackBW(rfm.RunOnlyRegressionTest):
    valid_systems = ['+nvgpu']
    valid_prog_environs = ['+cuda']
    sourcesdir = None
    num_tasks_per_node = 1
    executable = 'bash'
    executable_opts = ['-c']
    modules = ['cudatoolkit']
    reference = {
        '*': {
            'min_gpu_bw': (187.0, -0.02, None, 'Gb/s')
        }
    }

    @run_after('setup')
    def set_executable_opts(self):
        curr_part = self.current_partition
        gpu_count = curr_part.select_devices('gpu')[0].num_devices
        self.skip_if(gpu_count <= 1,
                     'test supported only for multi-gpu systems')
        self.gpu_combinations = list(
            itertools.combinations(range(gpu_count), 2)
        )
        loopback_cmds = ' '.join([
            f'cxi_gpu_loopback_bw -g Nvidia -t {gpu1} -r {gpu2};'
            for gpu1, gpu2 in self.gpu_combinations]
        )
        self.executable_opts += [f"'{loopback_cmds}'"]

    @sanity_function
    def assert_run_count(self):
        test_header_count = sn.count(
            sn.extractall(r'CXI Loopback Bandwidth Test', self.stdout))
        test_result_count = sn.count(
            sn.extractall(r'^\s+\d+(\s+\d+)\s+\S+\s+\S+\s*',
                          self.stdout))
        return sn.all([
            sn.assert_eq(len(self.gpu_combinations), test_header_count),
            sn.assert_eq(len(self.gpu_combinations), test_result_count)
        ])

    @performance_function('Gb/s')
    def min_gpu_bw(self):
        regex = r'^\s+\d+(\s+\d+)\s+(?P<bw>\S+)\s+\S+\s*'
        return sn.min(sn.extractall(regex, self.stdout, 'bw', float))
