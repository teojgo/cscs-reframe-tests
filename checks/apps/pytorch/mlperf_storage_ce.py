# Copyright 2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import getpass
import pathlib
import sys
import tempfile

import reframe as rfm
import reframe.utility.sanity as sn

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent / 'mixins'))
from container_engine import ContainerEngineMixin  # noqa: E402


class mlperf_storage_datagen_ce(rfm.RunOnlyRegressionTest,
                                ContainerEngineMixin):
    # from https://github.com/henrique/storage/blob/v1.0-cscs/Dockerfile
    container_image = ('jfrog.svc.cscs.ch#reframe-oci/mlperf-storage:'
                       'v1.0-mpi_4.2.1')
    container_workdir = None
    valid_systems = ['+nvgpu +ce']
    valid_prog_environs = ['builtin']
    ref_values = {
        '/capstor/scratch/cscs/': 93000,
        '/iopsstor/scratch/cscs/': 144000,
    }
    base_dir = parameter(list(ref_values.keys()))
    num_nodes = variable(int, value=32)
    time_limit = '15m'
    accelerator_type = 'h100'
    workload = variable(str, value='unet3d')
    env_vars = {
        'LC_ALL': 'C',
        'HYDRA_FULL_ERROR': '1',
        'RDMAV_FORK_SAFE': '1',
    }

    @run_after('setup')
    def setup_test(self):
        self.num_cpus_per_node = self.current_partition.processor.num_cores
        self.num_cpus_per_task = 6
        self.num_tasks_per_node = (self.num_cpus_per_node //
                                   self.num_cpus_per_task)
        self.num_tasks = self.num_nodes * self.num_tasks_per_node
        self.num_files = 64 * self.num_tasks
        self.storage = tempfile.mkdtemp(
            prefix='rfm_mlperf_storage_',
            dir=pathlib.Path(self.base_dir) / getpass.getuser()
        )

        self.container_mounts = [
            f'{self.stagedir}/unet3d.yaml:'
            '/workspace/storage/storage-conf/workload/unet3d_h100.yaml',
            f'{self.storage}:/mlperf_storage',
        ]

        self.executable = rf""" bash -c '
            ./benchmark.sh datagen --workload {self.workload} \
                --accelerator-type {self.accelerator_type} \
                --num-parallel {self.num_tasks} \
                --results-dir /mlperf_storage/resultsdir \
                --param dataset.num_files_train={self.num_files} \
                --param dataset.data_folder=/mlperf_storage/dataset \
                --param checkpoint.checkpoint_folder=/mlperf_storage/checkpoint
        ' """

    @run_before('run')
    def set_pmi2(self):
        self.job.launcher.options += ['--mpi=pmi2']

    @sanity_function
    def assert_data_generation_done(self):
        return sn.assert_found(r'.*Generation done.*', self.stderr)


class MLperfStorageCE(rfm.RunOnlyRegressionTest, ContainerEngineMixin):
    container_image = ('jfrog.svc.cscs.ch#reframe-oci/mlperf-storage:'
                       'v1.0-mpi_4.2.1')
    valid_systems = ['+nvgpu +ce']
    valid_prog_environs = ['builtin']
    tags = {'production', 'ce', 'maintenance'}
    mlperf_data = fixture(mlperf_storage_datagen_ce, scope='environment')

    # Add here to supress the warning, set by the fixture
    num_nodes = variable(int)

    @run_after('setup')
    def setup_test(self):
        self.num_nodes = self.mlperf_data.num_nodes
        self.num_tasks_per_node = self.mlperf_data.num_tasks_per_node
        self.num_cpus_per_task = self.mlperf_data.num_cpus_per_task
        self.num_tasks = self.mlperf_data.num_tasks
        self.env_vars = self.mlperf_data.env_vars
        self.workload = self.mlperf_data.workload
        self.container_workdir = self.mlperf_data.container_workdir
        num_files = self.mlperf_data.num_files
        accelerator_type = self.mlperf_data.accelerator_type
        self.container_mounts = [
            f'{self.stagedir}/unet3d.yaml:'
            '/workspace/storage/storage-conf/workload/unet3d_h100.yaml',
            f'{self.mlperf_data.storage}:/mlperf_storage',
        ]

        self.executable = rf""" bash -c '
            ./benchmark.sh run --workload {self.workload} \
                --accelerator-type {accelerator_type} \
                --num-accelerators {self.num_tasks} \
                --results-dir /mlperf_storage/resultsdir \
                --param dataset.num_files_train={num_files} \
                --param dataset.data_folder=/mlperf_storage/dataset \
                --param checkpoint.checkpoint_folder=/mlperf_storage/checkpoint
        ' """

        ref_value = self.mlperf_data.ref_values[self.mlperf_data.base_dir]
        self.reference = {
            '*': {
                'mb_per_sec_total': (ref_value, -0.2, None, 'MB/second'),
            }
        }

    @run_before('run')
    def set_pmi2(self):
        self.job.launcher.options += ['--mpi=pmi2']

    @sanity_function
    def assert_job_is_complete(self):
        return sn.assert_found(r'.*train_au_meet_expectation.*', self.stderr)

    @performance_function('MB/second')
    def mb_per_sec_total(self):
        return sn.avg(sn.extractall(
            r'Training I/O Throughput \(MB/second\): (?P<mbs_total>\S+)',
            self.stderr, 'mbs_total', float
        ))


@rfm.simple_test
class MLperfStorageCECleanup(rfm.RunOnlyRegressionTest):
    valid_systems = ['+nvgpu +ce']
    valid_prog_environs = ['builtin']
    num_tasks = 1
    executable = 'rm'
    mlperf = fixture(MLperfStorageCE, scope='environment')
    tags = {'production', 'ce', 'maintenance'}

    @run_after('setup')
    def setup_exec_opts(self):
        self.storage = self.mlperf.mlperf_data.storage
        self.executable_opts = ['-r', '-f', self.storage]

    @sanity_function
    def assert_successful_cleanup(self):
        return not sn.path_exists(self.storage)
