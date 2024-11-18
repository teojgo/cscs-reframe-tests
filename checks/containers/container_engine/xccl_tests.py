# Copyright 2024 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause

import pathlib
import sys

import reframe as rfm
import reframe.utility.sanity as sn

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent / 'mixins'))

from container_engine import ContainerEngineMixin  # noqa: E402


class XCCLTestBase(rfm.RunOnlyRegressionTest, ContainerEngineMixin):
    valid_prog_environs = ['builtin']
    sourcesdir = None
    test_name = parameter(['all_reduce', 'sendrecv'])
    num_nodes = variable(int, value=2)
    min_bytes = variable(str, value='1024M')
    max_bytes = variable(str, value='1024M')
    postrun_cmds = ['cat *_smi.out > smi_concat.out']
    container_env_table = {
        'annotations.com.hooks': {
            'cxi.enabled': 'true',
            #'aws_ofi_nccl.enabled': 'true'
        }
    }
    container_mounts = [
        '/capstor/scratch/cscs/manitart/nccl_plugin_todi/1.9.2-aws/libnccl-net.so:/usr/lib/libnccl-net-ofi.so',
        "/usr/lib64/libhwloc.so.15:/usr/lib/libhwloc.so.15",
        "/usr/lib64/libpciaccess.so.0:/usr/lib/libpciaccess.so.0",
        "/usr/lib64/libxml2.so.2:/usr/lib/libxml2.so.2"
    ]

    env_vars = {
        'NCCL_DEBUG': 'Info',
        'FI_CXI_DISABLE_HOST_REGISTER': '1',
        'FI_MR_CACHE_MONITOR': 'userfaultfd',
        'NCCL_NET_PLUGIN' : 'ofi',
        #'NCCL_DEBUG_SUBSYS': 'ALL',
        'FI_CXI_COMPAT': '0',
    }
    tags = {'production', 'ce'}

    @run_after('setup')
    def set_executable(self):
        self.executable = 'bash'

    @run_after('setup')
    def set_num_gpus_per_node(self):
        curr_part = self.current_partition
        self.num_gpus_per_node = curr_part.select_devices('gpu')[0].num_devices
        self.num_tasks_per_node = self.num_gpus_per_node
        self.num_tasks = self.num_nodes * self.num_gpus_per_node

    @run_after('setup')
    def set_executable_opts(self):
        self.executable_opts = [
            '-c',
            "'{ [[ $SLURM_LOCALID -eq 0 ]] && nvidia-smi > $(hostname)_smi.out; }; ",
            f'{self.test_name}_perf',
            f'-b {self.min_bytes}', f'-e {self.max_bytes}', f"-g 1'"
        ]

    @run_before('run')
    def set_pmi2(self):
        #self.job.launcher.options += ['--mpi=pmix']
        self.job.launcher.options += ['--mpi=pmi2', '-u']

    @sanity_function
    def assert_sanity(self):
        return sn.all([
            sn.assert_found(r'Out of bounds values\s*:\s*0\s*OK', self.stdout),
            sn.assert_found(
                r'NCCL INFO NET/OFI Selected Provider is cxi', self.stdout
            )
         ])

    @run_before('performance')
    def set_reference(self):
        self.reference = self.reference_per_test[self.test_name]

    @run_before('performance')
    def set_perf(self):
        self.perf_patterns = {
            'GB/s': sn.extractsingle(
                r'Avg bus bandwidth\s*:\s*(?P<gbs>\S+)',
                self.stdout, 'gbs', float),
            'Max_Free_MiB': sn.max(sn.extractall(
                r'(?P<free_mib>\d+)MiB\s*/\s*\d+MiB',
                'smi_concat.out', 'free_mib', float))
        }



@rfm.simple_test
class NCCLTestsCE(XCCLTestBase):
    valid_systems = ['+ce +nvgpu']
    image_tag = parameter(['cuda12.3'])

    # Disable Nvidia Driver requirement
    env_vars['NVIDIA_DISABLE_REQUIRE'] = 1

    reference_per_test = {
        'sendrecv': {
            '*': {
                'GB/s': (24.0, -0.05, None, 'GB/s')
            }
         },
        'all_reduce': {
            '*': {
                'GB/s': (75.0, -0.05, None, 'GB/s')
            }
        }
    }

    @run_after('init')
    def setup_ce(self):
        cuda_major = self.image_tag.split('.')[0]

        #self.container_image = '/capstor/scratch/cscs/manitart/ce_images/nccl-tests@torch_24.01.sqsh'
        self.container_image = (f'jfrog.svc.cscs.ch#reframe-oci/nccl-tests:'
                                f'{self.image_tag}')
        #self.container_env_table['annotations.com.hooks'].update({
        #    'aws_ofi_nccl.variant': cuda_major
        #})

@rfm.simple_test
class RCCLTestCE(XCCLTestBase):
    valid_systems = ['+ce +amdgpu']
    image_tag = parameter(['rocm57', 'rocm60'])
    reference_per_test = {
        'sendrecv': {
            '*': {
                'GB/s': (7.4, -0.05, None, 'GB/s')
            }
         },
        'all_reduce': {
            '*': {
                'GB/s': (105.0, -0.05, None, 'GB/s')
            }
        }
    }

    @run_after('init')
    def setup_ce(self):
        rocm_major = self.image_tag[:-1]
        self.container_image = (f'jfrog.svc.cscs.ch#reframe-oci/rccl-tests:'
                                f'{self.image_tag}')
        self.container_env_table['annotations.com.hooks'].update({
            'aws_ofi_nccl.variant': rocm_major
        })
