# Copyright 2016-2022 Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# ReFrame CSCS settings
#

import reframe.utility.osext as osext


site_configuration = {
    'systems': [
        {
            'name': 'ault',
            'descr': 'Ault TDS',
            'hostnames': ['ault'],
            'modules_system': 'lmod',
            'resourcesdir': '/apps/common/UES/reframe/resources',
            'partitions': [
                {
                    'name': 'login',
                    'scheduler': 'local',
                    'time_limit': '10m',
                    'environs': ['gnu'],
                    'descr': 'Login nodes',
                    'max_jobs': 4,
                    'launcher': 'local'
                },
                {
                    'name': 'a64fx',
                    'scheduler': 'slurm',
                    'time_limit': '10m',
                    'access': ['-pa64fx'],
                    'environs': ['gnu'],
                    'descr': 'Fujitsu A64FX CPUs',
                    'max_jobs': 100,
                    'launcher': 'srun',
                    'features': ['remote'],
                },
                {
                    'name': 'amda100',
                    'scheduler': 'slurm',
                    'time_limit': '10m',
                    'access': ['-pamda100'],
                    'environs': ['gnu', 'cuda'],
                    'descr': 'AMD Naples 32c + 4x NVIDIA A100',
                    'max_jobs': 100,
                    'launcher': 'srun',
                    'features': ['gpu', 'nvgpu', 'remote'],
                    'devices': [
                        {
                            'type': 'gpu',
                            'arch': 'sm_80',
                            'num_devices': 4
                        }
                    ]
                },
                {
                    'name': 'amdv100',
                    'scheduler': 'slurm',
                    'time_limit': '10m',
                    'access': ['-pamdv100'],
                    'environs': ['gnu', 'cuda'],
                    'descr': 'AMD Naples 32c + 2x NVIDIA V100',
                    'max_jobs': 100,
                    'launcher': 'srun',
                    'features': ['gpu', 'nvgpu', 'remote'],
                    'devices': [
                        {
                            'type': 'gpu',
                            'arch': 'sm_70',
                            'num_devices': 2
                        }
                    ]
                },
                {
                    'name': 'amdvega',
                    'scheduler': 'slurm',
                    'time_limit': '10m',
                    'access': ['-pamdvega'],
                    'environs': ['gnu', 'rocm'],
                    'descr': 'AMD Naples 32c + 3x AMD GFX900',
                    'max_jobs': 100,
                    'launcher': 'srun',
                    'features': ['gpu', 'amdgpu', 'remote'],
                    'devices': [
                        {
                            'type': 'gpu',
                            'arch': 'gfx900,gfx906',
                            'num_devices': 3
                        }
                    ]
                },
                {
                    'name': 'intelv100',
                    'scheduler': 'slurm',
                    'time_limit': '10m',
                    'access': ['-pintelv100'],
                    'environs': ['gnu', 'cuda'],
                    'descr': 'Intel Skylake 36c + 4x NVIDIA V100',
                    'max_jobs': 100,
                    'launcher': 'srun',
                    'features': ['gpu', 'nvgpu', 'remote'],
                    'devices': [
                        {
                            'type': 'gpu',
                            'arch': 'sm_70',
                            'num_devices': 4
                        }
                    ]
                },
                {
                    'name': 'intel',
                    'scheduler': 'slurm',
                    'time_limit': '10m',
                    'access': ['-pintel'],
                    'environs': ['gnu'],
                    'descr': 'Intel Skylake 36c',
                    'max_jobs': 100,
                    'launcher': 'srun',
                    'features': ['remote'],
                }
            ]
        },
        {
            'name': 'daint',
            'descr': 'Piz Daint',
            'hostnames': ['daint'],
            'modules_system': 'tmod',
            'resourcesdir': '/apps/common/UES/reframe/resources',
            'partitions': [
                {
                    'name': 'login',
                    'scheduler': 'local',
                    'time_limit': '10m',
                    'environs': [
                        'builtin',
                        'PrgEnv-cray',
                        'PrgEnv-gnu',
                        'PrgEnv-intel',
                        'PrgEnv-nvidia'
                    ],
                    'descr': 'Login nodes',
                    'max_jobs': 4,
                    'launcher': 'local'
                },
                {
                    'name': 'gpu',
                    'time_limit': '10m',
                    'scheduler': 'slurm',
                    'container_platforms': [
                        {
                            'type': 'Sarus',
                            'modules': ['sarus']
                        },
                        {
                            'type': 'Singularity',
                            'modules': ['singularity/3.6.4-daint']
                        }
                    ],
                    'modules': ['daint-gpu'],
                    'access': [
                        f'--constraint=gpu',
                        f'--account={osext.osgroup()}'
                    ],
                    'environs': [
                        'builtin',
                        'PrgEnv-cray',
                        'PrgEnv-gnu',
                        'PrgEnv-intel',
                        'PrgEnv-nvidia'
                    ],
                    'descr': 'Hybrid nodes (Haswell/P100)',
                    'max_jobs': 100,
                    'extras': {
                        'cn_memory': 64,
                    },
                    'features': ['gpu', 'nvgpu', 'remote'],
                    'resources': [
                        {
                            'name': 'switches',
                            'options': ['--switches={num_switches}']
                        },
                        {
                            'name': 'gres',
                            'options': ['--gres={gres}']
                        }
                    ],
                    'devices': [
                        {
                            'type': 'gpu',
                            'arch': 'sm_60',
                            'num_devices': 1
                        }
                     ],
                    'launcher': 'srun',
                },
                {
                    'name': 'mc',
                    'scheduler': 'slurm',
                    'time_limit': '10m',
                    'container_platforms': [
                        {
                            'type': 'Sarus',
                            'modules': ['sarus']
                        },
                        {
                            'type': 'Singularity',
                            'modules': ['singularity/3.6.4-daint']
                        }
                    ],
                    'modules': ['daint-mc'],
                    'access': [
                        f'--constraint=mc',
                        f'--account={osext.osgroup()}'
                    ],
                    'environs': [
                        'builtin',
                        'PrgEnv-cray',
                        'PrgEnv-gnu',
                        'PrgEnv-intel',
                        'PrgEnv-nvidia'
                    ],
                    'descr': 'Multicore nodes (Broadwell)',
                    'max_jobs': 100,
                    'extras': {
                        'cn_memory': 64,
                    },
                    'features': ['remote'],
                    'resources': [
                        {
                            'name': 'switches',
                            'options': ['--switches={num_switches}']
                        },
                        {
                            'name': 'gres',
                            'options': ['--gres={gres}']
                        }
                    ],
                    'launcher': 'srun'
                },
                {
                    'name': 'jupyter_gpu',
                    'scheduler': 'slurm',
                    'time_limit': '10m',
                    'environs': ['builtin'],
                    'access': [
                        f'-Cgpu',
                        f'--reservation=interact_gpu',
                        f'--account={osext.osgroup()}'
                    ],
                    'descr': 'JupyterHub GPU nodes',
                    'max_jobs': 10,
                    'launcher': 'srun',
                    'features': ['gpu', 'nvgpu', 'remote'],
                },
                {
                    'name': 'jupyter_mc',
                    'scheduler': 'slurm',
                    'time_limit': '10m',
                    'environs': ['builtin'],
                    'access': [
                        f'-Cmc',
                        f'--reservation=interact_mc',
                        f'--account={osext.osgroup()}'
                    ],
                    'descr': 'JupyterHub multicore nodes',
                    'max_jobs': 10,
                    'launcher': 'srun',
                    'features': ['remote'],
                },
                {
                    'name': 'xfer',
                    'scheduler': 'slurm',
                    'time_limit': '10m',
                    'environs': ['builtin'],
                    'access': [
                        f'--partition=xfer',
                        f'--account={osext.osgroup()}'
                    ],
                    'descr': 'Nordend nodes for internal transfers',
                    'max_jobs': 10,
                    'launcher': 'srun',
                    'features': ['remote'],
                }
            ]
        },
        {
            'name': 'dom',
            'descr': 'Dom TDS',
            'hostnames': ['dom'],
            'modules_system': 'tmod',
            'resourcesdir': '/apps/common/UES/reframe/resources',
            'partitions': [
                {
                    'name': 'login',
                    'scheduler': 'local',
                    'time_limit': '10m',
                    'environs': [
                        'builtin',
                        'PrgEnv-cray',
                        'PrgEnv-gnu',
                        'PrgEnv-intel',
                        'PrgEnv-nvidia'
                    ],
                    'descr': 'Login nodes',
                    'max_jobs': 4,
                    'launcher': 'local'
                },
                {
                    'name': 'gpu',
                    'scheduler': 'slurm',
                    'time_limit': '10m',
                    'container_platforms': [
                        {
                            'type': 'Sarus',
                            'modules': ['sarus']
                        },
                        {
                            'type': 'Singularity',
                            'modules': ['singularity/3.8.0-daint']
                        }
                    ],
                    'modules': ['daint-gpu'],
                    'access': [
                        f'--constraint=gpu',
                        f'--account={osext.osgroup()}'
                    ],
                    'environs': [
                        'builtin',
                        'PrgEnv-cray',
                        'PrgEnv-gnu',
                        'PrgEnv-intel',
                        'PrgEnv-nvidia'
                    ],
                    'descr': 'Hybrid nodes (Haswell/P100)',
                    'max_jobs': 100,
                    'extras': {
                        'cn_memory': 64,
                    },
                    'launcher': 'srun',
                    'features': ['gpu', 'nvgpu', 'remote'],
                    'resources': [
                        {
                            'name': 'gres',
                            'options': ['--gres={gres}']
                        }
                    ],
                    'devices': [
                        {
                            'type': 'gpu',
                            'arch': 'sm_60',
                            'num_devices': 1
                        }
                    ]
                },
                {
                    'name': 'mc',
                    'scheduler': 'slurm',
                    'time_limit': '10m',
                    'container_platforms': [
                        {
                            'type': 'Sarus',
                            'modules': ['sarus']
                        },
                        {
                            'type': 'Singularity',
                            'modules': ['singularity/3.8.0-daint']
                        }
                    ],
                    'modules': ['daint-mc'],
                    'access': [
                        f'--constraint=mc',
                        f'--account={osext.osgroup()}'
                    ],
                    'environs': [
                        'builtin',
                        'PrgEnv-cray',
                        'PrgEnv-gnu',
                        'PrgEnv-intel',
                        'PrgEnv-nvidia'
                    ],
                    'descr': 'Multicore nodes (Broadwell)',
                    'max_jobs': 100,
                    'extras': {
                        'cn_memory': 64,
                    },
                    'features': ['remote'],
                    'resources': [
                        {
                            'name': 'gres',
                            'options': ['--gres={gres}']
                        }
                    ],
                    'launcher': 'srun'
                },
                {
                    'name': 'jupyter_gpu',
                    'scheduler': 'slurm',
                    'time_limit': '10m',
                    'environs': ['builtin'],
                    'access': [
                        f'-Cgpu',
                        f'--reservation=interact_gpu',
                        f'--account={osext.osgroup()}'
                    ],
                    'descr': 'JupyterHub GPU nodes',
                    'max_jobs': 10,
                    'launcher': 'srun',
                    'features': ['gpu', 'nvgpu', 'remote'],
                },
                {
                    'name': 'jupyter_mc',
                    'scheduler': 'slurm',
                    'time_limit': '10m',
                    'environs': ['builtin'],
                    'access': [
                        f'-Cmc',
                        f'--reservation=interact_mc',
                        f'--account={osext.osgroup()}'
                    ],
                    'descr': 'JupyterHub multicore nodes',
                    'max_jobs': 10,
                    'launcher': 'srun',
                    'features': ['remote'],
                },
                {
                    'name': 'xfer',
                    'scheduler': 'slurm',
                    'time_limit': '10m',
                    'environs': ['builtin'],
                    'access': [
                        f'--partition=xfer',
                        f'--account={osext.osgroup()}'
                    ],
                    'descr': 'Dedicated nodes for internal transfers',
                    'max_jobs': 10,
                    'launcher': 'srun',
                    'features': ['remote'],
                }
            ]
        },
        {
            'name': 'arolla',
            'descr': 'Arolla MCH',
            'hostnames': [r'arolla-\w+\d+'],
            'modules_system': 'tmod',
            'resourcesdir': '/apps/common/UES/reframe/resources',
            'partitions': [
                {
                    'name': 'login',
                    'scheduler': 'local',
                    'time_limit': '10m',
                    'environs': [
                        'PrgEnv-pgi',
                        'PrgEnv-pgi-nompi',
                        'PrgEnv-pgi-nompi-nocuda',
                        'PrgEnv-gnu',
                        'PrgEnv-gnu-nompi',
                        'PrgEnv-gnu-nompi-nocuda'
                    ],
                    'descr': 'Arolla login nodes',
                    'max_jobs': 4,
                    'launcher': 'local'
                },
                {
                    'name': 'pn',
                    'scheduler': 'slurm',
                    'time_limit': '10m',
                    'access': ['--partition=pn-regression'],
                    'environs': [
                        'PrgEnv-pgi',
                        'PrgEnv-pgi-nompi',
                        'PrgEnv-pgi-nompi-nocuda',
                        'PrgEnv-gnu',
                        'PrgEnv-gnu-nompi',
                        'PrgEnv-gnu-nompi-nocuda'
                    ],
                    'descr': 'Arolla post-processing nodes',
                    'max_jobs': 50,
                    'launcher': 'srun',
                    'features': ['remote'],
                },
                {
                    'name': 'cn',
                    'scheduler': 'slurm',
                    'time_limit': '10m',
                    'access': ['--partition=cn-regression'],
                    'environs': [
                        'PrgEnv-gnu',
                        'PrgEnv-gnu-nompi',
                        'PrgEnv-gnu-nompi-nocuda',
                        'PrgEnv-pgi',
                        'PrgEnv-pgi-nompi',
                        'PrgEnv-pgi-nompi-nocuda'
                    ],
                    'descr': 'Arolla compute nodes',
                    'features': ['gpu', 'nvgpu', 'remote'],
                    'devices': [
                        {
                            'type': 'gpu',
                            'arch': 'sm_70',
                            'num_devices': 8
                        }
                    ],
                    'resources': [
                        {
                            'name': '_rfm_gpu',
                            'options': ['--gres=gpu:{num_gpus_per_node}']
                        }
                    ],
                    'max_jobs': 50,
                    'launcher': 'srun'
                }
            ]
        },
        {
            'name': 'tsa',
            'descr': 'Tsa MCH',
            'hostnames': [r'tsa-\w+\d+'],
            'modules_system': 'tmod',
            'resourcesdir': '/apps/common/UES/reframe/resources',
            'partitions': [
                {
                    'name': 'login',
                    'scheduler': 'local',
                    'time_limit': '10m',
                    'environs': [
                        'PrgEnv-pgi',
                        'PrgEnv-pgi-nompi',
                        'PrgEnv-pgi-nocuda',
                        'PrgEnv-pgi-nompi-nocuda',
                        'PrgEnv-gnu',
                        'PrgEnv-gnu-nompi',
                        'PrgEnv-gnu-nocuda',
                        'PrgEnv-gnu-nompi-nocuda'
                    ],
                    'descr': 'Tsa login nodes',
                    'max_jobs': 4,
                    'launcher': 'local'
                },
                {
                    'name': 'pn',
                    'scheduler': 'slurm',
                    'time_limit': '10m',
                    'access': ['--partition=pn-regression'],
                    'environs': [
                        'PrgEnv-pgi',
                        'PrgEnv-pgi-nompi',
                        'PrgEnv-pgi-nocuda',
                        'PrgEnv-pgi-nompi-nocuda',
                        'PrgEnv-gnu',
                        'PrgEnv-gnu-nompi',
                        'PrgEnv-gnu-nocuda',
                        'PrgEnv-gnu-nompi-nocuda'
                    ],
                    'descr': 'Tsa post-processing nodes',
                    'max_jobs': 20,
                    'extras': {
                        'cn_memory': 377,
                    },
                    'launcher': 'srun',
                    'features': ['remote'],
                },
                {
                    'name': 'cn',
                    'scheduler': 'slurm',
                    'time_limit': '10m',
                    'access': ['--partition=cn-regression'],
                    'environs': [
                        'PrgEnv-gnu',
                        'PrgEnv-gnu-nompi',
                        'PrgEnv-gnu-nocuda',
                        'PrgEnv-gnu-nompi-nocuda',
                        'PrgEnv-pgi',
                        'PrgEnv-pgi-nompi',
                        'PrgEnv-pgi-nocuda',
                        'PrgEnv-pgi-nompi-nocuda'
                    ],
                    'descr': 'Tsa compute nodes',
                    'max_jobs': 20,
                    'extras': {
                        'cn_memory': 377,
                    },
                    'features': ['gpu', 'nvgpu', 'remote'],
                    'resources': [
                        {
                            'name': '_rfm_gpu',
                            'options': ['--gres=gpu:{num_gpus_per_node}']
                        }
                    ],
                    'devices': [
                        {
                            'type': 'gpu',
                            'arch': 'sm_70',
                            'num_devices': 8
                        }
                    ],
                    'launcher': 'srun'
                }
            ]
        },
        {
            'name': 'eiger',
            'descr': 'Alps Cray EX Supercomputer',
            'hostnames': ['eiger'],
            'modules_system': 'lmod',
            'resourcesdir': '/apps/common/UES/reframe/resources',
            'partitions': [
                {
                    'name': 'login',
                    'scheduler': 'local',
                    'time_limit': '10m',
                    'environs': [
                        'builtin',
                        'PrgEnv-aocc',
                        'PrgEnv-cray',
                        'PrgEnv-gnu',
                        'PrgEnv-intel',
                        'cpeAMD',
                        'cpeCray',
                        'cpeGNU',
                        'cpeIntel'
                    ],
                    'descr': 'Login nodes',
                    'max_jobs': 4,
                    'launcher': 'local'
                },
                {
                    'name': 'mc',
                    'descr': 'Multicore nodes (AMD EPYC 7742, 256|512GB/cn)',
                    'scheduler': 'slurm',
                    'time_limit': '10m',
                    'container_platforms': [
                        {
                            'type': 'Sarus',
                        },
                        {
                            'type': 'Singularity',
                            'modules': ['singularity/3.5.3-eiger']
                        }
                    ],
                    'environs': [
                        'builtin',
                        'PrgEnv-aocc',
                        'PrgEnv-cray',
                        'PrgEnv-gnu',
                        'PrgEnv-intel',
                        'cpeAMD',
                        'cpeCray',
                        'cpeGNU',
                        'cpeIntel'
                    ],
                    'max_jobs': 100,
                    'extras': {
                        'cn_memory': 256,
                    },
                    'features': ['remote'],
                    'access': ['-Cmc', f'--account={osext.osgroup()}'],
                    'resources': [
                        {
                            'name': 'switches',
                            'options': ['--switches={num_switches}']
                        },
                        {
                            'name': 'memory',
                            'options': ['--mem={mem_per_node}']
                        },
                    ],
                    'launcher': 'srun'
                },
                {
                    'name': 'jupyter_mc',
                    'scheduler': 'slurm',
                    'time_limit': '10m',
                    'environs': ['builtin'],
                    'access': [
                        f'-Cmc',
                        f'--reservation=interact',
                        f'--account={osext.osgroup()}'
                    ],
                    'descr': 'JupyterHub multicore nodes',
                    'max_jobs': 10,
                    'launcher': 'srun',
                    'features': ['remote'],
                },
            ]
        },
        {
            'name': 'Clariden',
            'descr': 'Clariden AI/ML cluster',
            'hostnames': ['clariden'],
            'modules_system': 'lmod',
            'partitions': [
                {
                    'name': 'login',
                    'scheduler': 'local',
                    'time_limit': '10m',
                    'environs': [
                        'builtin',
                        'PrgEnv-aocc',
                        'PrgEnv-cray',
                        'PrgEnv-gnu',
                        'PrgEnv-nvhpc',
                        'PrgEnv-nvidia'
                    ],
                    'descr': 'Login nodes',
                    'max_jobs': 4,
                    'launcher': 'local'
                },
                {
                    'name': 'amdgpu',
                    'scheduler': 'slurm',
                    'time_limit': '10m',
                    'environs': [
                        'builtin',
                        'PrgEnv-aocc',
                        'PrgEnv-cray',
                        'PrgEnv-gnu',
                        'PrgEnv-nvhpc',
                        'PrgEnv-nvidia'
                    ],
                    'container_platforms': [
                        {
                            'type': 'Sarus',
                        },
                        {
                            'type': 'Singularity',
                        }
                    ],
                    'max_jobs': 100,
                    'extras': {
                        'cn_memory': 500,
                    },
                    'access': ['-pamdgpu'],
                    'resources': [
                        {
                            'name': 'switches',
                            'options': ['--switches={num_switches}']
                        },
                        {
                            'name': 'memory',
                            'options': ['--mem={mem_per_node}']
                        },
                    ],
                    'features': ['gpu', 'amdgpu', 'remote'],
                    'devices': [
                        {
                            'type': 'gpu',
                            'arch': 'mi250',
                            'num_devices': 8
                        }
                    ],
                    'launcher': 'srun'
                },
                {
                    'name': 'nvgpu',
                    'scheduler': 'slurm',
                    'time_limit': '10m',
                    'environs': [
                        'builtin',
                        'PrgEnv-aocc',
                        'PrgEnv-cray',
                        'PrgEnv-gnu',
                        'PrgEnv-nvhpc',
                        'PrgEnv-nvidia'
                    ],
                    'container_platforms': [
                        {
                            'type': 'Sarus',
                        },
                        {
                            'type': 'Singularity',
                        }
                    ],
                    'max_jobs': 100,
                    'extras': {
                        'cn_memory': 500,
                    },
                    'access': ['-pnvgpu'],
                    'resources': [
                        {
                            'name': 'switches',
                            'options': ['--switches={num_switches}']
                        },
                        {
                            'name': 'memory',
                            'options': ['--mem={mem_per_node}']
                        },
                    ],
                    'features': ['gpu', 'nvgpu', 'remote'],
                    'devices': [
                        {
                            'type': 'gpu',
                            'arch': 'sm_80',
                            'num_devices': 4
                        }
                    ],
                    'launcher': 'srun'
                }
            ]
        },
        {
            'name': 'hohgant',
            'descr': 'Hohgant vcluster',
            'hostnames': ['hohgant'],
            'modules_system': 'lmod',
            'partitions': [
                {
                    'name': 'login',
                    'scheduler': 'local',
                    'time_limit': '10m',
                    'environs': [
                        'builtin',
                        'PrgEnv-aocc',
                        'PrgEnv-cray',
                        'PrgEnv-gnu',
                        'PrgEnv-nvhpc',
                        'PrgEnv-nvidia'
                    ],
                    'descr': 'Login nodes',
                    'max_jobs': 4,
                    'launcher': 'local'
                },
                {
                    'name': 'nvgpu',
                    'scheduler': 'slurm',
                    'time_limit': '10m',
                    'environs': [
                        'builtin',
                        'PrgEnv-aocc',
                        'PrgEnv-cray',
                        'PrgEnv-gnu',
                        'PrgEnv-nvhpc',
                        'PrgEnv-nvidia'
                    ],
                    'container_platforms': [
                        {
                            'type': 'Sarus',
                        },
                        {
                            'type': 'Singularity',
                        }
                    ],
                    'max_jobs': 100,
                    'extras': {
                        'cn_memory': 500,
                    },
                    'access': ['-pnvgpu'],
                    'resources': [
                        {
                            'name': 'switches',
                            'options': ['--switches={num_switches}']
                        },
                        {
                            'name': 'memory',
                            'options': ['--mem={mem_per_node}']
                        },
                    ],
                    'features': ['gpu', 'nvgpu', 'remote'],
                    'devices': [
                        {
                            'type': 'gpu',
                            'arch': 'sm_80',
                            'num_devices': 4
                        }
                    ],
                    'launcher': 'srun'
                },
                {
                    'name': 'amdgpu',
                    'scheduler': 'slurm',
                    'time_limit': '10m',
                    'environs': [
                        'builtin',
                        'PrgEnv-aocc',
                        'PrgEnv-cray',
                        'PrgEnv-gnu',
                        'PrgEnv-nvhpc',
                        'PrgEnv-nvidia'
                    ],
                    'container_platforms': [
                        {
                            'type': 'Sarus',
                        },
                        {
                            'type': 'Singularity',
                        }
                    ],
                    'max_jobs': 100,
                    'extras': {
                        'cn_memory': 500,
                    },
                    'access': ['-pamdgpu'],
                    'resources': [
                        {
                            'name': 'switches',
                            'options': ['--switches={num_switches}']
                        },
                        {
                            'name': 'memory',
                            'options': ['--mem={mem_per_node}']
                        },
                    ],
                    'features': ['gpu', 'amdgpu', 'remote'],
                    'launcher': 'srun'
                }
            ]
        },
        {
            'name': 'pilatus',
            'descr': 'Alps Cray EX Supercomputer TDS',
            'hostnames': ['pilatus'],
            'modules_system': 'lmod',
            'resourcesdir': '/apps/common/UES/reframe/resources',
            'partitions': [
                {
                    'name': 'login',
                    'scheduler': 'local',
                    'time_limit': '10m',
                    'environs': [
                        'builtin',
                        'PrgEnv-aocc',
                        'PrgEnv-cray',
                        'PrgEnv-gnu',
                        'PrgEnv-intel',
                        'cpeAMD',
                        'cpeCray',
                        'cpeGNU',
                        'cpeIntel'
                    ],
                    'descr': 'Login nodes',
                    'max_jobs': 4,
                    'launcher': 'local'
                },
                {
                    'name': 'mc',
                    'descr': 'Multicore nodes (AMD EPYC 7742, 256|512GB/cn)',
                    'scheduler': 'slurm',
                    'time_limit': '10m',
                    'container_platforms': [
                        {
                            'type': 'Sarus',
                        },
                        {
                            'type': 'Singularity',
                            'modules': ['singularity/3.5.3-eiger']
                        }
                    ],
                    'environs': [
                        'builtin',
                        'PrgEnv-aocc',
                        'PrgEnv-cray',
                        'PrgEnv-gnu',
                        'PrgEnv-intel',
                        'cpeAMD',
                        'cpeCray',
                        'cpeGNU',
                        'cpeIntel'
                    ],
                    'max_jobs': 100,
                    'extras': {
                        'cn_memory': 256,
                    },
                    'access': ['-Cmc', f'--account={osext.osgroup()}'],
                    'features': ['remote'],
                    'resources': [
                        {
                            'name': 'switches',
                            'options': ['--switches={num_switches}']
                        },
                        {
                            'name': 'memory',
                            'options': ['--mem={mem_per_node}']
                        },
                    ],
                    'launcher': 'srun'
                },
            ]
        },
        {
            'name': 'generic',
            'descr': 'Generic fallback system',
            'partitions': [
                {
                    'name': 'default',
                    'scheduler': 'local',
                    'environs': ['builtin'],
                    'descr': 'Login nodes',
                    'launcher': 'local'
                }
            ],
            'hostnames': ['.*']
        }
    ],
    'environments': [
        {
            'name': 'gnu',
            'modules': ['gcc'],
            'cc': 'gcc',
            'cxx': 'g++',
            'ftn': 'gfortran',
            'target_systems': ['ault']
        },
        {
            'name': 'cuda',
            'modules': ['gcc', 'cuda'],
            'cc': 'gcc',
            'cxx': 'g++',
            'ftn': 'gfortran',
            'target_systems': ['ault'],
            'features': ['cuda']
        },
        {
            'name': 'rocm',
            'modules': ['gcc', 'rocm'],
            'cc': 'gcc',
            'cxx': 'g++',
            'ftn': 'gfortran',
            'target_systems': ['ault'],
            'features': ['hip']
        },
        {
            'name': 'PrgEnv-pgi-nompi-nocuda',
            'target_systems': ['arolla'],
            'modules': ['PrgEnv-pgi/19.9-nocuda'],
            'cc': 'pgcc',
            'cxx': 'pgc++',
            'ftn': 'pgf90'
        },
        {
            'name': 'PrgEnv-pgi-nompi-nocuda',
            'target_systems': ['tsa'],
            'modules': ['PrgEnv-pgi/20.4-nocuda'],
            'cc': 'pgcc',
            'cxx': 'pgc++',
            'ftn': 'pgf90'
        },
        {
            'name': 'PrgEnv-pgi-nompi',
            'target_systems': ['arolla'],
            'modules': ['PrgEnv-pgi/19.9'],
            'features': ['cuda'],
            'cc': 'pgcc',
            'cxx': 'pgc++',
            'ftn': 'pgf90'
        },
        {
            'name': 'PrgEnv-pgi-nompi',
            'target_systems': ['tsa'],
            'modules': ['PrgEnv-pgi/20.4'],
            'features': ['cuda'],
            'cc': 'pgcc',
            'cxx': 'pgc++',
            'ftn': 'pgf90'
        },
        {
            'name': 'PrgEnv-pgi',
            'target_systems': ['arolla'],
            'modules': ['PrgEnv-pgi/19.9'],
            'features': ['cuda'],
            'cc': 'mpicc',
            'cxx': 'mpicxx',
            'ftn': 'mpifort'
        },
        {
            'name': 'PrgEnv-pgi',
            'target_systems': ['tsa'],
            'modules': ['PrgEnv-pgi/20.4'],
            'features': ['cuda'],
            'cc': 'mpicc',
            'cxx': 'mpicxx',
            'ftn': 'mpifort'
        },
        {
            'name': 'PrgEnv-pgi-nocuda',
            'target_systems': ['arolla'],
            'modules': ['PrgEnv-pgi/19.9-nocuda'],
            'cc': 'mpicc',
            'cxx': 'mpicxx',
            'ftn': 'mpifort'
        },
        {
            'name': 'PrgEnv-pgi-nocuda',
            'target_systems': ['tsa'],
            'modules': ['PrgEnv-pgi/20.4-nocuda'],
            'cc': 'mpicc',
            'cxx': 'mpicxx',
            'ftn': 'mpifort'
        },
        {
            'name': 'PrgEnv-gnu',
            'target_systems': ['arolla', 'tsa'],
            'modules': ['PrgEnv-gnu/19.2'],
            'features': ['cuda'],
            'cc': 'mpicc',
            'cxx': 'mpicxx',
            'ftn': 'mpifort'
        },
        {
            'name': 'PrgEnv-gnu-nocuda',
            'target_systems': ['arolla', 'tsa'],
            'modules': ['PrgEnv-gnu/19.2-nocuda'],
            'cc': 'mpicc',
            'cxx': 'mpicxx',
            'ftn': 'mpifort'
        },
        {
            'name': 'PrgEnv-gnu-nompi',
            'target_systems': ['arolla', 'tsa'],
            'modules': ['PrgEnv-gnu/19.2'],
            'features': ['cuda'],
            'cc': 'gcc',
            'cxx': 'g++',
            'ftn': 'gfortran'
        },
        {
            'name': 'PrgEnv-gnu-nompi-nocuda',
            'target_systems': ['arolla', 'tsa'],
            'modules': ['PrgEnv-gnu/19.2-nocuda'],
            'cc': 'gcc',
            'cxx': 'g++',
            'ftn': 'gfortran'
        },
        {
            'name': 'PrgEnv-aocc',
            'target_systems': ['eiger', 'pilatus'],
            'modules': ['PrgEnv-aocc']
        },
        {
            'name': 'PrgEnv-cray',
            'target_systems': ['eiger', 'pilatus'],
            'modules': ['PrgEnv-cray']
        },
        {
            'name': 'PrgEnv-gnu',
            'target_systems': ['eiger', 'pilatus'],
            'modules': ['PrgEnv-gnu']
        },
        {
            'name': 'PrgEnv-intel',
            'target_systems': ['eiger', 'pilatus'],
            'modules': ['PrgEnv-intel']
        },
        {
            'name': 'PrgEnv-aocc',
            'target_systems': ['hohgant'],
            'modules': ['cray', 'PrgEnv-aocc']
        },
        {
            'name': 'PrgEnv-cray',
            'target_systems': ['hohgant'],
            'modules': ['cray', 'PrgEnv-cray']
        },
        {
            'name': 'PrgEnv-gnu',
            'target_systems': ['hohgant'],
            'modules': ['cray', 'PrgEnv-gnu']
        },
        {
            'name': 'PrgEnv-intel',
            'target_systems': ['hohgant'],
            'modules': ['cray', 'PrgEnv-intel']
        },
        {
            'name': 'PrgEnv-nvhpc',
            'target_systems': ['hohgant'],
            'modules': ['cray', 'PrgEnv-nvhpc'],
            'extras': {
                'launcher_options': '--mpi=pmi2',
            },
        },
        {
            'name': 'PrgEnv-nvidia',
            'target_systems': ['hohgant'],
            'modules': ['cray', 'PrgEnv-nvidia'],
            'extras': {
                # "MPIR_pmi_init(83)....: PMI2_Job_GetId returned 14"
                # -> add --mpi=pmi2 at runtime
                'launcher_options': '--mpi=pmi2',
            },
        },
        {
            'name': 'cpeAMD',
            'target_systems': ['eiger', 'pilatus'],
            'modules': ['cpeAMD']
        },
        {
            'name': 'cpeCray',
            'target_systems': ['eiger', 'pilatus'],
            'modules': ['cpeCray']
        },
        {
            'name': 'cpeGNU',
            'target_systems': ['eiger', 'pilatus'],
            'modules': ['cpeGNU']
        },
        {
            'name': 'cpeIntel',
            'target_systems': ['eiger', 'pilatus'],
            'modules': ['cpeIntel']
        },
        {
            'name': 'PrgEnv-cray',
            'modules': ['PrgEnv-cray']
        },
        {
            'name': 'PrgEnv-gnu',
            'modules': ['PrgEnv-gnu']
        },
        {
            'name': 'PrgEnv-intel',
            'modules': ['PrgEnv-intel']
        },
        {
            'name': 'PrgEnv-pgi',
            'modules': ['PrgEnv-pgi']
        },
        {
            'name': 'PrgEnv-nvidia',
            'target_systems': ['pilatus'],
            'modules': [
                'PrgEnv-nvidia',
                # FIXME: We should not be forcing a cdt version
                'cpe/21.06'
            ]
        },
        {
            'name': 'PrgEnv-nvidia',
            'modules': ['PrgEnv-nvidia'],
            'features': ['cuda'],
            'target_systems': ['daint', 'dom'],
        },
        {
            'name': 'builtin',
            'cc': 'cc',
            'cxx': 'CC',
            'ftn': 'ftn'
        },
        {
            'name': 'builtin-gcc',
            'cc': 'gcc',
            'cxx': 'g++',
            'ftn': 'gfortran'
        }
    ],
    'logging': [
        {
            'perflog_compat': True,
            'handlers': [
                {
                    'type': 'file',
                    'name': 'reframe.log',
                    'level': 'debug2',
                    'format': '[%(asctime)s] %(levelname)s: %(check_info)s: %(message)s',   # noqa: E501
                    'append': False
                },
                {
                    'type': 'stream',
                    'name': 'stdout',
                    'level': 'info',
                    'format': '%(message)s'
                },
                {
                    'type': 'file',
                    'name': 'reframe.out',
                    'level': 'info',
                    'format': '%(message)s',
                    'append': False
                }
            ],
            'handlers_perflog': [
                {
                    'type': 'filelog',
                    'prefix': '%(check_system)s/%(check_partition)s',
                    'level': 'info',
                    'format': '%(check_job_completion_time)s|reframe %(version)s|%(check_info)s|jobid=%(check_jobid)s|num_tasks=%(check_num_tasks)s|%(check_perf_var)s=%(check_perf_value)s|ref=%(check_perf_ref)s (l=%(check_perf_lower_thres)s, u=%(check_perf_upper_thres)s)|%(check_perf_unit)s',   # noqa: E501
                    'datefmt': '%FT%T%:z',
                    'append': True
                },
                {
                    'type': 'httpjson',
                    'url': 'http://httpjson-server:12345/rfm',
                    'level': 'info',
                    'extras': {
                        'facility': 'reframe',
                        'data-version': '1.0',
                    },
                    'ignore_keys': ['check_perfvalues']
                }
            ]
        }
    ],
    'modes': [
        {
            'name': 'maintenance',
            'options': [
                '--unload-module=reframe',
                '--exec-policy=async',
                '-Sstrict_check=1',
                '--output=$APPS/UES/$USER/regression/maintenance',
                '--perflogdir=$APPS/UES/$USER/regression/maintenance/logs',
                '--stage=$SCRATCH/regression/maintenance/stage',
                '--report-file=$APPS/UES/$USER/regression/maintenance/reports/maint_report_{sessionid}.json',
                '-Jreservation=maintenance',
                '--save-log-files',
                '--tag=maintenance',
                '--timestamp=%F_%H-%M-%S'
            ]
        },
        {
            'name': 'production',
            'options': [
                '--unload-module=reframe',
                '--exec-policy=async',
                '-Sstrict_check=1',
                '--output=$APPS/UES/$USER/regression/production',
                '--perflogdir=$APPS/UES/$USER/regression/production/logs',
                '--stage=$SCRATCH/regression/production/stage',
                '--report-file=$APPS/UES/$USER/regression/production/reports/prod_report_{sessionid}.json',
                '--save-log-files',
                '--tag=production',
                '--timestamp=%F_%H-%M-%S'
            ]
        },
        {
            'name': 'production',
            'options': [
                '--unload-module=reframe',
                '--exec-policy=async',
                '-Sstrict_check=1',
                '--prefix=$SCRATCH/$USER/regression/production',
                '--report-file=$SCRATCH/$USER/regression/production/reports/prod_report_{sessionid}.json',
                '--save-log-files',
                '--tag=production',
                '--timestamp=%F_%H-%M-%S'
            ],
            'target_systems': ['hohgant'],
        }
    ],
    'general': [
        {
            'check_search_path': ['checks/'],
            'check_search_recursive': True,
            'remote_detect': True
        }
    ]
}
