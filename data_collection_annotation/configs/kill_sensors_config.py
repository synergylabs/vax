"""
Checkpoint and runtime configs for all processes
"""

sensors_ckpt_config = {
    'micarray': {
        'ckpt_file': '/tmp/micarray.ckpt',
        'proc_file': 'run_micarrayv2.py',
        'launch_file': './launch_micarray.sh'
    },
    'audio': {
        'ckpt_file': '/tmp/audio.ckpt',
        'proc_file': 'run_audio.py',
        'launch_file': './launch_audio.sh'
    },
    'doppler': {
        'ckpt_file': '/tmp/doppler.ckpt',
        'proc_file': 'run_doppler.py',
        'launch_file': './launch_doppler.sh'
    },
    'lidar2d': {
        'ckpt_file': '/tmp/lidar2d.ckpt',
        'proc_file': 'run_lidar2d.py',
        'launch_file': './launch_lidar2d.sh'
    },
    'oakdlite': {
        'ckpt_file': '/tmp/oakdlite.ckpt',
        'proc_file': 'run_oakdlite.py',
        'launch_file': './launch_oakdlite.sh'
    },
    'thermal': {
        'ckpt_file': '/tmp/thermal.ckpt',
        'proc_file': 'run_thermal.py',
        'launch_file': './launch_thermal.sh'
    },
}