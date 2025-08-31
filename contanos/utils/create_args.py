import os
import argparse


def add_argument(parser, name, env_name, env_value, arg_type=None, nargs=None):
    parser.add_argument(
        f'--{name}',
        type=str if arg_type is None else arg_type,
        nargs=1 if nargs is None else nargs,
        default=os.getenv(env_name, env_value),
        help=env_name.replace('_', ' ').upper()
    )

def add_service_args(parser):
    """Add service configuration arguments to parser"""
    parser.add_argument(
        '--health_check_interval',
        type=int,
        default=int(os.getenv('HEALTH_CHECK_INTERVAL', '30')),
        help='Health check interval in seconds'
    )
    parser.add_argument(
        '--max_restart_attempts',
        type=int,
        default=int(os.getenv('MAX_RESTART_ATTEMPTS', '3')),
        help='Maximum restart attempts'
    )
    parser.add_argument(
        '--restart_cooldown',
        type=int,
        default=int(os.getenv('RESTART_COOLDOWN', '60')),
        help='Restart cooldown period in seconds'
    )
    parser.add_argument(
        '--run_until_complete',
        action='store_true',
        default=os.getenv('RUN_UNTIL_COMPLETE', 'true').lower() == 'true',
        help='Run until complete'
    )
    parser.add_argument(
        '--daemon_mode',
        action='store_true',
        default=os.getenv('DAEMON_MODE', 'false').lower() == 'true',
        help='Run in daemon mode'
    )


def add_compute_args(parser):
    """Add compute configuration arguments to parser"""
    parser.add_argument(
        '--num_workers_per_device',
        type=int,
        default=int(os.getenv('NUM_WORKERS_PER_DEVICE', '1')),
        help='Number of workers per device'
    )
    parser.add_argument(
        '--backend',
        default=os.getenv('BACKEND', 'onnxruntime'),
        help='Inference backend for rtmlib.Body (opencv, onnxruntime, openvino)'
    )