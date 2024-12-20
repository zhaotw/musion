import argparse
import logging

import musion
from musion.utils.pkg_util import task_names, get_task_instance

def main():
    parser = argparse.ArgumentParser(description="""
    MUSION
    """)

    parser.add_argument('task', type=str, choices=task_names, help='Choose the task you would like to perform.')
    parser.add_argument('audio_path', type=str, nargs='+', 
                        help='Absolute file path to perform the task. Could be any number of files or a directory.')
    parser.add_argument('--num_threads', type=int, default=0,
                        help='Set to a proper number to enable parallel processing.')
    parser.add_argument('--print_result', action='store_true', help='Whether to print the task result(s)')
    parser.add_argument('--show_keys', action='store_true', help='Print the result keys for the specific task.')
    parser.add_argument('--save_dir', type=str,
                        help='Directory path that you want to save the results in. Will save all result by default, \
                              you may also proivde --save_keys to only choose certain keys to save.')
    parser.add_argument('--save_keys', type=str, nargs='+',
                        help='Choose which keys to save in a file. Query for avalible keys by --show_keys.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite the results if they already exist.')
    parser.add_argument('--target_instrument', type=str, choices=['piano', 'vocal'],
                        help='Specify the target instrument for the transcribe task.')

    args = parser.parse_args()

    init_kwargs = {}
    if args.task == 'transcribe':
        if not args.target_instrument:
            raise ValueError('Please add --target_instrument for the transcribe task.')
        init_kwargs['target_instrument'] = args.target_instrument

    musion_task = get_task_instance(args.task, **init_kwargs)

    if args.show_keys:
        print(musion_task.result_keys)
        return

    args_dict = vars(args)
    if args.save_dir:
        if not args.save_keys:
            logging.info(f'--save_keys not provided. Will save all the following results: {musion_task.result_keys}')
        save_cfg = musion.SaveConfig(args.save_dir, args.save_keys)
        args_dict.update({'save_cfg': save_cfg})
    
    if len(args_dict['audio_path']) == 1:
        args_dict['audio_path'] = args_dict['audio_path'][0]

    res = musion_task(**args_dict)

    if args.print_result:
        print(res)

if __name__ == '__main__':
    main()
