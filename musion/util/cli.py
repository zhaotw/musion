import argparse

import musion
from musion.util.pkg_util import task_names, get_task_instance

def main():
    parser = argparse.ArgumentParser(description="""
    MUSION
    """)

    parser.add_argument('task', type=str, choices=task_names, help='Choose the task you would like to perform.')
    parser.add_argument('--audio_path', type=str, help='Absolute file path to perform the task.')
    parser.add_argument('--dir_path', type=str, help='Directory that contains audio files you want to process.')
    parser.add_argument('--num_threads', type=int, default=0,
                        help='Used with --dir_path. Set to a proper number to enable parallel processing.')
    parser.add_argument('--print_result', action='store_true', help='Whether to print the task result(s)')
    parser.add_argument('--show_keys', action='store_true', help='Print the result keys for the specific task.')
    parser.add_argument('--save_dir', type=str,
                        help='Directory path that you want to save the results in. Should also proivde --save_keys.')
    parser.add_argument('--save_keys', type=str, nargs='+',
                        help='Choose which keys to save in a file. Query for avalible keys by --show_keys.')

    args = parser.parse_args()

    musion_task = get_task_instance(args.task)

    if args.show_keys:
        print(musion_task.result_keys)
        return

    args_dict = vars(args)
    if args.save_dir:
        if not args.save_keys:
            raise ValueError('Must provide --save_keys if you want to save a file for the result.')
        save_cfg = musion.SaveConfig(args.save_dir, args.save_keys)
        args_dict.update({'save_cfg': save_cfg})

    if args.audio_path:
        res = musion_task(**args_dict)
    elif args.dir_path:
        res = musion_task.process_dir(**args_dict)
    else:
        raise ValueError('Must provide either --audio_path or --dir_path to proceed.')

    if args.print_result:
        print(res)

if __name__ == '__main__':
    main()
