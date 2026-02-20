def parse_cmd_arguments(parser) -> object:
    parser.add_argument('--data_resolution', type=int, default=5, help='resolution of data in minutes')
    parser.add_argument('--valid_portion', type=int, default=10 * 24 * 12, help='number of hours used for validation')
    parser.add_argument('--label_differencing', type=bool, default=True, help='if True delta_bgl is used as label')
    parser.add_argument('--seq_len_in_minutes', type=int, default=3 * 60, help='sequence length in minutes as input)')
    parser.add_argument('--return_sequence', type=bool, default=False, help='return sequence')
    parser.add_argument('--keep_future', type=bool, default=False, help='keep future disturbances in the train inputs')
    return parser.parse_args()

def customize_args(args):
    args.pid_list = ['540' , '544', '552', '567', '584', '596', '559', '563', '570', '575', '588', '591'] ##
    args.data_files = [pID + '-ws-training.xml' for pID in args.pid_list]
    args.data_files_test = [pID + '-ws-testing.xml' for pID in args.pid_list]
    args.scale_mean = {p: 0. for p in args.pid_list}
    args.scale_std = {p: 1. for p in args.pid_list}
    return args

def init_para(argparse):
    args = parse_cmd_arguments(argparse)
    args.seq_len = args.seq_len_in_minutes // args.data_resolution
    args.pred_horizon = args.pred_horizon_in_minutes // args.data_resolution
    args = customize_args(args)

    return args
