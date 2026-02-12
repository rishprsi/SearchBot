import argparse


def get_parser(parser, commands, opt_args, choice_args, bool_args, query_type, help):
    subparsers = parser.add_subparsers(dest="command", description="Available commands")
    command_parsers = dict()

    for command in commands:
        command_parsers[command] = subparsers.add_parser(command, help=help[command])
        for argument in commands[command]:
            command_parsers[command].add_argument(
                argument, type=query_type[argument], help=help[argument]
            )
        if command in opt_args:
            for opt_arg, default_size in opt_args[command]:
                command_parsers[command].add_argument(
                    "--" + opt_arg,
                    type=query_type[opt_arg],
                    help=help[opt_arg],
                    default=default_size,
                )
        if command in bool_args:
            for bool_arg, action in bool_args[command]:
                command_parsers[command].add_argument(
                    "--" + bool_arg, action=action, help=help[bool_arg]
                )
        if command in choice_args:
            for choice_arg in choice_args[command].keys():
                command_parsers[command].add_argument(
                    "--" + choice_arg,
                    type=query_type[choice_arg],
                    choices=choice_args[command][choice_arg],
                    help=help[choice_arg],
                )
    return parser
