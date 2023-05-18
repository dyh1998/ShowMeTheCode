from simpletransformers.config.model_args import Seq2SeqArgs, ModelArgs
from simpletransformers.config.global_args import global_args


def load_model_args(model_args_dir):
    args = Seq2SeqArgs()
    args.load(model_args_dir)
    return args


args = Seq2SeqArgs()
print(args)

args = load_model_args('facebook/bart-base')
print(args)

args = ModelArgs()
print(args)
