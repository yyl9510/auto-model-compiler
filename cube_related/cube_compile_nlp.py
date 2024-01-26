# pip install sentencepiece transformers fuzzywuzzy concurrent-log-handler psutilfrom multiprocessing.managers import ListProxy
# pip install transformers timeout-decorator
import warnings

from requests import HTTPError
warnings.filterwarnings("ignore")

import os
import subprocess
import traceback
from _collections_abc import MutableMapping

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM
from cube.graph.parser.converter import to_fx_graph

import psutil
import logging
import time
import timeout_decorator
from cube.runtime.utils import microbatches
import cube
from examples.utils import get_policy
import examples.mlp.policy.gallery as gallery
from fairseq.cube.pas_policies import PASRandomSPMD
from functools import partial
import inspect

text: str = "Huggingface is a really excellent project!"
cache_dir: str = "/mnt/msrasrg/yileiyang/hf_cache"

current_file_path = os.path.abspath(__file__)
current_folder = os.path.dirname(current_file_path)
model_name_list_path = os.path.join(current_folder, "models/Natural Language Processing")  # Natural Language Processing
log_dir = os.path.join(current_folder, "logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logger_list = []

def setup_logger(log_file, level = logging.INFO, need_timestamp = True):
    # different process has different logger, with timestamp
    logger = logging.getLogger(log_file)
    logger.setLevel(level)
    # logger will only init once for one log_file
    if not logger.handlers:
        handler = logging.FileHandler(log_file, "a")
        if need_timestamp:
            formatter = logging.Formatter('%(asctime)s [PID %(process)d][%(levelname)s]: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
            handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger_list.append(logger)
    return logger

def logger_redirect(logger1, to_logger_file, prefix = '', need_timestamp=True) -> logging.FileHandler:
    import logging
    result_handler = logging.FileHandler(to_logger_file, 'a')
    if need_timestamp:
        formatter = logging.Formatter(f'%(asctime)s [PID %(process)d][%(levelname)s]: {prefix} %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    else:
        formatter = logging.Formatter(f'{prefix} %(message)s')
    result_handler.setFormatter(formatter)
    logger1.addHandler(result_handler)
    return result_handler

loglevel = logging.INFO
info_path = os.path.join(log_dir, f'cube_compile_1_info.log')
logger = setup_logger(os.path.join(log_dir, info_path), loglevel)

tried_logger = setup_logger(os.path.join(log_dir, f'cube_compile_2_tried.log'), loglevel, False)
loaded_logger = setup_logger(os.path.join(log_dir, f'cube_compile_3_loaded.log'), loglevel, False)
traced_logger = setup_logger(os.path.join(log_dir, f'cube_compile_4_traced.log'), loglevel, False)
trace_aligned_logger = setup_logger(os.path.join(log_dir, f'cube_compile_5_trace_aligned.log'), loglevel, False)
compiled_logger = setup_logger(os.path.join(log_dir, f'cube_compile_6_compiled.log'), loglevel, False)
aligned_logger = setup_logger(os.path.join(log_dir, f'cube_compile_7_compile_aligned.log'), loglevel, False)

logger_redirect(tried_logger, os.path.join(log_dir, info_path), prefix="model tried: ")
logger_redirect(loaded_logger, os.path.join(log_dir, info_path), prefix="model loaded: ")
logger_redirect(traced_logger, os.path.join(log_dir, info_path), prefix="model traced: ")
logger_redirect(trace_aligned_logger, os.path.join(log_dir, info_path), prefix="model trace aligned: ")
logger_redirect(compiled_logger, os.path.join(log_dir, info_path), prefix="model compiled: ")
logger_redirect(aligned_logger, os.path.join(log_dir, info_path), prefix="model aligned: ")

error_out_cube_logger = setup_logger(os.path.join(log_dir, f'cube_compile_8_error_out_cube.log'), loglevel)
error_in_cube_logger = setup_logger(os.path.join(log_dir, f'cube_compile_9_error_in_cube.log'), loglevel)

logger_redirect(error_in_cube_logger, os.path.join(log_dir, info_path), "", False)
logger_redirect(error_out_cube_logger, os.path.join(log_dir, info_path), "", False)

torch.set_printoptions(edgeitems = 2)

cube.init()
# # get policy
# policy = get_policy([gallery], "PASRandomSPMD")
# policy = partial(policy, torch.distributed.get_world_size(), seed=100)

for tmp_logger in logger_list:
    if torch.distributed.get_rank() == 0:
        tmp_logger.setLevel(logging.INFO)
    else:
        tmp_logger.setLevel(logging.WARNING)

# @timeout_decorator.timeout(60, timeout_exception=TimeoutError)
def load_model_by_config(config, trust_remote_code = True):
    torch.manual_seed(0)
    return AutoModel.from_config(config, trust_remote_code=trust_remote_code)

# @timeout_decorator.timeout(10, timeout_exception=TimeoutError)
def load_model_by_pretrain(model_name, cache_dir, trust_remote_code = True):
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=trust_remote_code)

# from transformers import cached_path, WEIGHTS_NAME 
def load_model(config, model_name, cache_dir):
    try:
        model = load_model_by_config(config)
        return model
    except Exception:
        logger.info("logging by config failed, try from_pretrained")
        error_message = traceback.format_exc().strip() + "\n"
        logger.error(error_message)
        try:
            # if torch.distributed.get_rank() == 0:
            #     logger.info("cache by rank 0")
            #     success_flag = torch.tensor([1])
            #     model = load_model_by_pretrain(model_name, cache_dir=cache_dir)
            #     logger.info("pretrained weight cached")
            # else:
            #     success_flag = torch.tensor([0])
            # torch.distributed.broadcast(success_flag, src=0)
            # torch.distributed.barrier()
            # if success_flag.item() == 1:
            model = load_model_by_pretrain(model_name, cache_dir=cache_dir)
            # else:
            #     raise RuntimeError(f"{model_name} not loaded successfully")
            return model
        except Exception:
            logger.info("logging by pretrain failed, exit loading")
            error_message = traceback.format_exc().strip() + "\n"
            logger.error(error_message)
            raise
        
    


def print_memory_usage(logger, prefix : str = ""):
    process = psutil.Process()
    mem_info = process.memory_info()
    logger.debug("When " + prefix + f": Current memory usage: {mem_info.rss / (1024 ** 3):.2f} GB")
    try:
        smi_output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        memory_info = smi_output.strip().split('\n')
        gpu_mem_tuple = []
        for idx, mem in enumerate(memory_info):
            used, total = mem.split(', ')
            gpu_mem_tuple.append((idx, int(used) / 1024, int(total) / 1024))
        logger.debug(f"GPU memory usage (index, used-GB, total-GB): {gpu_mem_tuple}")
    except subprocess.CalledProcessError as e:
        logger.error("Can't execute nvidia-smi command:", e.output)
    except FileNotFoundError:
        logger.error("nvidia-smi command not found , make sure nvidia driver has been install successfully.")

def concrete_trace_wrap(model, dummy_input):
    if torch.cuda.is_available():
        try:
            traced_gm = to_fx_graph(model, dummy_input)

        except:
            raise Exception("Failed to trace with gpu")
        print("Successfully traced with gpu")
        return traced_gm
    else:
        raise RuntimeError("CUDA is not available")

def check_align(before_trace, after_trace):
    for key in after_trace.keys():
        if isinstance(after_trace[key], torch.Tensor):
            if not torch.allclose(before_trace[key].to(torch.cuda.current_device()),
                                  after_trace[key].to(torch.cuda.current_device()), 
                                  rtol=1e-04, atol=1e-04):
                return False
    return True

from cube.parallel import ComputeConfig
from cube.graph.function.anchor import IRGraphAnchor
from cube.graph.function.dimops import IRDimops
from cube.graph.graph import IRGraph
from cube.graph.segment import IRSegment
from cube.ir.operator import IRDataOperation, IRFwOperation
from typing import List, Optional
import random

def _tp(graph: IRGraph, node: IRDimops, devs: List[int], idx: int, dim: int):
    sub_nodes = graph.partition(
        node, node.algorithms('dim'), idx=idx, dim=dim, num=len(devs))
    for devid, sub_node in zip(devs, sub_nodes):
        graph.assign(sub_node, devid)
    return sub_nodes

def _replica(graph: IRGraph, node, devs: List[int]):
    sub_nodes = graph.replicate(node, times=len(devs))
    for devid, sub_node in zip(devs, sub_nodes):
        graph.assign(sub_node, devid)
    return sub_nodes

from cube.runtime.resource import EnvResource

def PASRandomSPMD(graph: IRGraph, env_resource: EnvResource):
    """
    Random SPMD policy
    """
    ngpus = env_resource.ngpus
    # get the current random state
    state = random.getstate()

    seed = 1
    # print(f'> set random SPDM policy seed to {seed}')
    random.seed(seed)
    devs = list(range(ngpus))

    for ftensor in graph.full_tensors():
        if ftensor.is_grad(): continue
        if len(graph.consumers(ftensor)) > 1:
            graph.multiref(ftensor)

    for node in graph.select(ntype=(IRFwOperation, IRDataOperation)):
        if node.name == 'multiref' or isinstance(node, IRGraphAnchor):
            continue
        if isinstance(node, IRDimops):
            configs = node.transform_space()
            if len(configs) == 0:
                _replica(graph, node, devs)
            else:
                configs = sorted(configs, reverse=True,
                                 key=lambda config: node.input(config[0]).shape[config[1]])
                random.shuffle(configs)
                for (idx, dim) in configs:
                    if node.input(idx).shape[dim] % len(devs) != 0: continue
                    if node.algorithms('dim').satisfy(idx=idx, dim=dim, num=len(devs)):
                        # print(f'> partition node {node.name} ({node.cid}) with config idx={idx}, dim={dim}')
                        _tp(graph, node, devs, idx, dim)
                        break
                else:
                    _replica(graph, node, devs)
        else:
            _replica(graph, node, devs)

    # restore the random state
    random.setstate(state)
    # print(graph.extra_repr())
    return graph

# from fairseq.cube.pas_policies import PASRandomSPMD
# policy = partial(PASRandomSPMD, torch.distributed.get_world_size())

# def trace_worker(model_name: str):
#     import multiprocessing
#     p = multiprocessing.Process(target=cube_compile_check, args=(model_name, ))   # , daemon=True
#     p.start()
#     p.join()

def cube_compile_check(model_name: str):
    try:
        start_time = time.time()
        if torch.distributed.get_rank() == 0:
            subprocess.run('rm -rf gencode*.py fullmodel.pt.* dist_param_map.pt', shell=True, check=True)
        # load tokenizer, config, model

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
        logger.info(f"{model_name} Tokenizer loaded")

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir)
        logger.info(f"{model_name} config loaded")

        model = load_model(config, model_name, cache_dir)
        loaded_logger.info(f"{model_name}, {config.architectures if 'config' in locals() and config else None}")
        logger.info(f"{model_name} has parameter: {sum(p.numel() for p in model.parameters())}")
        print_memory_usage(logger, f"after load model {model_name}")

        # build dummy_input and forward
        dummy_input = tokenizer(text, return_tensors="pt")
        if isinstance(dummy_input, MutableMapping):
            dummy_input = dict(dummy_input)
        logger.debug(f"{model_name} tokenized")
        logger.debug(f"dummy_input: {dummy_input}")
        
        model.eval()
        before_trace = model(**dummy_input)
        # logger.debug(f"original logit: {before_trace['last_hidden_state'][:2]}")

        if (torch.distributed.get_world_size() == 1 and torch.distributed.get_rank() == 0) or \
            (torch.distributed.get_world_size() > 1 and torch.distributed.get_rank() == 1):

            traced_gm = concrete_trace_wrap(model, dummy_input)
            traced_logger.info(f"{model_name}, {config.architectures if 'config' in locals() and config else None}")
            traced_gm.eval()
            after_trace = traced_gm(**dummy_input)
            # logger.debug(f"traced logit: {after_trace['last_hidden_state'][:2]}")

            if check_align(before_trace, after_trace):
                trace_aligned_logger.info(f"{model_name}, {config.architectures}")
            else:
                error_in_cube_logger.error(f"{model_name} not aligned before and after trace\n before trace: {before_trace}\n after trace: {after_trace}\n")

        # cube compile model
        forward_signature = inspect.signature(model.forward)
        if 'decoder_input_ids' in forward_signature.parameters:
            dummy_input['decoder_input_ids'] = dummy_input.get('input_ids', None)
        params_with_defaults = [
            v.default if k not in dummy_input else dummy_input[k].to(torch.cuda.current_device())
            for k, v in forward_signature.parameters.items()
            if v.default is not inspect.Parameter.empty
        ]
        logger.debug(f"forward_signature: {forward_signature}")
        logger.debug(f"params_with_defaults: {params_with_defaults}")
        dataloader = microbatches((params_with_defaults, ) * 2)
        
        model = load_model(config, model_name, cache_dir)
        @cube.compile(model, dataloader, PAS=PASRandomSPMD)
        def train_iter(model, dataloader):
            data = next(dataloader)
            logit = model(*data)
            return logit

        # torch.distributed.barrier()

        smodel = cube.utils.load_model()
        compiled_logger.info(f"{model_name}, {config.architectures if 'config' in locals() and config else None}")
        smodel.eval()
        compiled_logit = train_iter(smodel, dataloader)
        logger.debug(f"compiled logit: {compiled_logit['last_hidden_state'][:2]}")

        if check_align(before_trace, compiled_logit):
            aligned_logger.info(f"aligned before trace and after compile: {model_name}, {config.architectures}")
        else:
            error_in_cube_logger.error(f"{model_name} not aligned before trace and after compile\n before trace: {before_trace}\n after trace: {compiled_logit}\n")

    except (TimeoutError, HTTPError, OSError, NameError, RuntimeError, KeyError) as e:
        if torch.distributed.get_rank() == 0:
            logger.error(f"fail when loading model: {model_name}", exc_info=False)
            error_message = traceback.format_exc().strip() + "\n"
            error_out_cube_logger.error(f"{model_name}, {config.architectures if 'config' in locals() and config else None}, failed")
            error_out_cube_logger.error(error_message)
    except Exception as e:
        if torch.distributed.get_rank() == 0:
            torch.distributed.barrier()

            logger.error(f"fail when compiling model: {model_name}", exc_info=False)
            error_message = traceback.format_exc().strip() + "\n"
            if 'MagicCube' in error_message:
                error_in_cube_logger.error(f"{model_name}, {config.architectures if 'config' in locals() and config else None}, failed")
                error_in_cube_logger.error(error_message)
            else:
                error_out_cube_logger.error(f"{model_name}, {config.architectures if 'config' in locals() and config else None}, failed")
                error_out_cube_logger.error(error_message)
    finally:
        end_time = time.time()
        tried_logger.info(f"{model_name}")
        logger.info(f"Finish trying model: {model_name}, time: {end_time - start_time:.2f} s")
        torch.distributed.barrier()


if __name__ == "__main__":
    model_names = []
    with open(model_name_list_path, 'r') as f:
        for line in f:
            model_name = line.strip()
            if model_name and model_name not in model_names:
                model_names.append(model_name)
    print(f"# model_name_list: {len(model_names)}")

    tried_models = []
    if os.path.exists(os.path.join(log_dir, "cube_compile_2_tried.log")):
        with open(os.path.join(log_dir, "cube_compile_2_tried.log"), 'r') as file:
            tried_models = [line.strip() for line in file]
    model_name_list = [model for model in model_names if model not in tried_models]
    print(f"# already_tried: {len(tried_models)}")
    print(f"# need_to_try: {len(model_name_list)}")

    model_numbers = 0
    torch.distributed.barrier()
    for model_name in model_name_list:
        cube_compile_check(model_name)
        model_numbers += 1
        logger.info(f"Process: {model_numbers} / {len(model_name_list)}, Percentage: {model_numbers / len(model_name_list) * 100:.2f}%\n\n")
