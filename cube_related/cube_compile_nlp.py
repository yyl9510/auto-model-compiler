# pip install sentencepiece transformers fuzzywuzzy concurrent-log-handler psutil protobuf
# pip install transformers timeout-decorator protobuf sacremoses
import warnings

from requests import HTTPError
warnings.filterwarnings("ignore")

import os
log_dir = os.path.expanduser('~/hf_logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

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
import cube
from cube.runtime.utils import microbatches
from typing import List

import inspect
from cube.graph.function.anchor import IRGraphAnchor
from cube.graph.function.dimops import IRDimops
from cube.graph.graph import IRGraph
from cube.ir.operator import IRDataOperation, IRFwOperation
import random

torch.set_printoptions(edgeitems = 2)

cube.init()


current_file_path = os.path.abspath(__file__)
current_folder = os.path.dirname(current_file_path)

text: str = "Huggingface is a really excellent project!"
cache_dir: str = "/mnt/msrasrg/yileiyang/hf_cache2"
model_name_list_path = os.path.join(current_folder, "models/Top_100_Tried")  # Natural Language Processing
LOAD_MODEL_TIMEOUT = False
error_out_cube_dict = {} # error_type: [model_name]
error_in_cube_dict = {}

loglevel = logging.INFO

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
            handler.setLevel(level)
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


info_path = os.path.join(log_dir, f'cube_compile_1_info.log')
logger = setup_logger(os.path.join(log_dir, info_path), loglevel)

tried_logger = setup_logger(os.path.join(log_dir, f'cube_compile_2_tried.log'), loglevel, False)
loaded_logger = setup_logger(os.path.join(log_dir, f'cube_compile_3_loaded.log'), loglevel, False)
traced_logger = setup_logger(os.path.join(log_dir, f'cube_compile_4_traced.log'), loglevel, False)
trace_aligned_logger = setup_logger(os.path.join(log_dir, f'cube_compile_5_trace_aligned.log'), loglevel, False)
compiled_logger = setup_logger(os.path.join(log_dir, f'cube_compile_6_compiled.log'), loglevel, False)
compile_aligned_logger = setup_logger(os.path.join(log_dir, f'cube_compile_7_compile_aligned.log'), loglevel, False)

logger_redirect(tried_logger, os.path.join(log_dir, info_path), prefix="model tried: ")
logger_redirect(loaded_logger, os.path.join(log_dir, info_path), prefix="model loaded: ")
logger_redirect(traced_logger, os.path.join(log_dir, info_path), prefix="model traced: ")
logger_redirect(trace_aligned_logger, os.path.join(log_dir, info_path), prefix="model trace aligned: ")
logger_redirect(compiled_logger, os.path.join(log_dir, info_path), prefix="model compiled: ")
logger_redirect(compile_aligned_logger, os.path.join(log_dir, info_path), prefix="model compile aligned: ")

error_out_cube_logger = setup_logger(os.path.join(log_dir, f'cube_compile_8_error_out_cube.log'), loglevel)
error_in_cube_logger = setup_logger(os.path.join(log_dir, f'cube_compile_9_error_in_cube.log'), loglevel)

logger_redirect(error_in_cube_logger, os.path.join(log_dir, info_path), "", False)
logger_redirect(error_out_cube_logger, os.path.join(log_dir, info_path), "", False)

for tmp_logger in logger_list:
    if torch.distributed.get_rank() == 0:
        tmp_logger.setLevel(logging.INFO)
    else:
        tmp_logger.setLevel(logging.WARNING)

def conditional_timeout_decorator(condition, timeout_time=30):
    def decorator(func):
        if condition:
            return timeout_decorator.timeout(timeout_time)(func)
        else:
            return func
    return decorator

@conditional_timeout_decorator(condition=LOAD_MODEL_TIMEOUT, timeout_time=30)
def load_model_from_config(config):  
    torch.manual_seed(0)
    return AutoModel.from_config(config, trust_remote_code=True)

@conditional_timeout_decorator(condition=LOAD_MODEL_TIMEOUT, timeout_time=120)
def load_model_from_pretrain(model_name, cache_dir, resume_download = True):
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir, resume_download = resume_download, trust_remote_code=True)


# from transformers import cached_path, WEIGHTS_NAME 
def load_hf_model(config, model_name, cache_dir, resume_download = True):
    try:
        model = load_model_from_config(config)
        return model
    except Exception:
        logger.info("load model from config failed, try by pretrain")
        # error_message = traceback.format_exc().strip() + "\n"
        # logger.error(error_message)
        try:
            model = load_model_from_pretrain(model_name, cache_dir=cache_dir, resume_download=resume_download)
            return model
        except Exception:
            logger.info("load model from pretrain failed, exit loading")
            # error_message = traceback.format_exc().strip() + "\n"
            # logger.error(error_message)
            raise

def load_hf_nlp_tokenizer(model_name, cache_dir):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, trust_remote_code=True)
        return tokenizer
    except OSError:
        # The script uses just one of the seven tokenizers below, as we're only checking if the logits match for the same input.
        # BertTokenizerFast, CamembertTokenizerFast tokenizer, XLMRobertaTokenizerFast tokenizer, DistilBertTokenizerFast tokenizer
        # T5TokenizerFast tokenizer, RobertaTokenizerFast tokenizer, GPT2TokenizerFast tokenizer
        logger.debug("loading pretrained tokenizer failed, use bert-base-uncased tokenizer instead")
        from transformers import BertTokenizerFast  
        return BertTokenizerFast.from_pretrained('bert-base-uncased', cache_dir=cache_dir, trust_remote_code=True)

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
            raise
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

def _prepare_nlp_input(model, dummy_input):
    if isinstance(dummy_input, MutableMapping):
        dummy_input = dict(dummy_input)
    assert isinstance(dummy_input, dict)
    forward_signature = inspect.signature(model.forward)
    if 'decoder_input_ids' in forward_signature.parameters and 'decoder_input_ids' not in dummy_input:
        dummy_input['decoder_input_ids'] = dummy_input.get('input_ids', None)
    return dummy_input

def prepare_dataloader(model, dummy_input):
    forward_signature = inspect.signature(model.forward)
    params_with_defaults = [
        v.default if k not in dummy_input else dummy_input[k].to(torch.cuda.current_device())
        for k, v in forward_signature.parameters.items()
    ]
    # logger.debug(f"forward_signature: {forward_signature}")
    # logger.debug(f"params_with_defaults: {params_with_defaults}")
    dataloader = microbatches((params_with_defaults, ) * 2)
    return dataloader

def trace_forward_check(model: torch.nn.Module, dummy_input: dict, before_trace: dict):
    traced_gm = concrete_trace_wrap(model, dummy_input)
    traced_logger.info(f"{model_name}")
    traced_gm.eval()
    after_trace = traced_gm(**dummy_input)
    # logger.debug(f"traced logit: {after_trace['last_hidden_state'][:2]}")

    if check_align(before_trace, after_trace):
        trace_aligned_logger.info(f"{model_name}")
    else:
        error_in_cube_logger.error(f"{model_name} not aligned before and after trace\n before trace: {before_trace}\n after trace: {after_trace}\n")

def compile_forward_check(model: torch.nn.Module, dummy_input: dict, before_trace: dict, policy = PASRandomSPMD):
    dataloader = prepare_dataloader(model, dummy_input)
    @cube.compile(model, dataloader, PAS=policy)
    def train_iter(model, dataloader):
        data = next(dataloader)
        logit = model(*data)
        return logit

    smodel = cube.utils.load_model()
    compiled_logger.info(f"{model_name}")
    smodel.eval()
    compiled_logit = train_iter(smodel, dataloader)

    if check_align(before_trace, compiled_logit):
        compile_aligned_logger.info(f"{model_name}")
    else:
        error_in_cube_logger.error(f"{model_name} not aligned before trace and after compile\n before trace: {before_trace}\n after trace: {compiled_logit}\n")

def compile_hf_nlp_worker(model_name: str, do_trace = True, do_compile = True):
    try:
        start_time = time.time()
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            subprocess.run('rm -rf gencode*.py fullmodel.pt.* dist_param_map.pt', shell=True, check=True)
        # load tokenizer, config, model
        tried_logger.info(f"{model_name}")

        config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
        logger.info(f"{model_name} config loaded")

        tokenizer = load_hf_nlp_tokenizer(model_name, cache_dir=cache_dir)
        logger.info(f"{model_name} Tokenizer loaded")

        model = load_hf_model(config, model_name, cache_dir)
        loaded_logger.info(f"{model_name}, {config.architectures if 'config' in locals() and config else None}")
        logger.info(f"{model_name} has parameter: {sum(p.numel() for p in model.parameters())}")
        print_memory_usage(logger, f"after load model {model_name}")

        # build dummy_input and forward
        dummy_input = tokenizer(text, return_tensors="pt")
        dummy_input = _prepare_nlp_input(model, dummy_input)
        model.eval()
        before_trace = model(**dummy_input)
        
        if do_trace and torch.distributed.get_rank() == 0:
            trace_forward_check(model, dummy_input, before_trace)
        
        if do_compile:
            model = load_hf_model(config, model_name, cache_dir)
            compile_forward_check(model, dummy_input, before_trace)

    except (TimeoutError, HTTPError, OSError, NameError, RuntimeError, KeyError) as e:
        if torch.distributed.get_rank() == 0:
            logger.error(f"fail when loading model: {model_name}", exc_info=False)

            error_message = traceback.format_exc().strip() + "\n"
            error_out_cube_logger.error(f"{model_name}, {config.architectures if 'config' in locals() and config else None}, failed")
            error_out_cube_logger.error(error_message)
            summarize_error(model_name, error_out_cube_dict, os.path.join(log_dir, "error_out_cube.json"))
    except Exception as e:  
        # Exception will be cause by cube.compile, or the program will be blocked
        if torch.distributed.get_rank() == 0:
            torch.distributed.barrier()
            
            logger.error(f"fail when compiling model: {model_name}", exc_info=False)
            error_message = traceback.format_exc().strip() + "\n"
            if 'MagicCube' in error_message:
                error_in_cube_logger.error(f"{model_name}, {config.architectures if 'config' in locals() and config else None}, failed")
                error_in_cube_logger.error(error_message)
                summarize_error(model_name, error_in_cube_dict, os.path.join(log_dir, "error_in_cube.json"))
            else:
                error_out_cube_logger.error(f"{model_name}, {config.architectures if 'config' in locals() and config else None}, failed")
                error_out_cube_logger.error(error_message)
                summarize_error(model_name, error_out_cube_dict, os.path.join(log_dir, "error_out_cube.json"))
    finally:
        end_time = time.time()
        logger.info(f"Finish trying model: {model_name}, time: {end_time - start_time:.2f} s")

def summarize_error(model_name, error_dict, log_path):
    import sys
    import json
    exc_type, exc_value, exc_traceback = sys.exc_info()
    first_line = f"{exc_type.__name__}: {exc_value}"
    first_line = first_line.replace(model_name, r"{model_name}")
    # print(f"first line is: {first_line}")
    
    # formatted_exception = traceback.format_exception(exc_type, exc_value, exc_traceback)
    # exception_string = ''.join(formatted_exception).strip() + "\n"
    # print(f"original error is:\n {exception_string}")

    if first_line in error_dict:
        error_dict[first_line]['model_name'].append(model_name)
        error_dict[first_line]['count'] += 1
    else:
        error_dict[first_line] = {"count": 1, 'model_name': [model_name]}   #, "example": exception_string
    
    error_dict = dict(sorted(error_dict.items(), key=lambda item: item[1]["count"], reverse=True))
    
    with open(log_path, 'w') as json_file:
        json.dump(error_dict, json_file, indent=4)

def _load_model_names(model_name_list_path, log_dir):
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
    print(f"# already_tried: {len(tried_models)}")
    model_name_list = [model for model in model_names if model not in tried_models]
    print(f"# need_to_try: {len(model_name_list)}")
    return model_name_list

def _load_error_summary(log_dir):
    import json
    error_out = {}
    error_in = {}
    if os.path.exists(os.path.join(log_dir, "error_out_cube.json")):
        with open(os.path.join(log_dir, "error_out_cube.json"), 'r') as json_file:
            error_out = json.load(json_file)
    if os.path.exists(os.path.join(log_dir, "error_in_cube.json")):
        with open(os.path.join(log_dir, "error_in_cube.json"), 'r') as json_file:
            error_in = json.load(json_file)
    return error_out, error_in

if __name__ == "__main__":
    model_name_list = _load_model_names(model_name_list_path, log_dir)
    error_out_cube_dict, error_in_cube_dict = _load_error_summary(log_dir)

    fxparser_warning_path = f'{log_dir}/FxModuleParser_Warning.log'
    with open(fxparser_warning_path, 'a') as file:
        total_models = len(model_name_list)
        for index, model_name in enumerate(model_name_list, 1):
            file.write(f"\n{model_name}\n")
            compile_hf_nlp_worker(model_name)
            logger.info(f"Process: {index} / {total_models}, Percentage: {(index / total_models) * 100:.2f}%\n\n")  


