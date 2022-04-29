# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
################################################

Testcase_PrepareCondition:

Testcase_TestSteps:

Testcase_ExpectedResult:

"""
import os, datetime, collections
import pytest
import numpy as np
from absl import flags
import tests.common.polyenv as environment
import tests.common.polygym as polygym
import akg.utils as utils
from akg.utils.result_analysis import count_unequal_element
from tests.common.base import TestBase
from tests.common.test_run.abs_run import abs_run
from tests.common.test_run.add_run import add_run
from tests.common.test_run import conv_run
from tests.common.test_run import batch_matmul_run

flags.DEFINE_string('out_dir', '', 'Root dir to store the results.')
flags.DEFINE_boolean('with_baselines', False, 'Benchmark baselines.')
flags.DEFINE_boolean('with_isl_tuning', False, 'Benchmark isl and tune.')
flags.DEFINE_boolean('with_polyenv', False, 'Benchmark polyenv random walk.')
flags.DEFINE_string('with_polyenv_sampling_bias', None, 'A sampling bias.')

flags.DEFINE_integer('stop_at', None, 'Number of OK samples to stop at.')

flags.DEFINE_string('with_action_import_sample_name', '', 'The sample name of the action sequence to import.')
flags.DEFINE_string('with_action_import_actions', '', 'The action sequence to import.')
FLAGS = flags.FLAGS

############################################################
# TestCase= class: put to tests/*/
############################################################
class TestCase(TestBase):
    def setup(self):
        case_name = "test_polygym"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.test_args = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            # ("000_abs_input_1_1", abs_run, ((1, 1), "float16"), ["level0"]),
            # ("001_abs_input_2_1", abs_run, ((2, 1), "float16"), ["level0"]),
            # ("002_abs_input_2_2_2", abs_run, ((2, 2, 2), "float16"), ["level0"]),
            # ("003_abs_input_1280_1280", abs_run, ((1280, 1280), "float16"), ["level0"]),
            # ("000_add", add_run, ((512, 1), (512, 1), 'float32'), ["level0"]),
            # ("001_add", add_run, ((1024, 2), (1024, 2), 'float32'), ["level0"]),
            # ("002_add", add_run, ((1024, 1024), (1024, 1024), 'float32'), ["level0"]),
            # ("000_conv", conv_run, ((32, 64, 56, 56), (64, 64, 3, 3), (1, 1), (1, 1, 1, 1), (1,1), "float32", "float32", "NCHW"), ["level0"]),
            # ("001_conv", conv_run, ((16, 4, 4, 16), (16, 3, 3, 16), (1, 1), (0, 0, 0, 0), (1, 1), "float16", "float16"), ["level0"]),
            # ("002_conv", conv_run, ((64, 6, 6, 64), (64, 3, 3, 64), (1, 1), (0, 0, 0, 0), (1, 1), "float16", "float32"), ["level0"]),
            ("003_conv", conv_run, ((64, 6, 6, 64), (64, 3, 3, 64), (1, 1), (0, 0, 0, 0), (1, 1), "float16", "float16"), ["level0"]),
            # ("000_batch_matmul", batch_matmul_run, ((128, 64), (128, 64), 'float32', 'float32', "NHDT", "NHDT", "NHDT",
            #     (1, ), False, False), ["level0"]),
            # ("001_batch_matmul", batch_matmul_run, ((32, 12, 128, 128), (32, 12, 128, 64), 'float16', 'float16', "NHDT",
            #     "NHTD", "NHDT", (1, ), False, True), ["level0"]),
            # ("002_batch_matmul", batch_matmul_run, ((256, 128), (64, 128), 'float16', 'float16', "NHDT", "NHDT", "NHDT",
            #     (1, ), False, True), ["level0"]),
            ("003_batch_matmul", batch_matmul_run, ((128, 32), (128, 512), 'float16', 'float16', "NHTD", "NHTD", "NHDT",
                (1, ), False, True), ["level0"]),
            # ("004_batch_matmul", batch_matmul_run, ((128, 64), (64, 32), 'float16', 'float16', "NHDT", "NHTD", "NHDT",
            #     (1, ), False, True), ["level0"]),
            # ("005_batch_matmul", batch_matmul_run, ((32, 1, 128, 64), (32, 1, 64, 32), 'float16', 'float16', "NHDT", "NHTD", "NHDT",
            #     (1, ), False, True), ["level0"]),
        ]
        return True

    def teardown(self):
        self._log.info("{0} Teardown".format(self.casename))
        super(TestCase, self).teardown()
        return

    def run_cases_polygym(self, cases, target, level="level0", profiling=False, repeat_times=1000):
        return self.run_test_arg_func_polygym(cases, level, target=target, profiling=profiling, repeat_times=repeat_times)

    def run_test_arg_func_polygym(self, test_args=[], attr=None, target=utils.CCE, profiling=False, repeat_times=1000):
        if not attr:
            self._log.info("attr is None")
            return False
        self.set_target(target)
        self.set_profiling(profiling)
        self.set_repeat_times(repeat_times)
        run_mode = self.get_env_var("RUNTIME_MODE")
        if run_mode in ["compile_cloud", "compile_mini"]:
            mode = "compile"
        else:
            mode = "execute"
        for arg in test_args:
            self._log.info(arg)
            if attr in arg[-1]:
                case_result, _ = self.common_run_polygym([arg[0:-1]], mode=mode)
                if not case_result:
                    self._log.info("{0} run failed".format(arg))
                    return False
        return True

    def common_run_polygym(self, args_list, dtype_list=None, mode="execute", is_conv=False, raise_exception=True):
        """
        :param dtype_list:operator program data type
        :param mode: operator run mode: such as rpc_cloud/aicmodel
        :param raise_exception: By default, when an exception occurs in the compilation,
                                the assert is used to interrupt the program.
        :return:
        """
        for arg in args_list:
            starttime = datetime.datetime.now()
            caseflag, func, args, kwargs = self.ana_args(arg, is_conv)

            if dtype_list:
                if not self.set_args_dtype(args, func, dtype_list):
                    self._log.error("common_run failed for set_args_dtype")
                    return False

            if isinstance(func, str):
                self._log.info("common_run :: run {funcname} with args:{args}".format(funcname=func, args=args))
                func = self.import_get_func(func, mode)
            else:
                self._log.info("common_run :: run {funcname} with args:{args}".format(funcname=func.__name__, args=args))

            mod = None
            if mode == "compile":
                try:
                    mod = func(*args, **kwargs)
                except Exception as e:
                    TestBase.pandora_logger_.traceback()
                    self._exception = e
                finally:
                    if (not mod) or self._exception:
                        self._log.error("common_run :: circle {0} fail !".format(self.translate_func_name(arg)))
                        self._log.error("common_run :: compile failed !")
                        self.case_result = False

            elif mode == "execute":
                input, output, expect, runres, cycles = func(*args, **kwargs)
                kernel_name = None
                with open("kernel_name", "r") as file:
                    kernel_name = file.readline().strip()
                scop_file = os.path.join(os.curdir, kernel_name + '_0', "poly", "%entry.split---%for.end40.jscop")
                env = environment.PolyEnv()
                info = gen_and_bench_random_schedule(env, scop_file, kernel_name, "bias_select_dep")
                compute_schedule_file = os.path.join(os.curdir, kernel_name + '_0', "poly", "ComputeSchedule.txt")
                env._get_current_schedule(compute_schedule_file)
                env.sample['isl_execution_time'] = cycles
                input, output, expect, runres, cycles = func(*args, **kwargs)
                reward = env._calc_speedup(cycles)
                # speedup = env.reward_to_speedup(reward)
                # exec_time = env.speedup_to_execution_time(speedup)
                # status = info['status']
                # ast = info['ast'] if 'ast' in info else None
                # isl_map = info['isl_map'] if 'isl_map' in info else None

                rtol = atol = 0
                compare_res = []
                if isinstance(runres, list):
                    if isinstance(runres[-1], (list, tuple)):
                        rtol = runres[-1][0]
                        atol = runres[-1][1]
                        runres = list(runres[:-1])
                    compare_res = runres
                    runres = all(runres)
                elif isinstance(runres, collections.Iterable):
                    compare_res = list(runres)
                else:
                    compare_res = [runres]

                if not runres:
                    runtime_mode = os.environ.get("RUNTIME_MODE")
                    if runtime_mode in ["rpc", "rpc_cloud", "air", "air_cloud"]:
                        for retry in range(self.max_retry):
                            self._log.error("Case result is incorrect, but RPC server occasionally produce incorrect "
                                            "output. Retry it before reporting failure. Retry count: " + str(retry + 1))
                            input, output, expect, runres, _ = func(*args, **kwargs)

                            if isinstance(runres, list):
                                if isinstance(runres[-1], (list, tuple)):
                                    rtol = runres[-1][0]
                                    atol = runres[-1][1]
                                    runres = list(runres[:-1])
                                compare_res = runres
                                runres = all(runres)
                            elif isinstance(runres, collections.Iterable):
                                compare_res = list(runres)
                            else:
                                compare_res = [runres]

                            if runres:
                                break
                            else:
                                self.data_dump(input, output, arg, retry)
                                if rtol == 0:
                                    rtol = atol = 1e-4
                                    for datas in [input, output]:
                                        for data in list(datas):
                                            if isinstance(data, np.ndarray) and data.dtype == "float16":
                                                rtol, atol = get_rtol_atol("", "float16")
                                                break
                                    self._log.error("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                                    self._log.error(
                                        "Caution: the 'rtol' and 'atol' is default $$$$$(%s, %s)$$$$$" % (rtol, atol))
                                    self._log.error("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

                                if isinstance(expect, (tuple, list)):
                                    for i, tmp in enumerate(expect):
                                        count_unequal_element(tmp, output[i], rtol, atol)
                                else:
                                    if not isinstance(expect, np.ndarray):
                                        expect = np.atleast_1d(expect)
                                    count_unequal_element(expect, output, rtol, atol)
                if not runres:
                    self._log.error("common_run :: circle {0} fail !".format(self.translate_func_name(arg)))
                    self._log.error("common_run :: CompareResult: %s", str(compare_res))
                    self.case_result = False
                else:
                    self._log.info("common_run :: circle {0} pass !".format(self.translate_func_name(arg)))
                    self.case_result &= True

            endtime = datetime.datetime.now()
            self._log.info("{0} testcase use ***Running Time*** is: {1}s. "
                           .format(caseflag, (endtime - starttime).seconds))
        self._log.info(self.case_result)
        '''
        use assert in the common_run function:
        Because the common_run function in the use cases does not verify the return value, the result cannot be 
        printed normally after the program ends, so the execution result needs to be judged in the common_run function.
        '''
        if (not self.case_result) and raise_exception:
            assert self.case_result
        return self.case_result, self._exception

    @pytest.mark.level0
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_gpu_level0(self):
        return self.run_cases_polygym(self.test_args, utils.CUDA, "level0")

def gen_and_bench_random_schedule(env, scop_file, sample_name, sampling_bias=None, predef_actions=None):
    with_ast_and_map = True if sample_name in ['gemm', 'matvect'] else False
    state = env.reset(scop_file, sample_name, with_ast_and_map=with_ast_and_map)

    actions = []
    done = False
    reward = None

    exec_time = None

    try:
        if predef_actions:
            predef_actions_idx = 0
        while not done:
            if predef_actions:
                action_idx = predef_actions[predef_actions_idx]
                predef_actions_idx += 1
            elif sampling_bias:
                mask = state['action_mask']
                possibilities = mask * range(len(mask))
                if sampling_bias == 'bias_coeff0':
                    p = mask * [1, 1, 1, 1, 0.15, 0.15]
                elif sampling_bias == 'bias_select_dep':
                    p = mask * [0.2, 0.2, 0.6, 1, 1, 1]
                else:
                    raise Exception
                p /= p.sum()        # Normalize
                action_idx = int(np.random.choice(possibilities, p=p))
            else:
                action_idx = np.random.choice(np.nonzero(state['action_mask'])[0])
            action = list(environment.Action)[action_idx]
            actions.append(action_idx)

            state, _, done, info = env.step(action)

    except (polygym.ChernikovaTimeoutException) as e:
        status = e.__class__.__name__

    return info

def get_rtol_atol(op_name, dtype, rtol=5e-03, atol=5e-03):
    from akg.utils import composite_op_helper as helper
    return helper.get_rtol_atol(op_name, dtype, rtol, atol)

def get_splitted_cases(cases, split_nums, split_idx):
    if not isinstance(cases, (list, tuple)):
        raise TypeError("Argument cases must be of type list or tuple.")
    if not isinstance(split_nums, int) or not isinstance(split_idx, int):
        raise TypeError("Arguments split_nums and split_idx must be of type int.")
    if split_nums <= 0 or split_idx < 0 or split_idx >= split_nums:
        raise ValueError("Argument split_nums must > 0, split_idx must be in range [0, split_nums)")

    cases = list(cases)
    all_cases = len(cases)
    fragment = (all_cases + split_nums - 1) // split_nums

    start_idx = split_idx * fragment
    if start_idx >= all_cases:
        return []

    end_idx = start_idx + fragment
    if end_idx > all_cases:
        end_idx = all_cases

    return cases[start_idx:end_idx]
