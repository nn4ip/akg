/**
 * Copyright 2019 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "poly/dump_log.h"

#include <unistd.h>
#include <libgen.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <fstream>
#include <iostream>
#include <iomanip>

#include "poly/poly_util.h"
#include "poly/dma_inject.h"

namespace akg {
namespace ir {
namespace poly {
#if (!PRETTY_PRINT_IR)
// dump schedule tree to file
void DumpSchTreeToFile(std::FILE *fp, const isl::schedule &sch) {
  isl_printer *printer = nullptr;

  CHECK(sch.get());

  printer = isl_printer_to_file(isl_schedule_get_ctx(sch.get()), fp);
  printer = isl_printer_set_yaml_style(printer, ISL_YAML_STYLE_BLOCK);
  printer = isl_printer_print_schedule(printer, sch.get());

  static_cast<void>(isl_printer_free(printer));
}
#endif

// dump schedule tree to file
void DumpRawSchTreeToFile(std::FILE *fp, const isl::schedule &sch) {
  isl_printer *printer = nullptr;

  CHECK(sch.get());

  printer = isl_printer_to_file(isl_schedule_get_ctx(sch.get()), fp);
  printer = isl_printer_print_schedule(printer, sch.get());

  static_cast<void>(isl_printer_free(printer));
}

// dump schedule tree to string
std::string DumpSchTreeToString(const isl::schedule &sch) {
  isl_printer *printer = nullptr;

  CHECK(sch.get());

  printer = isl_printer_to_str(sch.ctx().get());
  CHECK(printer);
  printer = isl_printer_set_yaml_style(printer, ISL_YAML_STYLE_BLOCK);
  printer = isl_printer_print_schedule(printer, sch.get());
  const char *s = isl_printer_get_str(printer);
  static_cast<void>(isl_printer_free(printer));

  std::string str(s);
  std::free(reinterpret_cast<void *>(const_cast<char *>(s)));
  return str;
}

/*
 * Type 1: "{ a; b; c }" format to "{ a;" "b;" "c }"
 * Type 2: "[ a, b, c ]" format to "[ a," "b," "c ]"
 */
std::string FormatMupaStr(const std::string &mupa_str, bool checkInString) {
  const char *src = mupa_str.c_str();
  CHECK(src != nullptr);
  std::stringstream dst;
  const int max_bracket_depth = 2;
  const int domain_bracket_min_depth = 2;
  std::vector<char> bracket_stack;
  int indent_spaces[max_bracket_depth + 1] = {0};
  int col_pos = 0;
  int bracket_depth = 0;
  bool in_string = false;

  while (*src != '\0') {
    if (*src == '"') {
      in_string = !in_string;
      bracket_depth = 0;
      indent_spaces[0] = col_pos;
    } else if (*src == '\n' || *src == '\r') {
      col_pos = -1;
    } else if (*src == '\t') {
      const int tab_width = 2;
      col_pos += tab_width;
    } else if (in_string || !checkInString) {
      char c = *src;
      if (c == '{' || c == '[') {
        bracket_depth++;
        bracket_stack.push_back(c);
        if (bracket_depth <= max_bracket_depth) {
          indent_spaces[bracket_depth] = col_pos;
          // find the first non white-space char after the bracket
          const char *t = src + 1;
          while (*t == ' ') {
            t++;
            indent_spaces[bracket_depth]++;
          }
        }
      } else if (c == '}' || c == ']') {
        bracket_depth--;
        bracket_stack.pop_back();
      } else if ((c == ',' || c == ';') && bracket_depth <= max_bracket_depth) {
        bool not_inside_domain =
          (bracket_depth >= domain_bracket_min_depth && (bracket_stack[0] != '{' || bracket_stack[1] != '['));
        if (bracket_depth < domain_bracket_min_depth || not_inside_domain) {
          dst << c << (in_string ? '"' : ' ') << '\n';
          for (int i = 0; i < indent_spaces[bracket_depth]; i++) {
            dst << " ";
          }
          dst << (in_string ? '"' : ' ');
          col_pos = indent_spaces[bracket_depth] + 1;

          src++;
          // remove immediate spaces after newline string
          while (*src == ' ') {
            src++;
          }
          continue;
        }
      }
    }
    col_pos++;
    dst << *src++;
  }
  return dst.str();
}

std::string FormatMupaStr(const isl::union_map &map) { return FormatMupaStr(map.to_str()); }

std::string FormatMupaStr(const isl::union_set &set) { return FormatMupaStr(set.to_str()); }

std::string FormatMupaStr(const isl::multi_aff &aff) { return FormatMupaStr(aff.to_str()); }

std::string FormatMupaStr(const isl::multi_pw_aff &mpa) { return FormatMupaStr(mpa.to_str()); }

std::string FormatMupaStr(const isl::multi_union_pw_aff &mupa) { return FormatMupaStr(mupa.to_str()); }

std::string FormatMupaStr(const isl::union_pw_aff &upa) { return FormatMupaStr(upa.to_str()); }

std::string FormatSchTreeStr(const std::string &sch_tree_str) { return FormatMupaStr(sch_tree_str, true); }

void PrettyPrintSchTree(std::FILE *fp, const isl::schedule &sch) {
  std::string sch_tree_str = DumpSchTreeToString(sch);
  std::string pretty_str = FormatSchTreeStr(sch_tree_str);
  if (fwrite(pretty_str.c_str(), 1, pretty_str.size(), fp) != pretty_str.size()) {
    LOG(WARNING) << "failed to write schedule tree to file";
  }
}

std::string PrettyPrintSchTree(const isl::schedule &sch) {
  std::string sch_tree_str = DumpSchTreeToString(sch);
  return FormatSchTreeStr(sch_tree_str);
}

/*
 * Check that file name is a simple relative path (does not start with "/", and does not include "." or "..").
 * FileName should not include extension, and the extension will be appended to FileName.
 */
std::string FilePathCanonicalize(const std::string &file_name, bool is_log) {
  CHECK(!file_name.empty()) << "file name must not be empty";
  CHECK(file_name.c_str()[0] != '/') << "file name must not be an absolute path, found " << file_name;
  CHECK(file_name.find('.') == std::string::npos)
    << "To avoid attacks, file name cannot include '.' character: " << file_name;
  if (!is_log) {
    return file_name + ".cc";
  } else {
    return file_name + ".log";
  }
}

bool CreateFileIfNotExist(const std::string &file_name) {
  if (access(file_name.c_str(), F_OK) == -1) {
    int fd = creat(file_name.c_str(), S_IRUSR | S_IWUSR);
    if (fd == -1) {
      LOG(WARNING) << "failed to create dumpfile " << file_name;
      return false;
    }
    int ret = close(fd);
    if (ret != 0) {
      LOG(WARNING) << "failed to close dumpfile" << file_name;
      return false;
    }
  }
  return true;
}

// dump schedule tree to file
void DumpSchTreeImpl(const std::string &file_name, const isl::schedule &sch) {
#if DUMP_IR
  std::string canonical_file_name = FilePathCanonicalize(file_name, false);
  if (!CreateFileIfNotExist(canonical_file_name)) return;
  FILE *fp = fopen(canonical_file_name.c_str(), "w");
  if (fp != nullptr) {
#if PRETTY_PRINT_IR
    PrettyPrintSchTree(fp, sch);
#else
    DumpSchTreeToFile(fp, sch);
#endif
    int status = fclose(fp);
    if (status != 0) LOG(WARNING) << "Failed to close dump schedule tree file " << canonical_file_name;
  } else {
    LOG(WARNING) << "Failed to open dump schedule tree file " << canonical_file_name;
  }
#endif
}

// dump raw schedule tree to file
void DumpRawSchTreeImpl(const std::string &file_name, const isl::schedule &sch) {
#if DUMP_IR
  std::string canonical_file_name = FilePathCanonicalize(file_name, false);
  if (!CreateFileIfNotExist(canonical_file_name)) return;
  FILE *fp = fopen(canonical_file_name.c_str(), "w");
  if (fp != nullptr) {
    DumpRawSchTreeToFile(fp, sch);
    int status = fclose(fp);
    if (status != 0) LOG(WARNING) << "Failed to close dump schedule tree file " << canonical_file_name;
  } else {
    LOG(WARNING) << "Failed to open dump schedule tree file " << canonical_file_name;
  }
#endif
}

static bool IsSpaceOrDoubleQuote(char c) { return isspace(c) || c == '"'; }

bool CompareSchTreeWithString(const std::string &compare_sch_, const isl::schedule &sch) {
  std::string sch_tree_str = DumpSchTreeToString(sch);
  sch_tree_str.erase(remove_if(sch_tree_str.begin(), sch_tree_str.end(), IsSpaceOrDoubleQuote), sch_tree_str.end());

  auto compare_sch = compare_sch_;
  compare_sch.erase(remove_if(compare_sch.begin(), compare_sch.end(), IsSpaceOrDoubleQuote), compare_sch.end());
  return (sch_tree_str == compare_sch);
}

void PrintHeader(std::ofstream &of, const std::string &str) {
  of << std::endl << ">>>>>>>>>> " << str << " <<<<<<<<<<" << std::endl;
}
void PrintHeader(const std::string &str) { std::cout << ">>>>>>>>>> " << str << " <<<<<<<<<<" << std::endl; }

void DumpNode(std::ofstream &of, const air::Node *node) {
  if (node->IsInstance<Provide>()) {
    auto op = static_cast<const Provide *>(node);
    of << Provide::make(op->func, op->value_index, op->value, op->args);
  } else if (node->IsInstance<IfThenElse>()) {
    auto op = static_cast<const IfThenElse *>(node);
    of << IfThenElse::make(op->condition, op->then_case, op->else_case);
  } else if (node->IsInstance<For>()) {
    auto op = static_cast<const For *>(node);
    of << For::make(op->loop_var, op->min, op->extent, op->for_type, op->device_api, op->body);
  } else if (node->IsInstance<Block>()) {
    auto op = static_cast<const Block *>(node);
    of << Block::make(op->first, op->rest);
  } else if (node->IsInstance<Call>()) {
    auto op = static_cast<const Call *>(node);
    of << Call::make(op->type, op->name, op->args, op->call_type, op->func, op->value_index);
  } else {
    of << "unknown node type " << node->GetTypeKey() << ", addr " << node;
  }
}

void CreateDirIfNotExist(const std::string &file_name) {
  char *file_name_ = strdup(file_name.c_str());
  CHECK(file_name_ != nullptr);
  // dirname() updates "filename" in place, so "dir" is a substring of "filename".
  // Do not free "dir", and "filename" should be freed after both "dir" and "filename" are no longer used.
  char *dir = dirname(file_name_);
  if (strcmp(dir, file_name.c_str()) == 0) {
    LOG(WARNING) << "Cannot create root directory " << file_name;
    free(file_name_);
    return;
  }
  struct stat info;
  if (stat(dir, &info) == 0) {
    if (!(info.st_mode & S_IFDIR)) {
      LOG(WARNING) << "Directory " << std::string(dir) << " already exists but it is not a directory";
    }
    free(file_name_);
    return;
  }
  const int dir_mode = S_IRUSR | S_IWUSR | S_IXUSR;
  if (mkdir(dir, dir_mode) != 0) {
    char *dir_copy = strdup(dir);
    CHECK(dir_copy != nullptr);
    char *parent_dir = dirname(dir_copy);
    CHECK(parent_dir != nullptr);
    CreateDirIfNotExist(parent_dir);
    free(dir_copy);
    if (mkdir(dir, dir_mode) != 0) {
      LOG(WARNING) << "Failed to create directory " << std::string(dir);
    }
  }
  free(file_name_);
}

void AnalysisResult::DumpScopDataBasics(std::ofstream &of) {
  PrintHeader(of, "statements");
  for (const auto &stmt : GetStatementMap()) {
    of << stmt.first << " : ";
    DumpNode(of, stmt.second);
    of << std::endl;
  }

  PrintHeader(of, "accesses");
  for (const auto &stmt : GetAccessMap()) {
    of << stmt.second << " : ";
    DumpNode(of, stmt.first);
    of << std::endl;
  }

  PrintHeader(of, "domains");
  for (const auto &stmt : GetOperatorDomainMap()) {
    of << stmt.first << " : param_space " << stmt.second.param_space << std::endl;
  }

  PrintHeader(of, "stmt_op_Info");
  for (const auto &stmt : GetStmtOpInfoMap()) {
    of << stmt.first << " : ops [ ";
    for (auto op : stmt.second.ops) {
      of << int(op) << ", ";
    }
    of << "] readtensors [ ";
    for (const auto &id : stmt.second.readtensors) {
      of << id << ", ";
    }
    of << "]" << std::endl;
  }

  PrintHeader(of, "reads");
  of << FormatMupaStr(GetReads()) << std::endl;

  PrintHeader(of, "writes");
  of << FormatMupaStr(GetWrites()) << std::endl;

  PrintHeader(of, "copyin");
  of << FormatMupaStr(GetCopyin()) << std::endl;

  PrintHeader(of, "fake_copyin");
  of << FormatMupaStr(GetFakeCopyin()) << std::endl;

  PrintHeader(of, "inter_band_dependency");
  of << FormatMupaStr(GetInnerBandDependency()) << std::endl;

  PrintHeader(of, "transfer_stmt");
  of << FormatMupaStr(GetTransferStmt()) << std::endl;

  PrintHeader(of, "reduce_stmts");
  for (const auto &stmt : GetReduceTensorInfoMap()) {
    of << stmt.first << ": reduce axis [ ";
    for (const auto &axis : stmt.second.axis_vec) {
      of << axis << " ";
    }
    of << "]" << std::endl;
  }

  PrintHeader(of, "reduce_out_tensors");
  for (const auto &item : GetReduceOutTensors()) {
    of << item << std::endl;
  }
}

void ScopInfo::DumpScopDataAdvanced(std::ofstream &of) {
  PrintHeader(of, "binds");
  auto binds = user_config_.GetBind();
  for (auto bind : binds) {
    of << bind.first << " : " << bind.second << std::endl;
  }

  PrintHeader(of, "binds_orig");
  auto binds_orig = user_config_.GetOriginBind();
  for (auto bind : binds_orig) {
    of << bind.first << " : " << bind.second << std::endl;
  }

  PrintHeader(of, "realize_from_input");
  auto realize_from_input = user_config_.GetRealizeFromInput();
  for (const auto &id : realize_from_input) {
    of << id << ", ";
  }
  of << std::endl;

  PrintHeader(of, "dim_infos");
  for (const auto &dim_info : analysis_result_.GetTileSizes()) {
    of << "index=" << dim_info.index << " axis=" << dim_info.axis << " c1_tiling_size=" << dim_info.c1_tiling_size
       << " c0_tiling_size=" << dim_info.c0_tiling_size << " dim_seq=" << dim_info.dim_seq << std::endl;
  }

  PrintHeader(of, "fractal_int_info");
  for (const auto &info : mmu_info_.fractal_int_info_) {
    of << info.first << " : " << info.second << std::endl;
  }

  PrintHeader(of, "fractal_str_info");
  for (const auto &info : mmu_info_.fractal_str_info_) {
    of << info.first << " : " << info.second << std::endl;
  }

  PrintHeader(of, "conditional_write_buffer_footprints");
  auto conditional_write_buffer_footprints = analysis_result_.GetConditionalWriteBufferFootprints();
  for (const auto &tensor : conditional_write_buffer_footprints) {
    of << tensor << std::endl;
  }

  PrintHeader(of, "tensor_name_flows");
  auto tensor_name_flows = analysis_result_.GetTensorNameFlows();
  for (const auto &name_flow : tensor_name_flows) {
    of << name_flow.first << " : [ ";
    for (const auto &name : name_flow.second) {
      of << name << ", ";
    }
    of << "]" << std::endl;
  }

  PrintHeader(of, "tensor_memflows");
  auto tensor_mem_flows = analysis_result_.GetTensorMemFlows();
  for (const auto &mem_flow : tensor_mem_flows) {
    of << mem_flow.first << " : [ ";
    for (auto mem : mem_flow.second) {
      of << static_cast<int>(mem) << ", ";
    }
    of << "]" << std::endl;
  }

  PrintHeader(of, "active_buffer_footprints");
  for (const auto &active_buffer_footprint : analysis_result_.active_buffer_footprints_) {
    bool empty_cluster = (active_buffer_footprint.second.cluster.get() == nullptr);
    if (!empty_cluster) {
      of << "cluster_id : " << active_buffer_footprint.second.cluster_id << std::endl
         << "domain : " << FormatMupaStr(active_buffer_footprint.first) << std::endl
         << "cluster : " << *(active_buffer_footprint.second.cluster) << std::endl
         << "outer_schedule : " << FormatMupaStr(active_buffer_footprint.second.outer_schedule) << std::endl
         << std::endl;
    } else {
      of << "cluster_id : " << active_buffer_footprint.second.cluster_id << std::endl
         << "domain : " << FormatMupaStr(active_buffer_footprint.first) << std::endl
         << "cluster : null " << std::endl
         << "outer_schedule : " << FormatMupaStr(active_buffer_footprint.second.outer_schedule) << std::endl
         << std::endl;
    }
  }

  PrintHeader(of, "buffered_decl_infos");
  analysis_result_.DumpBufferDefInfos(of);
  of << std::endl;

  PrintHeader(of, "attr_info");
  for (const auto &info : mmu_info_.GetConvAttrInfo()) {
    of << info.first << " : " << info.second << std::endl;
  }
}

void UserConfig::DumpScopDataScheduleAttrs(std::ofstream &of) {
  PrintHeader(of, "MindTrick");
  of << "mind trick status : " << mind_trick_status_ << std::endl;

  PrintHeader(of, "schedule attrs");
  of << "dump_poly_dir : " << GetDumpPolyDir() << std::endl;

  of << "dump_tuning_level : " << GetDumpTuningLevel() << std::endl;
  of << "dim : " << GetBDim() << std::endl;

  of << "pragma_rmselfdep : " << GetRemoveSelfDependence() << std::endl;
  of << "pragma_force_rmselfdep : " << GetForceRemoveSelfDependence() << std::endl;
  of << "pragma_disable_whole_component: " << GetDisableWholeComponent() << std::endl;
  of << "pragma_disable_schedule_shift : " << GetDisableScheduleShift() << std::endl;
  of << "pragma_enable_schedule_max_constant : " << GetEnableScheduleMaxConstant() << std::endl;
  of << "pragma_disable_loop_reversal : " << GetDisableLoopReversal() << std::endl;
  of << "pragma_disable_loop_fusion : " << GetDisableLoopFusion() << std::endl;
  of << "pragma_modshift : " << GetModScheduleShift() << std::endl;
  of << "pragma_reorder_schedule : " << GetReorderSchedule() << std::endl;
  of << "pragma_checkcoincident : " << GetTileCheckCoincident() << std::endl;
  of << "pragma_opt_for_dsa : " << GetOptimizeForNPU() << std::endl;
  of << "pragma_sink_last_axis : " << GetSinkLastAxis() << std::endl;
  of << "pragma_keep_outer_band_order : " << GetKeepOuterBandOrder() << std::endl;
  of << "pragma_disable_group : " << GetDisableGroup() << std::endl;
  of << "pragma_tile_inner_band : " << GetTileInnerBand() << std::endl;
  of << "isolated_idx : " << GetIsolatedIdx() << std::endl;
  of << "pragma_outerband_need_split : " << GetOuterBandNeedSplit() << std::endl;

  of << "dynamic_shape_bound : " << GetDynamicShapeBound() << std::endl;
  of << "pragma_tilesize_is_var : " << GetTileSizeIsVar() << std::endl;

  of << "kernel_name : " << GetKernelName() << std::endl;
  of << "kernel_h : " << GetMatBDimH() << std::endl;
  of << "kernel_w : " << GetMatBDimW() << std::endl;
  of << "conv_backprop_filter : " << GetConvBackPropFilter() << std::endl;
  of << "bypassL1 : " << GetByPathC1() << std::endl;
  of << "pragma_is_conv : " << GetPragmaIsConv() << std::endl;
  of << "pragma_conv_special_dma : " << GetConvSpecialDma() << std::endl;
}

bool ScopInfo::DumpScopData(const std::string &file_name) {
  std::string canonical_log_name = FilePathCanonicalize(file_name, true);
  if (!CreateFileIfNotExist(canonical_log_name)) return false;
  std::ofstream of;
  of.open(canonical_log_name, std::ios::out);
  if (!of.is_open()) return false;

  analysis_result_.DumpScopDataBasics(of);

  DumpScopDataAdvanced(of);

  user_config_.DumpScopDataScheduleAttrs(of);

  of.close();
  return true;
}

bool ScopInfo::DumpJscopData(const std::string &file_name, const isl::schedule &sch) {
  std::string canonical_log_name = file_name + ".jscop";
  if (!CreateFileIfNotExist(canonical_log_name)) return false;
  std::ofstream of;
  of.open(canonical_log_name, std::ios::out);
  if (!of.is_open()) return false;

  /* some globally used data */
  std::set<std::string> params;
  std::string params_str;

  struct Stmt {
    std::string name;
    std::unordered_map<std::string, int> param_upper, param_lower;
    isl::map_list reads, writes;
    std::vector<int> ris, wis;

    /* extract (indices of) read and write accesses of this statement into ris and wis */
    void init_ris() {
      for (unsigned i = 0; i < reads.size(); ++i) {
        if (reads.at(i).space().domain().to_str().find(name) != std::string::npos) {
          ris.push_back(i);
        }
      }
    }
    void init_wis() {
      for (unsigned i = 0; i < writes.size(); ++i) {
        if (writes.at(i).space().domain().to_str().find(name) != std::string::npos) {
          wis.push_back(i);
        }
      }
    }

    /* iterate over reads or writes and extract necessary information */
    void iterate_accesses(bool is_writes) {
      for (const int & i : (is_writes ? wis : ris)) {
        std::string domain = (is_writes ? writes : reads).at(i).domain().to_str();
        domain = domain.substr(domain.find(":") + 1);
        domain.pop_back();
        std::size_t p;
        while ((p = domain.find("and")) != std::string::npos) {
          domain[p] = ',';
          domain[p + 1] = ' ';
          domain[p + 2] = ' ';
        }
        std::string domain_trimmed;
        for (const char & c : domain) {
          if (c != ' ') {
            domain_trimmed.push_back(c);
          }
        }
        std::istringstream ss(domain_trimmed);
        std::string bound;
        while (std::getline(ss, bound, ',')) {
          std::string param;
          for (p = 0; bound[p] != '='; ++p);
          for (p = p + 1; bound[p] != '<'; ++p) {
            param.push_back(bound[p]);
          }
          int lower = std::stoi(bound.substr(0, bound.find_first_of("<")));
          int upper = std::stoi(bound.substr(bound.find_last_of("=") + 1));
          if (param_lower.find(param) == param_lower.end()) {
            param_lower.insert({param, lower});
          } else if (param_lower[param] > lower) {
            param_lower[param] = lower;
          }
          if (param_upper.find(param) == param_upper.end()) {
            param_upper.insert({param, upper});
          } else if (param_upper[param] < upper) {
            param_upper[param] = upper;
          }
        }
      }
    }
  };
  std::vector<Stmt> stmts;

  /* preprocessing phase: iterate over each statement and collect information */
  {
    /* collect names of params */
    for (const auto &stmt : analysis_result_.GetOperatorDomainMap()) {
      std::string raw_str = stmt.second.param_space.to_str();
      std::size_t p = raw_str.find_first_of("["), q = raw_str.find_first_of("]");
      raw_str = raw_str.substr(p + 1, q - p - 1);
      std::istringstream ss(raw_str);
      std::string param;
      while (std::getline(ss, param, ',')) {
        if (param[0] == ' ') {
          param = param.substr(1);
        }
        params.insert(param);
      }
    }

    /* pre-calculate params_str */
    // params_str.push_back('[');
    // for (auto it = params.begin(); it != params.end(); ++it) {
    //   if (it != params.begin()) {
    //     params_str = params_str + ", ";
    //   }
    //   params_str = params_str + *it + "_p";
    // }
    // params_str.push_back(']');

    /* no params */
    params_str = "[]";

    /* pre-calculate data for each statement */
    for (const auto &stmt : analysis_result_.GetStatementMap()) {
      Stmt st;
      st.name = stmt.first.to_str();
      st.reads = analysis_result_.GetReads().map_list();
      st.writes = analysis_result_.GetWrites().map_list();
      st.init_ris();
      st.init_wis();
      st.iterate_accesses(false);
      st.iterate_accesses(true);
      stmts.emplace_back(std::move(st));
    }
  }

  of << "{" << std::endl;

  /* context */
  {
    of << "\t" << "\"context\" : \"" << params_str << " -> {  : ";
    
    // no params
    // for (auto it = params.begin(); it != params.end(); it++) {
    //   if (it != params.begin()) {
    //     of << " and ";
    //   }
    //   of << "-2147483648 <= " << *it + "_p" << " <= 2147483647";
    // }

    of << " }\"," << std::endl;
  }
  
  /* name */
  {
    of << "\t" << "\"name\" : \"%entry.split---%for.end40\"," << std::endl;
  }
  
  /* statements */
  {
    of << "\t" << "\"statements\" : [" << std::endl;

    for (auto st = stmts.begin(); st != stmts.end(); ++st) {
      
      /* print head formatting */
      if (st != stmts.begin()) {
        of << "," << std::endl;
      }
      of << "\t\t" << "{" << std::endl;

      /* accesses */
      {
        of << "\t\t\t" << "\"accesses\" : [" << std::endl;

        int total_rws = st->ris.size() + st->wis.size();

        /* print read accesses in specified format */
        for (const int &i : st->ris) {
          std::string read = st->reads.at(i).flatten().to_str();
          read = read.substr(2, read.find(":") - 3);
          read = st->name + read;
          std::size_t p, q;
          while ((q = read.find("=")) != std::string::npos) {
            q++;
            p = q;
            while (read[p - 1] != ',' && read[p - 1] != '[') {
              p--;
            }
            if (read[p - 1] == '[') {
              q++;
            }
            read.erase(p, q - p); // erase [p, q)
          }
          read = params_str + " -> { " + read + " }";

          of << "\t\t\t\t" << "{" << std::endl;
          of << "\t\t\t\t\t" << "\"kind\" : \"read\"," << std::endl;
          of << "\t\t\t\t\t" << "\"relation\" : \"" << read << "\"" << std::endl;
          of << "\t\t\t\t" << (i < total_rws - 1 ? "}," : "}") << std::endl;
        }

        /* print write accesses in specified format */
        for (const int &i : st->wis) {
          std::string write = st->writes.at(i).flatten().to_str();
          write = write.substr(2, write.find(":") - 3);
          write = st->name + write;
          std::size_t p, q;
          while ((q = write.find("=")) != std::string::npos) {
            q++;
            p = q;
            while (write[p - 1] != ',' && write[p - 1] != '[') {
              p--;
            }
            if (write[p - 1] == '[') {
              q++;
            }
            write.erase(p, q - p); // erase [p, q)
          }
          write = params_str + " -> { " + write + " }";

          of << "\t\t\t\t" << "{" << std::endl;
          of << "\t\t\t\t\t" << "\"kind\" : \"write\"," << std::endl;
          of << "\t\t\t\t\t" << "\"relation\" : \"" << write << "\"" << std::endl;
          of << "\t\t\t\t" << (i + static_cast<int>(st->ris.size()) < total_rws - 1 ? "}," : "}") << std::endl;
        }

        of << "\t\t\t" << "]," << std::endl;
      }

      /* domain */
      {
        of << "\t\t\t" << "\"domain\" : \"" << params_str << " -> { ";

        std::string stmt_full_name;
        if (!st->ris.empty()) {
          stmt_full_name = st->reads.at(st->ris[0]).space().domain().to_str();
        } else {
          stmt_full_name = st->writes.at(st->wis[0]).space().domain().to_str();
        }
        stmt_full_name = stmt_full_name.substr(3, stmt_full_name.find("]") - 2);
        of << stmt_full_name << " : ";

        /* extract params of this statement */
        std::vector<std::string> stmt_params;
        for (std::size_t p = stmt_full_name.find("[") + 1; p < stmt_full_name.size();) {
          /* no params */
          if (stmt_full_name[p] == ']') {
            break;
          }

          std::size_t q = p + 1;
          /* move q to ',' or ']' */
          while (stmt_full_name[q] != ',' && stmt_full_name[q] != ']') {
            ++q;
          }
          stmt_params.push_back(stmt_full_name.substr(p, q - p));

          /* reached the end */
          if (stmt_full_name[q] == ']') {
            break;
          }

          /* move p to next non-blank character */
          for (p = q + 1; stmt_full_name[p] == ' '; ++p);
        }

        for (auto it = stmt_params.begin(); it != stmt_params.end(); ++it) {
          if (it != stmt_params.begin()) {
            of << "and ";
          }
          of << st->param_lower[*it] << " <= " << *it << " < " << st->param_upper[*it] + 1 << " ";
        }
        
        of << "}\"," << std::endl;
      }

      /* name */
      {
        of << "\t\t\t" << "\"name\" : \"" << st->name << "\"," << std::endl;
      }

      /* schedule */
      {
        isl::map_list sch_list = sch.map().map_list();
        for (unsigned int i = 0; i < sch_list.size(); ++i) {
          if (sch_list.at(i).to_str().find(st->name) != std::string::npos) {
            of << "\t\t\t" << "\"schedule\" : \"" << params_str << " -> ";
            of << sch_list.at(i);
            of << "\"" << std::endl;
            break;
          }
        }
      }

      of << "\t\t" << "}";
    }
    of << std::endl;
    of << "\t" << "]" << std::endl;
  }

  of << "}";

  of.close();
  return true;
}

void ScopInfo::DumpSchTree(const std::string &file_name, const isl::schedule &sch_dump) {
  std::stringstream final_file_name;
  final_file_name << std::setw(2) << std::setfill('0') << dump_schtree_count << "_" << file_name
                  << std::string(mmu_info_.IsSpecGemm() ? "_specgemm" : "");
  if (user_config_.GetDumpPassIr()) {
#if DUMP_IR
    DumpSchTreeImpl(CreateDumpDir(final_file_name.str()), sch_dump);
    dump_schtree_count++;
#if DUMP_RAW_SCH_TREE
    DumpRawSchTreeImpl(CreateDumpDir("Raw_" + final_file_name.str()), sch_dump);
#endif
#endif

#if DUMP_SCOP_DATA
#if DUMP_SCOP_DATA_PER_PASS
    static_cast<void>(DumpScopData(CreateDumpDir(final_file_name.str())));
#else
    static_cast<void>(DumpScopData(CreateDumpDir("scop")));
#endif
#endif
  }
}

void ScopInfo::DumpJscop(const std::string &file_name, const isl::schedule &sch_dump) {
  std::stringstream final_file_name;
  final_file_name << file_name << std::string(mmu_info_.IsSpecGemm() ? "_specgemm" : "");
  if (user_config_.GetDumpPassIr()) {
    static_cast<void>(DumpJscopData(CreateDumpDir(final_file_name.str()), sch_dump));
    // DumpRawSchTreeImpl(CreateDumpDir("Raw_" + final_file_name.str()), sch_dump);
  }
}

std::string ScopInfo::AddDumpDir(const std::string &file_name) {
  std::string real_file_name = file_name;
  bool is_specgemm = (user_config_.GetIsolatedIdx() > 0);
  if (is_specgemm) {
    std::string dump_isolate_dir = "specgemm_" + std::to_string(user_config_.GetIsolatedIdx());
    real_file_name = dump_isolate_dir + '/' + real_file_name;
  }

#if (!DUMP_IN_CURRENT_DIR)
  if (!user_config_.GetDumpPolyDir().empty()) {
    real_file_name = user_config_.GetDumpPolyDir() + '/' + real_file_name;
  }
#endif
  return real_file_name;
}

std::string ScopInfo::CreateDumpDir(const std::string &file_name) {
  std::string real_file_name = AddDumpDir(file_name);
  CreateDirIfNotExist(real_file_name);
  return real_file_name;
}

void AnalysisResult::DumpBufferDefInfos(std::ostream &out) {
  for (size_t index = 0; index < buffer_def_infos_.size(); index++) {
    out << "\r\nbufferedDefInfos_[" << index << "]: " << std::endl;
    out << "    tensor_id       : " << buffer_def_infos_[index].tensor_id << std::endl;
    out << "   dst_tensor_id    : " << buffer_def_infos_[index].dst_tensor_id << std::endl;
    out << " ancester_tensor_id : " << buffer_def_infos_[index].ancester_tensor_id << std::endl;
    out << "    mem_type        : " << static_cast<int>(buffer_def_infos_[index].mem_type) << std::endl;
    out << "    mark_tag        : " << buffer_def_infos_[index].mark_tag << std::endl;
    out << "    find_buffer     : " << buffer_def_infos_[index].find_buffer << std::endl;
    out << "    is_bind_tensor  : " << buffer_def_infos_[index].is_bind_tensor << std::endl;
  }
}

void ScopInfo::DumpTransform(const std::string &file_name, PassInfo &pass_info) {
  auto real_path = CreateDumpDir(file_name);
  std::ofstream of;
  of.open(real_path, std::ios::out);
  if (!of.is_open()) {
    return;
  }

  PrintHeader(of, "group_filter_map");
  for (const auto &group : pass_info.group_filter_map_) {
    of << group.first << " : [ ";
    for (auto filter : group.second) {
      of << filter << ", ";
    }
    of << "]" << std::endl;
  }

  PrintHeader(of, "dependences");
  of << FormatMupaStr(pass_info.dependences_.to_str()) << std::endl;

  PrintHeader(of, "constraints");
  isl_printer *p;
  char *s = nullptr;
  p = isl_printer_to_str(GetCtx().get());
  CHECK(p != nullptr);
  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
  p = isl_printer_print_schedule_constraints(p, pass_info.constraints_.get());
  s = isl_printer_get_str(p);
  if (s) {
    of << FormatMupaStr(s);
    free(s);
  }
  static_cast<void>(isl_printer_free(p));

  PrintHeader(of, "time_records");
  for (auto time_log : time_records_) {
    of << time_log << std::endl;
  }

  of.close();
}
}  // namespace poly
}  // namespace ir
}  // namespace akg
