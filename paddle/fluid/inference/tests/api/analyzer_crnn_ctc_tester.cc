// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <fstream>
#include <opencv2/opencv.hpp>

#include "paddle/fluid/inference/analysis/analyzer.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/api/paddle_inference_pass.h"
#include "paddle/fluid/platform/profiler.h"

#include <boost/optional.hpp>

DEFINE_bool(mkldnn_used, false, "Use MKLDNN.");
DEFINE_string(data_list, "",
              "Path to a file with a list of images. Format of a line: h w "
              "filename,comma-separated indices.");
DEFINE_string(image_dir, "",
              "Directory with images given in a data_list argument.");
DEFINE_int64(iterations, 0, "Number of iterations.");
DEFINE_string(infer_model, "", "Saved inference model.");
DEFINE_int64(batch_size, 1, "Batch size.");
DEFINE_bool(print_results, false, "Print inference results.");
DEFINE_int64(skip_batches, 0, "Number of warm-up iterations.");
DECLARE_bool(profile);

// Default values for a Baidu dataset that we're using
DEFINE_int64(image_height, 48, "Height of an image.");
DEFINE_int64(image_width, 384, "Width of an image.");

namespace paddle {
struct GrayscaleImageReader {
  static constexpr int64_t channels = 1;

  std::shared_ptr<float> operator()(int64_t image_height, int64_t image_width,
                                    int64_t height, int64_t width,
                                    const std::string& filename) {
    cv::Mat image = cv::imread(filename, cv::IMREAD_GRAYSCALE);

    cv::Mat resized_image;
    if (height != image_height || width != image_width) {
      cv::resize(image, resized_image, cv::Size(image_width, image_height));
    } else {
      resized_image = image;
    }

    cv::Mat float_image;
    resized_image.convertTo(float_image, CV_32FC1);
    cv::Scalar mean = {127.5};

    float_image -= mean;

    size_t image_rows = float_image.rows;
    size_t image_cols = float_image.cols;
    size_t image_size = image_rows * image_cols;

    std::shared_ptr<float> image_vec{new float[image_size],
                                     std::default_delete<float[]>()};
    std::copy_n(float_image.ptr<const float>(), image_size, image_vec.get());

    return image_vec;
  }
};

struct DatafileParser {
  explicit DatafileParser(bool is_circular, std::string image_dir,
                          std::string data_list_file)
      : is_circular{is_circular}, image_dir{image_dir} {
    if (data_list_file.empty()) {
      throw std::invalid_argument("Name of the data list file empty.");
    }

    data_list_stream.open(data_list_file);

    if (!data_list_stream) {
      if (data_list_stream.is_open()) data_list_stream.close();

      throw std::invalid_argument("Couldn't open a file " + data_list_file);
    }
  }

  template <typename R, typename ConvertFunc, typename IT>
  std::pair<R, IT> retrieve_token(IT s, IT e, char sep, ConvertFunc f) {
    std::string token_str;

    auto it = std::find(s, e, sep);
    std::copy(s, it, std::back_inserter(token_str));
    R r = f(token_str);

    return std::make_pair(std::move(r), std::move(it));
  }

  using parse_results =
      std::tuple<int64_t, int64_t, std::string, std::vector<int64_t>>;
  parse_results parse_single_line(std::string line) {
    // w h file_name idx1,idx2,idx3,...
    auto it = std::begin(line);

    auto width_pair = retrieve_token<int64_t>(
        it, std::end(line), ' ',
        [](std::string s) -> int64_t { return std::stoll(s); });
    auto width = width_pair.first;
    it = width_pair.second;
    it++;

    auto height_pair = retrieve_token<int64_t>(
        it, std::end(line), ' ',
        [](std::string s) -> int64_t { return std::stoll(s); });
    auto height = height_pair.first;
    it = height_pair.second;
    it++;

    auto filename_pair = retrieve_token<std::string>(
        it, std::end(line), ' ',
        [](std::string s) -> std::string { return s; });
    auto filename = filename_pair.first;
    it = filename_pair.second;
    it++;

    auto indices_pair = retrieve_token<std::string>(
        it, std::end(line), ' ',
        [](std::string s) -> std::string { return s; });
    auto indices_str = indices_pair.first;

    std::vector<int64_t> indices;
    it = std::begin(indices_str);

    while (true) {
      auto p = retrieve_token<int64_t>(
          it, std::end(indices_str), ',',
          [](std::string s) -> int64_t { return std::stoll(s); });
      indices.push_back(p.first);
      it = p.second;

      if (it != std::end(indices_str))
        it++;
      else
        break;
    }

    return std::make_tuple(height, width, filename, indices);
  }

  boost::optional<parse_results> Line() {
    std::int64_t height;
    std::int64_t width;
    std::string filename;
    std::vector<int64_t> indices;

    // If file is empty and circular, there could be an infinite loop.
    do {
      std::string line;
      std::getline(data_list_stream, line);

      if (data_list_stream.good()) {
        std::tie(height, width, filename, indices) = parse_single_line(line);

        return std::make_tuple(height, width, image_dir + "/" + filename,
                               indices);
      } else {
        if (is_circular) {
          data_list_stream.clear();
          data_list_stream.seekg(0);
        } else {
          return boost::none;
        }
      }
    } while (true);
    return boost::none;
  }

 private:
  bool is_circular;
  std::string image_dir;
  std::ifstream data_list_stream;
};

struct Once {
  bool operator()() { return true; }
};

struct Iterations {
  explicit Iterations(int64_t iterations) : elapsed_iters{iterations} {}

  bool operator()() {
    bool to_continue = elapsed_iters > 0;
    elapsed_iters--;
    return to_continue;
  }

  int64_t elapsed_iters;
};

struct DataReader {
  explicit DataReader(int64_t iterations, std::string image_dir,
                      std::string data_list_file, int64_t batch_size,
                      int64_t image_height, int64_t image_width)
      : datafile_parser{iterations != 0, image_dir, data_list_file},
        elapsed_iters{iterations},
        batch_size{batch_size},
        image_height{image_height},
        image_width{image_width} {
    auto check_mode = [iterations]() -> std::function<bool()> {
      if (!iterations)
        return Once{};
      else
        return Iterations{iterations};
    };

    mode_func = check_mode();
  }

  using DataRecord = std::tuple<int64_t, int64_t, int64_t, std::vector<int64_t>,
                                std::shared_ptr<float>>;

  using DataChunk =
      std::tuple<int64_t, int64_t, int64_t, int64_t,
                 std::vector<std::vector<int64_t>>, std::shared_ptr<float>>;

 private:
  DatafileParser datafile_parser;
  GrayscaleImageReader image_reader;

  int64_t elapsed_iters;
  int64_t batch_size;
  const int64_t channels = GrayscaleImageReader::channels;
  int64_t image_height;
  int64_t image_width;
  std::function<bool()> mode_func;

 public:
  boost::optional<DataRecord> get_data_record() {
    std::int64_t height;
    std::int64_t width;
    std::string filename;
    std::vector<int64_t> indices;

    auto parsed_line = datafile_parser.Line();

    if (parsed_line) {
      std::tie(height, width, filename, indices) = *parsed_line;

      auto image =
          image_reader(image_height, image_width, height, width, filename);

      return std::make_tuple(channels, image_height, image_width, indices,
                             image);
    }

    return boost::none;
  }

 public:
  std::vector<DataRecord> Next() {
    std::vector<DataRecord> data_records;

    bool to_continue = mode_func();
    int bi = 0;
    while ((bi < batch_size) && to_continue) {
      auto data_record = get_data_record();
      if (data_record) {
        data_records.push_back(*data_record);
        bi++;
      } else {
        break;
      }
    }

    return data_records;
  }

  boost::optional<DataChunk> Batch() {
    auto data_records = Next();

    if (data_records.empty()) return boost::none;

    size_t image_size = channels * image_height * image_width;
    std::shared_ptr<float> data_chunk{new float[batch_size * image_size],
                                      std::default_delete<float[]>()};

    using image_iterator = std::vector<float>::iterator;

    std::accumulate(std::begin(data_records), std::end(data_records),
                    data_chunk.get(),
                    [image_size](float* ptr, const DataRecord& dr) {
                      auto img = std::get<4>(dr).get();
                      std::copy(img, img + image_size, ptr);
                      return ptr + image_size;
                    });

    std::vector<std::vector<int64_t>> indices;
    std::transform(std::begin(data_records), std::end(data_records),
                   std::back_inserter(indices),
                   [](const DataRecord& dr) -> std::vector<int64_t> {
                     return std::get<3>(dr);
                   });

    return std::make_tuple(batch_size, channels, image_height, image_width,
                           indices, data_chunk);
  }
};

PaddleTensor PrepareData(const DataReader::DataChunk& data_chunk) {
  paddle::PaddleTensor input;
  input.name = "image";

  auto batch_size = std::get<0>(data_chunk);
  auto channels = std::get<1>(data_chunk);
  auto height = std::get<2>(data_chunk);
  auto width = std::get<3>(data_chunk);
  auto indices = std::get<4>(data_chunk);
  auto image = std::get<5>(data_chunk);

  input.shape = {static_cast<int>(batch_size), static_cast<int>(channels),
                 static_cast<int>(height), static_cast<int>(width)};

  input.dtype = PaddleDType::FLOAT32;
  input.data.Reset(image.get(),
                   batch_size * channels * height * width * sizeof(float));

  return input;
}

void PrintResults(const std::vector<PaddleTensor>& output) {
  auto lod = output[0].lod[0];

  int64_t* output_data = static_cast<int64_t*>(output[0].data.data());
  std::ostream_iterator<std::string> ot{std::cout};

  auto it = std::begin(lod);
  std::transform(it + 1, std::end(lod), it, ot,
                 [output_data](int64_t f, int64_t e) -> std::string {
                   std::ostringstream ss;
                   std::ostream_iterator<int64_t> is{ss, ","};

                   std::copy(output_data + e, output_data + f, is);
                   return ss.str();
                 });
  std::cout << "\n";
}

TEST(crnn_ctc, basic) {
  DataReader data_reader{FLAGS_iterations,   FLAGS_image_dir,
                         FLAGS_data_list,    FLAGS_batch_size,
                         FLAGS_image_height, FLAGS_image_width};

  contrib::AnalysisConfig config;
  config.model_dir = FLAGS_infer_model;
  config.use_gpu = false;
  config.enable_ir_optim = false;
  config._use_mkldnn = FLAGS_mkldnn_used;

  auto predictor = CreatePaddlePredictor<contrib::AnalysisConfig,
                                         PaddleEngineKind::kAnalysis>(config);

  inference::Timer timer;
  inference::Timer total_timer;

  std::vector<double> fpses;

  auto run_experiment =
      [&predictor](const PaddleTensor& input) -> std::vector<PaddleTensor> {
    std::vector<paddle::PaddleTensor> output_slots;
    predictor->Run({input}, &output_slots);

    return output_slots;
  };

  std::cout << "Warm-up: " << FLAGS_skip_batches << " iterations.\n";
  for (size_t i = 0; i < FLAGS_skip_batches; ++i) {
    auto data_chunk = data_reader.Batch();

    run_experiment(PrepareData(*data_chunk));
  }

  if (FLAGS_profile) {
    auto pf_state = paddle::platform::ProfilerState::kCPU;
    paddle::platform::EnableProfiler(pf_state);
    paddle::platform::ResetProfiler();
  }

  std::cout << "Execution iterations: " << FLAGS_iterations << " iterations.\n";

  int i = 0;
  while (auto data_chunk = data_reader.Batch()) {
    auto input = PrepareData(*data_chunk);

    if (i == 0) {
      total_timer.tic();
    }

    timer.tic();
    auto output_slots = run_experiment(input);
    double batch_time = timer.toc() / 1000;

    double fps = FLAGS_batch_size / batch_time;
    fpses.push_back(fps);

    std::cout << "Iteration: " << i << " latency: " << batch_time
              << " fps: " << fps << "\n";

    if (FLAGS_print_results) {
      PrintResults(output_slots);
    }
    i++;
  }

  if (FLAGS_profile) {
    paddle::platform::DisableProfiler(paddle::platform::EventSortingKey::kTotal,
                                      "/tmp/profiler");
  }

  double total_time = total_timer.toc() / 1000;

  double avg_fps =
      std::accumulate(std::begin(fpses), std::end(fpses), 0.f) / fpses.size();
  double avg_latency = total_time / i;

  std::cout << "Iterations: " << i << " average latency: " << avg_latency
            << " average fps: " << avg_fps << "\n";
}
}  // namespace paddle
