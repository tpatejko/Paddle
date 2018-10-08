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
#include "paddle/fluid/inference/analysis/ut_helper.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/api/paddle_inference_pass.h"

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

// Default values for a Baidu dataset that we're using
DEFINE_int64(image_height, 48, "Height of an image.");
DEFINE_int64(image_width, 384, "Width of an image.");

namespace paddle {
/*
  std::pair<int64_t, std::unique_ptr<float[]>> retrieve_image_data(
      int64_t height, int64_t width, std::string filename) {
    auto full_filename = image_dir + "/" + filename;

    cv::Mat image = cv::imread(full_filename, cv::IMREAD_GRAYSCALE);

    cv::Mat resized_image;
    if (height != image_height || width != image_width) {
      cv::resize(image, resized_image, cv::Size(image_width, image_height));
    } else {
      resized_image = image;
    }

    cv::Mat float_image;
    resized_image.convertTo(float_image, CV_32FC1);

    std::vector<cv::Mat> image_channels;
    cv::split(float_image, image_channels);

    size_t image_rows = float_image.rows;
    size_t image_cols = float_image.cols;
    size_t image_size = image_rows * image_cols;

    std::unique_ptr<float[]> image_ptr{new float[channels * image_size]};

    std::vector<float> mean(channels, 127.5);

    std::transform(std::begin(image_channels), std::end(image_channels),
                   std::begin(mean), std::begin(image_channels),
                   [](cv::Mat& mat, float mean) -> cv::Mat {
                     return mat - cv::Scalar{mean};
                   });

    std::accumulate(std::begin(image_channels), std::end(image_channels),
                    image_ptr.get(),
                    [image_size](float* img_ptr, const cv::Mat& mat) -> float* {
                      std::copy_n(mat.ptr<const float>(), image_size, img_ptr);
                      return img_ptr + image_size;
                    });

    return std::make_pair(channels, std::move(image_ptr));
  }
*/

struct GrayscaleImageReader {
  static constexpr int64_t channels = 1;

  std::unique_ptr<float[]> operator()(int64_t image_height, int64_t image_width,
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

    resized_image -= mean;

    size_t image_rows = float_image.rows;
    size_t image_cols = float_image.cols;
    size_t image_size = image_rows * image_cols;

    std::unique_ptr<float[]> image_ptr{new float[image_size]};
    std::copy_n(resized_image.ptr<const float>(), image_size, image_ptr.get());

    return image_ptr;
  }
};

struct DataReader {
  explicit DataReader(std::string data_list_file, std::string image_dir,
                      int64_t batch_size, int64_t image_height,
                      int64_t image_width)
      : image_dir{image_dir},
        batch_size{batch_size},
        image_height{image_height},
        image_width{image_width} {
    if (data_list_file.empty()) {
      throw std::invalid_argument("Name of the data list file empty.");
    }

    data_list_stream.open(data_list_file);

    if (!data_list_stream) {
      if (data_list_stream.is_open()) data_list_stream.close();

      throw std::invalid_argument("Couldn't open a file " + data_list_file);
    }
  }

  struct DataRecord {
    int64_t channels;
    int64_t height;
    int64_t width;
    std::vector<int64_t> indices;
    std::unique_ptr<float[]> data;
  };

  struct DataChunk {
    int64_t batch_size;
    int64_t channels;
    int64_t height;
    int64_t width;
    std::vector<std::vector<int64_t>> indices;
    std::unique_ptr<float[]> data;
  };

 private:
  std::string image_dir;
  std::ifstream data_list_stream;

  GrayscaleImageReader image_reader;

  int64_t batch_size;
  const int64_t channels = GrayscaleImageReader::channels;
  int64_t image_height;
  int64_t image_width;

  template <typename R, typename ConvertFunc, typename IT>
  std::pair<R, IT> retrieve_token(IT s, IT e, char sep, ConvertFunc f) {
    std::string token_str;

    auto it = std::find(s, e, sep);
    std::copy(s, it, std::back_inserter(token_str));
    R r = f(token_str);

    return std::make_pair(std::move(r), std::move(it));
  }

 public:
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

  DataRecord get_data_record(std::string line) {
    std::int64_t height;
    std::int64_t width;
    std::string filename;
    std::vector<int64_t> indices;

    std::tie(height, width, filename, indices) = parse_single_line(line);
    auto image_ptr = image_reader(image_height, image_width, height, width,
                                  image_dir + "/" + filename);

    return DataRecord{channels, image_height, image_width, indices,
                      std::move(image_ptr)};
  }

  std::vector<DataRecord> data_records;

 public:
  bool Next(bool is_circular = false) {
    data_records.clear();
    int bi = 0;
    while (bi < batch_size) {
      std::string line;
      if (std::getline(data_list_stream, line)) {
        auto data_record = get_data_record(line);
        data_records.push_back(std::move(data_record));
        bi++;
      } else {
        if (is_circular) {
          data_list_stream.clear();
          data_list_stream.seekg(0);
        } else {
          break;
        }
      }
    }

    return data_list_stream.good();
  }

  DataChunk Batch() {
    DataChunk data_chunk;
    size_t image_size = channels * image_height * image_width;
    data_chunk.data.reset(new float[batch_size * image_size]);

    std::accumulate(std::begin(data_records), std::end(data_records),
                    data_chunk.data.get(),
                    [image_size](float* ptr, const DataRecord& dr) -> float* {
                      auto img_ptr = dr.data.get();

                      std::copy(img_ptr, img_ptr + image_size, ptr);
                      return ptr + image_size;
                    });

    std::transform(std::begin(data_records), std::end(data_records),
                   std::back_inserter(data_chunk.indices),
                   [](const DataRecord& dr) -> std::vector<int64_t> {
                     return dr.indices;
                   });

    data_chunk.batch_size = batch_size;
    data_chunk.channels = channels;
    data_chunk.height = image_height;
    data_chunk.width = image_width;

    return data_chunk;
  }
};

PaddleTensor PrepareData(const DataReader::DataChunk& data_chunk) {
  paddle::PaddleTensor input;
  input.name = "image";

  input.shape = {static_cast<int>(data_chunk.batch_size),
                 static_cast<int>(data_chunk.channels),
                 static_cast<int>(data_chunk.height),
                 static_cast<int>(data_chunk.width)};

  input.dtype = PaddleDType::FLOAT32;
  input.data.Reset(data_chunk.data.get(),
                   data_chunk.batch_size * data_chunk.channels *
                       data_chunk.height * data_chunk.width * sizeof(float));

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
  DataReader data_reader{FLAGS_data_list, FLAGS_image_dir, FLAGS_batch_size,
                         FLAGS_image_height, FLAGS_image_width};

  contrib::AnalysisConfig config;
  config.model_dir = FLAGS_infer_model;
  config.use_gpu = false;
  //  config.enable_ir_optim = false;

  auto predictor = CreatePaddlePredictor<contrib::AnalysisConfig,
                                         PaddleEngineKind::kAnalysis>(config);

  inference::Timer timer;
  inference::Timer total_timer;

  std::vector<double> fpses;

  auto run_experiment =
      [&predictor](const PaddleTensor& input) -> std::vector<PaddleTensor> {
    std::vector<paddle::PaddleTensor> output_slots;
    CHECK(predictor->Run({input}, &output_slots));

    return output_slots;
  };

  std::cout << "Warm-up: " << FLAGS_skip_batches << " iterations.\n";
  for (size_t i = 0; i < FLAGS_skip_batches; ++i) {
    data_reader.Next(true /* is_circular */);
    auto data_chunk = data_reader.Batch();

    run_experiment(PrepareData(data_chunk));
  }

  std::cout << "Execution iterations: " << FLAGS_iterations << " iterations.\n";
  for (size_t i = 0; i < FLAGS_iterations; ++i) {
    data_reader.Next(true /* is_circular */);
    auto data_chunk = data_reader.Batch();

    auto input = PrepareData(data_chunk);

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
  }

  double total_time = total_timer.toc() / 1000;

  double avg_fps =
      std::accumulate(std::begin(fpses), std::end(fpses), 0.f) / fpses.size();
  double avg_latency = total_time / FLAGS_iterations;

  std::cout << "Iterations: " << FLAGS_iterations
            << " average latency: " << avg_latency
            << " average fps: " << avg_fps << "\n";
}
}  // namespace paddle
