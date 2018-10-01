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

namespace paddle {

struct DataReader {
  explicit DataReader(std::string data_list_file, std::string image_dir,
                      int64_t batch_size)
      : image_dir{image_dir}, batch_size{batch_size} {
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

 private:
  std::string image_dir;
  std::ifstream data_list_stream;
  int64_t batch_size;

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
    // h w file_name idx1,idx2,idx3,...

    auto it = std::begin(line);

    auto height_pair = retrieve_token<int64_t>(
        it, std::end(line), ' ',
        [](std::string s) -> int64_t { return std::stoll(s); });
    auto height = height_pair.first;
    it = height_pair.second;
    it++;

    auto width_pair = retrieve_token<int64_t>(
        it, std::end(line), ' ',
        [](std::string s) -> int64_t { return std::stoll(s); });
    auto width = width_pair.first;
    it = width_pair.second;
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

  std::pair<int64_t, std::unique_ptr<float[]>> retrieve_image_data(
      int64_t height, int64_t width, std::string filename) {
    auto full_filename = image_dir + "/" + filename;

    cv::Mat image = cv::imread(full_filename, cv::IMREAD_GRAYSCALE);

    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(width, height));

    cv::Mat float_image;
    resized_image.convertTo(float_image, CV_32FC1);

    cv::Scalar mean = {127.5};

    std::vector<cv::Mat> image_channels;
    cv::split(float_image, image_channels);

    constexpr size_t channels = 1;
    size_t image_rows = float_image.rows;
    size_t image_cols = float_image.cols;
    size_t image_size = image_rows * image_cols;

    std::unique_ptr<float[]> image_ptr{new float[channels * image_size]};

    for (size_t c = 0; c < channels; ++c) {
      image_channels[c] -= mean[c];

      std::copy_n(image_channels[c].ptr<float>(), image_size,
                  image_ptr.get() + c * image_size);
    }

    return std::make_pair(channels, std::move(image_ptr));
  }

  DataRecord get_data_record(std::string line) {
    std::int64_t height;
    std::int64_t width;
    std::string filename;
    std::vector<int64_t> indices;

    std::unique_ptr<float[]> image_ptr;
    int64_t channels;

    std::tie(height, width, filename, indices) = parse_single_line(line);
    std::tie(channels, image_ptr) =
        retrieve_image_data(height, width, filename);

    return DataRecord{channels, height, width, indices, std::move(image_ptr)};
  }

 public:
  bool Next() { return data_list_stream.good(); }

  std::vector<DataRecord> Batch() {
    std::vector<DataRecord> batch;

    size_t bi = 0;
    std::string line;
    while (getline(data_list_stream, line) && bi < batch_size) {
      auto data_record = get_data_record(line);
      batch.push_back(std::move(data_record));
      bi++;
    }

    return batch;
  }
};

TEST(crnn_ctc, basic) {
  DataReader data_reader{FLAGS_data_list, FLAGS_image_dir, FLAGS_batch_size};

  while (data_reader.Next()) {
    auto data_record = data_reader.Batch();

    std::cout << "Batch\n";
    for (auto& dr : data_record) {
      for (auto i : dr.indices) {
        std::cout << i << " ";
      }
    }
  }
  //  auto data_record = data_reader.get_data_record("384 48 325_dame_19109.jpg
  //  67,64,76,68");

  /*


    contrib::AnalysisConfig config;
    config.model_dir = FLAGS_infer_model;
    config.use_gpu = false;

    auto predictor = CreatePaddlePredictor<contrib::AnalysisConfig,
                                           PaddleEngineKind::kAnalysis>(config);

    paddle::PaddleTensor input;
    input.name = "image";
    input.shape = {1, static_cast<int>(data_record.channels),
    static_cast<int>(data_record.height), static_cast<int>(data_record.width)};
    input.dtype = PaddleDType::FLOAT32;
    input.data.Reset(data_record.data.get(), data_record.channels *
    data_record.height * data_record.width * sizeof(float));

    std::vector<paddle::PaddleTensor> output_slots;

    CHECK(predictor->Run({input}, &output_slots));

    int32_t* output_data = static_cast<int32_t*>(output_slots[0].data.data());
    std::cout << "First value " << output_data[0] << "\n";
    std::cout << "First value " << output_data[1] << "\n";
    std::cout << "First value " << output_data[2] << "\n";
    */
}

}  // namespace paddle
