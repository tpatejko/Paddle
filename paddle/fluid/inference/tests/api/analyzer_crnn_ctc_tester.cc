#include <fstream>
#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

DEFINE_string(data_list, "", "Path to a file with a list of images. Format of a line: h w filename,comma-separated indices.");
DEFINE_string(image_dir, "", "Directory with images given in a data_list argument.");

namespace paddle {

struct DataReader {
  explicit DataReader(std::string data_list_file,
                      std::string image_dir) 
    : image_dir{image_dir} {
    if (data_list_file.empty()) {
      throw std::invalid_argument("Name of the data list file empty.");
    }

    data_list_stream.open(data_list_file);

    if (!data_list_stream) {
      if (data_list_stream.is_open())
        data_list_stream.close();

      throw std::invalid_argument("Couldn't open a file " + data_list_file);
    }
  }

  struct DataRecord {
    std::vector<int64_t> indices;
    std::unique_ptr<float[]> data;
  };
  
private:
  std::string image_dir;
  std::ifstream data_list_stream;

  template<typename R, typename ConvertFunc, typename IT>
  std::pair<R, IT> retrieve_token(IT s, IT e, char sep, ConvertFunc f) {
    std::string token_str;

    auto it = std::find_if(s, e, [sep](char a) -> bool { return a == sep; });
    std::copy(s, it, std::back_inserter(token_str));
    R r = f(token_str);

    return std::make_pair<R, IT>(std::move(r), std::move(it));
  }

public:
  using parse_results = std::tuple<int64_t, int64_t, std::string, std::vector<std::string>>;
  parse_results parse_single_line(std::string line) {
    // h w file_name idx1,idx2,idx3,...

    auto it = std::begin(line);

    auto height_pair = retrieve_token<int64_t>(it, std::end(line), ' ', [](std::string s) -> int64_t { return std::stoll(s); });
    auto height = height_pair.first;
    it = height_pair.second;
    it++;

    auto width_pair = retrieve_token<int64_t>(it, std::end(line), ' ', [](std::string s) -> int64_t { return std::stoll(s); });
    auto width = width_pair.first;
    it = width_pair.second;
    it++;

    auto filename_pair = retrieve_token<std::string>(it, std::end(line), ' ', [](std::string s) -> std::string { return s; });
    auto filename = filename_pair.first;
    it = filename_pair.second;
    it++;

    auto indices_pair = retrieve_token<std::string>(it, std::end(line), ' ', [](std::string s) -> std::string { return s; });
    auto indices_str = indices_pair.first;

    std::vector<int64_t> indices;
    it = std::begin(indices_str);

    while(true) {
      auto p = retrieve_token<int64_t>(it, std::end(indices_str), ',', [](std::string s) -> int64_t { return std::stoll(s); });
      indices.push_back(p.first);
      it = p.second;

      if (it != std::end(indices_str))
        it++;
      else
        break;
    }

    return std::make_tuple(height, width, filename, indices);
  }

  std::unique_ptr<float[]> retrieve_image_data(int64_t height, int64_t width, std::string filename) {
    auto full_filename = image_dir + "/" + filename;

    cv::Mat image = cv::imread(full_filename, cv::IMREAD_COLOR);

    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(height, width));

    cv::Mat float_image;
    cv::convertTo(resized_image, CV_32FC3);

    float_image /= 255.f;
    cv::Scalar mean{0.406f, 0.456f, 0.485f};
    cv::Scalar std{0.255f, 0.224f, 0.229f};

    std::vector<cv::Mat> image_channels;
    cv::split(float_image, image_channels);

    constexpr size_t channels = 3;
    size_t image_rows = float_image.rows;
    size_t image_cols = float_image.cols;

    std::unique_ptr<float[]> image_ptr{new float[channels*image_rows*image_cols]}

    for (size_t c = 0; c < channels; ++c) {
      image_channels[c] -= mean[c];
      image_channels[c] /= std[c];

      for (size_t r = 0; r < float_image.rows; ++r) {
        auto row_begin = image_channels[c].ptr<float>(r);
        auto row_end = row_begin + float_image.cols;

        std::copy(row_begin, row_end,
                  image_ptr.get() + c*image_rows
      }
    }

  }

  DataRecord get_data_record(std::string line) {
    std::int64_t height;
    std::int64_t width;
    std::string filename;
    std::vector<int64_t> indices;

    std::tie(heigth, width, filename, indices) = parse_single_line(line);

  }
};

TEST(crnn_ctc, basic) {
  DataReader data_reader{FLAGS_data_list, FLAGS_image_dir};
  auto data_record = data_reader.parse_single_line("384 48 325_dame_19109.jpg 67,64,76,68");

  for (auto i : data_record.indices) {
    std::cout << i << " ";
  }

  std::cout << std::endl;
}

}  // namespace paddle
