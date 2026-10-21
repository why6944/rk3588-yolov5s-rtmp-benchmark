#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "rknn_api.h"

namespace {

constexpr int kClasses = 80;
constexpr int kIouCount = 10;
constexpr float kCandidateThreshold = 0.001f;
constexpr float kNmsThreshold = 0.60f;
constexpr float kPrecisionRecallThreshold = 0.25f;
const float kAnchors[9][2] = {
    {10, 13}, {16, 30}, {33, 23}, {30, 61}, {62, 45},
    {59, 119}, {116, 90}, {156, 198}, {373, 326},
};

struct Box {
  float x1;
  float y1;
  float x2;
  float y2;
};

struct Detection {
  int class_id;
  float score;
  Box box;
};

struct Label {
  int class_id;
  Box box;
};

struct PredictionStat {
  int class_id;
  float score;
  std::array<unsigned char, kIouCount> correct{};
};

struct Metrics {
  int images = 0;
  int labels = 0;
  int detections = 0;
  int classes_with_labels = 0;
  float map50 = 0.0f;
  float map50_95 = 0.0f;
  float precision = 0.0f;
  float recall = 0.0f;
};

float sigmoid(float value) {
  return 1.0f / (1.0f + std::exp(-value));
}

float iou(const Box& a, const Box& b) {
  const float x1 = std::max(a.x1, b.x1);
  const float y1 = std::max(a.y1, b.y1);
  const float x2 = std::min(a.x2, b.x2);
  const float y2 = std::min(a.y2, b.y2);
  const float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
  const float area_a = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
  const float area_b = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
  return intersection / (area_a + area_b - intersection + 1e-16f);
}

std::vector<std::string> list_jpgs(const std::string& directory) {
  DIR* dir = opendir(directory.c_str());
  if (!dir) throw std::runtime_error("Cannot open image directory: " + directory);
  std::vector<std::string> files;
  while (dirent* entry = readdir(dir)) {
    const std::string name(entry->d_name);
    if (name.size() > 4 && name.substr(name.size() - 4) == ".jpg") files.push_back(name);
  }
  closedir(dir);
  std::sort(files.begin(), files.end());
  return files;
}

std::set<std::string> load_excluded(const std::string& path) {
  std::set<std::string> names;
  std::ifstream file(path);
  if (!file) throw std::runtime_error("Cannot open exclusion list: " + path);
  std::string line;
  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (!line.empty()) names.insert(line);
  }
  return names;
}

std::vector<Label> load_labels(const std::string& path, int width, int height) {
  std::ifstream file(path);
  std::vector<Label> labels;
  if (!file) return labels;
  int class_id;
  float cx, cy, w, h;
  while (file >> class_id >> cx >> cy >> w >> h) {
    const float px = cx * width;
    const float py = cy * height;
    const float pw = w * width;
    const float ph = h * height;
    labels.push_back({class_id, {px - pw / 2.0f, py - ph / 2.0f, px + pw / 2.0f, py + ph / 2.0f}});
  }
  return labels;
}

class RknnYolo {
 public:
  RknnYolo(const std::string& model_path, bool skip_sigmoid) : skip_sigmoid_(skip_sigmoid) {
    load_model(model_path);
  }

  ~RknnYolo() {
    if (context_) rknn_destroy(context_);
  }

  int width() const { return input_width_; }
  int height() const { return input_height_; }

  std::vector<Detection> infer(const cv::Mat& bgr) {
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(input_width_, input_height_));
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    rknn_input input{};
    input.index = 0;
    input.type = RKNN_TENSOR_UINT8;
    input.fmt = RKNN_TENSOR_NHWC;
    input.size = static_cast<uint32_t>(rgb.total() * rgb.elemSize());
    input.buf = rgb.data;
    if (rknn_inputs_set(context_, 1, &input) != RKNN_SUCC) throw std::runtime_error("rknn_inputs_set failed");
    if (rknn_run(context_, nullptr) != RKNN_SUCC) throw std::runtime_error("rknn_run failed");

    std::vector<rknn_output> outputs(io_num_.n_output);
    for (auto& output : outputs) output.want_float = 0;
    if (rknn_outputs_get(context_, io_num_.n_output, outputs.data(), nullptr) != RKNN_SUCC) {
      throw std::runtime_error("rknn_outputs_get failed");
    }

    std::vector<Detection> detections;
    try {
      for (uint32_t index = 0; index < io_num_.n_output; ++index) {
        decode_head(static_cast<const int8_t*>(outputs[index].buf), output_attrs_[index], detections);
      }
    } catch (...) {
      rknn_outputs_release(context_, io_num_.n_output, outputs.data());
      throw;
    }
    rknn_outputs_release(context_, io_num_.n_output, outputs.data());
    return class_aware_nms(detections);
  }

 private:
  void load_model(const std::string& model_path) {
    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open model: " + model_path);
    const std::streamsize size = file.tellg();
    model_data_.resize(static_cast<size_t>(size));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(model_data_.data()), size)) throw std::runtime_error("Cannot read model");
    if (rknn_init(&context_, model_data_.data(), model_data_.size(), RKNN_FLAG_PRIOR_HIGH, nullptr) != RKNN_SUCC) {
      throw std::runtime_error("rknn_init failed for " + model_path);
    }
    if (rknn_query(context_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_)) != RKNN_SUCC || io_num_.n_input != 1 || io_num_.n_output != 3) {
      throw std::runtime_error("Unexpected RKNN input/output count");
    }
    rknn_tensor_attr input_attr{};
    input_attr.index = 0;
    if (rknn_query(context_, RKNN_QUERY_INPUT_ATTR, &input_attr, sizeof(input_attr)) != RKNN_SUCC) {
      throw std::runtime_error("Cannot query RKNN input");
    }
    if (input_attr.fmt == RKNN_TENSOR_NHWC) {
      input_height_ = input_attr.dims[1];
      input_width_ = input_attr.dims[2];
    } else if (input_attr.fmt == RKNN_TENSOR_NCHW) {
      input_height_ = input_attr.dims[2];
      input_width_ = input_attr.dims[3];
    } else {
      throw std::runtime_error("Unsupported input layout");
    }
    output_attrs_.resize(io_num_.n_output);
    for (uint32_t index = 0; index < io_num_.n_output; ++index) {
      output_attrs_[index].index = index;
      if (rknn_query(context_, RKNN_QUERY_OUTPUT_ATTR, &output_attrs_[index], sizeof(rknn_tensor_attr)) != RKNN_SUCC) {
        throw std::runtime_error("Cannot query RKNN output");
      }
    }
  }

  float value(const int8_t* data, int offset, const rknn_tensor_attr& attr) const {
    float result = (static_cast<int>(data[offset]) - attr.zp) * attr.scale;
    return skip_sigmoid_ ? result : sigmoid(result);
  }

  void decode_head(const int8_t* data, const rknn_tensor_attr& attr, std::vector<Detection>& detections) const {
    const int grid_h = attr.dims[attr.n_dims - 2];
    const int grid_w = attr.dims[attr.n_dims - 1];
    const int stride = input_height_ / grid_h;
    int anchor_offset = -1;
    if (stride == 8) anchor_offset = 0;
    if (stride == 16) anchor_offset = 3;
    if (stride == 32) anchor_offset = 6;
    if (anchor_offset < 0) throw std::runtime_error("Unexpected YOLO output grid");
    const int grid_len = grid_h * grid_w;
    for (int anchor = 0; anchor < 3; ++anchor) {
      for (int row = 0; row < grid_h; ++row) {
        for (int col = 0; col < grid_w; ++col) {
          const int base = anchor * (kClasses + 5) * grid_len + row * grid_w + col;
          const float objectness = value(data, base + 4 * grid_len, attr);
          int class_id = 0;
          float class_prob = value(data, base + 5 * grid_len, attr);
          for (int class_index = 1; class_index < kClasses; ++class_index) {
            const float probability = value(data, base + (5 + class_index) * grid_len, attr);
            if (probability > class_prob) {
              class_prob = probability;
              class_id = class_index;
            }
          }
          const float score = objectness * class_prob;
          if (score < kCandidateThreshold) continue;
          const float x = (value(data, base, attr) * 2.0f - 0.5f + col) * stride;
          const float y = (value(data, base + grid_len, attr) * 2.0f - 0.5f + row) * stride;
          const float w = std::pow(value(data, base + 2 * grid_len, attr) * 2.0f, 2.0f) * kAnchors[anchor_offset + anchor][0];
          const float h = std::pow(value(data, base + 3 * grid_len, attr) * 2.0f, 2.0f) * kAnchors[anchor_offset + anchor][1];
          detections.push_back({class_id, score, {x - w / 2.0f, y - h / 2.0f, x + w / 2.0f, y + h / 2.0f}});
        }
      }
    }
  }

  std::vector<Detection> class_aware_nms(const std::vector<Detection>& candidates) const {
    std::vector<Detection> selected;
    for (int class_id = 0; class_id < kClasses; ++class_id) {
      std::vector<cv::Rect> boxes;
      std::vector<float> scores;
      std::vector<int> candidate_indices;
      for (size_t index = 0; index < candidates.size(); ++index) {
        if (candidates[index].class_id != class_id) continue;
        const Box& box = candidates[index].box;
        boxes.emplace_back(static_cast<int>(box.x1), static_cast<int>(box.y1),
                           std::max(0, static_cast<int>(box.x2 - box.x1)), std::max(0, static_cast<int>(box.y2 - box.y1)));
        scores.push_back(candidates[index].score);
        candidate_indices.push_back(static_cast<int>(index));
      }
      std::vector<int> kept;
      if (!boxes.empty()) cv::dnn::NMSBoxes(boxes, scores, 0.0f, kNmsThreshold, kept);
      for (int kept_index : kept) selected.push_back(candidates[candidate_indices[kept_index]]);
    }
    std::sort(selected.begin(), selected.end(), [](const Detection& a, const Detection& b) { return a.score > b.score; });
    return selected;
  }

  rknn_context context_ = 0;
  rknn_input_output_num io_num_{};
  std::vector<unsigned char> model_data_;
  std::vector<rknn_tensor_attr> output_attrs_;
  int input_width_ = 0;
  int input_height_ = 0;
  bool skip_sigmoid_ = false;
};

void match_predictions(const std::vector<Detection>& detections, const std::vector<Label>& labels,
                       std::vector<PredictionStat>& stats) {
  for (const Detection& detection : detections) stats.push_back({detection.class_id, detection.score, {}});
  for (int threshold_index = 0; threshold_index < kIouCount; ++threshold_index) {
    const float threshold = 0.50f + 0.05f * threshold_index;
    std::vector<bool> used(labels.size(), false);
    for (size_t detection_index = 0; detection_index < detections.size(); ++detection_index) {
      float best_iou = threshold;
      int best_label = -1;
      for (size_t label_index = 0; label_index < labels.size(); ++label_index) {
        if (used[label_index] || labels[label_index].class_id != detections[detection_index].class_id) continue;
        const float overlap = iou(detections[detection_index].box, labels[label_index].box);
        if (overlap >= best_iou) {
          best_iou = overlap;
          best_label = static_cast<int>(label_index);
        }
      }
      if (best_label >= 0) {
        used[best_label] = true;
        stats[stats.size() - detections.size() + detection_index].correct[threshold_index] = 1;
      }
    }
  }
}

float compute_ap(const std::vector<float>& recall, const std::vector<float>& precision) {
  float sum = 0.0f;
  for (int point = 0; point <= 100; ++point) {
    const float target = point / 100.0f;
    float best = 0.0f;
    for (size_t index = 0; index < recall.size(); ++index) {
      if (recall[index] >= target) best = std::max(best, precision[index]);
    }
    sum += best;
  }
  return sum / 101.0f;
}

Metrics summarize(const std::vector<PredictionStat>& stats, const std::array<int, kClasses>& label_counts, int images, int labels) {
  Metrics metrics;
  metrics.images = images;
  metrics.labels = labels;
  metrics.detections = static_cast<int>(stats.size());
  float sum_ap50 = 0.0f;
  float sum_ap = 0.0f;
  float sum_precision = 0.0f;
  float sum_recall = 0.0f;
  for (int class_id = 0; class_id < kClasses; ++class_id) {
    const int count = label_counts[class_id];
    if (!count) continue;
    ++metrics.classes_with_labels;
    std::vector<PredictionStat> class_stats;
    for (const auto& stat : stats) if (stat.class_id == class_id) class_stats.push_back(stat);
    std::sort(class_stats.begin(), class_stats.end(), [](const PredictionStat& a, const PredictionStat& b) { return a.score > b.score; });
    std::array<int, kIouCount> true_positives{};
    std::array<int, kIouCount> false_positives{};
    std::array<std::vector<float>, kIouCount> recalls;
    std::array<std::vector<float>, kIouCount> precisions;
    for (const auto& stat : class_stats) {
      for (int index = 0; index < kIouCount; ++index) {
        if (stat.correct[index]) ++true_positives[index]; else ++false_positives[index];
        recalls[index].push_back(static_cast<float>(true_positives[index]) / count);
        precisions[index].push_back(static_cast<float>(true_positives[index]) / (true_positives[index] + false_positives[index]));
      }
    }
    for (int index = 0; index < kIouCount; ++index) {
      const float ap = compute_ap(recalls[index], precisions[index]);
      sum_ap += ap;
      if (index == 0) sum_ap50 += ap;
    }
    int tp = 0;
    int fp = 0;
    for (const auto& stat : class_stats) {
      if (stat.score < kPrecisionRecallThreshold) continue;
      if (stat.correct[0]) ++tp; else ++fp;
    }
    sum_precision += static_cast<float>(tp) / (tp + fp + 1e-16f);
    sum_recall += static_cast<float>(tp) / count;
  }
  if (metrics.classes_with_labels) {
    metrics.map50 = sum_ap50 / metrics.classes_with_labels;
    metrics.map50_95 = sum_ap / (metrics.classes_with_labels * kIouCount);
    metrics.precision = sum_precision / metrics.classes_with_labels;
    metrics.recall = sum_recall / metrics.classes_with_labels;
  }
  return metrics;
}

void write_predictions(std::ofstream& output, const std::string& image, const std::vector<Detection>& detections, bool first) {
  if (!first) output << ",\n";
  output << "  {\"image\":\"" << image << "\",\"detections\":[";
  for (size_t index = 0; index < detections.size(); ++index) {
    const Detection& det = detections[index];
    if (index) output << ',';
    output << "{\"class\":" << det.class_id << ",\"score\":" << det.score << ",\"box\":["
           << det.box.x1 << ',' << det.box.y1 << ',' << det.box.x2 << ',' << det.box.y2 << "]}";
  }
  output << "]}";
}

Metrics evaluate_model(const std::string& name, const std::string& model_path, bool skip_sigmoid,
                       const std::string& image_dir, const std::string& label_dir,
                       const std::vector<std::string>& images, const std::string& prediction_path) {
  RknnYolo model(model_path, skip_sigmoid);
  std::array<int, kClasses> label_counts{};
  std::vector<PredictionStat> stats;
  int label_total = 0;
  std::ofstream predictions(prediction_path);
  if (!predictions) throw std::runtime_error("Cannot write predictions: " + prediction_path);
  predictions << "[\n";
  for (size_t index = 0; index < images.size(); ++index) {
    const std::string& image_name = images[index];
    cv::Mat image = cv::imread(image_dir + "/" + image_name);
    if (image.empty()) throw std::runtime_error("Cannot read image: " + image_name);
    const std::string stem = image_name.substr(0, image_name.size() - 4);
    const std::vector<Label> labels = load_labels(label_dir + "/" + stem + ".txt", model.width(), model.height());
    for (const Label& label : labels) {
      if (label.class_id >= 0 && label.class_id < kClasses) ++label_counts[label.class_id];
    }
    label_total += static_cast<int>(labels.size());
    const std::vector<Detection> detections = model.infer(image);
    match_predictions(detections, labels, stats);
    write_predictions(predictions, image_name, detections, index == 0);
    if ((index + 1) % 16 == 0 || index + 1 == images.size()) {
      std::cout << name << ": " << index + 1 << "/" << images.size() << std::endl;
    }
  }
  predictions << "\n]\n";
  return summarize(stats, label_counts, static_cast<int>(images.size()), label_total);
}

void print_metrics(const std::string& name, const Metrics& metrics) {
  std::cout << name << " mAP50=" << metrics.map50 << " mAP50-95=" << metrics.map50_95
            << " P@0.25=" << metrics.precision << " R@0.25=" << metrics.recall
            << " images=" << metrics.images << " labels=" << metrics.labels << std::endl;
}

void write_report(const std::string& path, const Metrics& silu, const Metrics& relu) {
  std::ofstream output(path);
  if (!output) throw std::runtime_error("Cannot write report: " + path);
  output << "{\n  \"evaluation\": \"RK3588 RKNN Runtime offline YOLOv5 evaluation\",\n"
         << "  \"images\": " << silu.images << ",\n  \"candidate_threshold\": " << kCandidateThreshold
         << ",\n  \"nms_threshold\": " << kNmsThreshold << ",\n  \"precision_recall_threshold\": " << kPrecisionRecallThreshold << ",\n"
         << "  \"models\": [\n";
  const auto write_model = [&output](const char* name, const Metrics& metrics, bool comma) {
    output << "    {\"name\":\"" << name << "\",\"mAP50\":" << metrics.map50
           << ",\"mAP50_95\":" << metrics.map50_95 << ",\"precision_at_025\":" << metrics.precision
           << ",\"recall_at_025\":" << metrics.recall << ",\"images\":" << metrics.images
           << ",\"labels\":" << metrics.labels << ",\"detections\":" << metrics.detections
           << ",\"classes_with_labels\":" << metrics.classes_with_labels << "}" << (comma ? "," : "") << "\n";
  };
  write_model("silu", silu, true);
  write_model("relu", relu, false);
  output << "  ]\n}\n";
}

std::string require_value(int argc, char** argv, const std::string& name) {
  for (int index = 1; index + 1 < argc; ++index) if (argv[index] == name) return argv[index + 1];
  throw std::runtime_error("Missing argument " + name);
}

std::string optional_value(int argc, char** argv, const std::string& name, const std::string& fallback) {
  for (int index = 1; index + 1 < argc; ++index) if (argv[index] == name) return argv[index + 1];
  return fallback;
}

bool has_flag(int argc, char** argv, const std::string& name) {
  for (int index = 1; index < argc; ++index) if (argv[index] == name) return true;
  return false;
}

std::vector<std::string> load_class_names(const std::string& path) {
  std::ifstream file(path);
  std::vector<std::string> names;
  std::string line;
  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    names.push_back(line);
  }
  return names;
}

int run_video(int argc, char** argv) {
  const std::string video_path = require_value(argc, argv, "--video");
  const std::string model_path = require_value(argc, argv, "--model");
  const std::string output_path = require_value(argc, argv, "--output");
  const std::string labels_path = require_value(argc, argv, "--labels");
  const bool skip_sigmoid = has_flag(argc, argv, "--skip-sigmoid");
  const float display_threshold = std::stof(optional_value(argc, argv, "--display-threshold", "0.6"));
  const std::vector<std::string> class_names = load_class_names(labels_path);

  cv::VideoCapture capture(video_path);
  if (!capture.isOpened()) throw std::runtime_error("Cannot open video: " + video_path);
  const int width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
  const int height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
  double fps = capture.get(cv::CAP_PROP_FPS);
  if (fps < 1.0) fps = 25.0;
  cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(width, height));
  if (!writer.isOpened()) throw std::runtime_error("Cannot create output video: " + output_path);

  RknnYolo model(model_path, skip_sigmoid);
  cv::Mat frame;
  int frames = 0;
  while (capture.read(frame)) {
    const std::vector<Detection> detections = model.infer(frame);
    const float scale_x = static_cast<float>(frame.cols) / model.width();
    const float scale_y = static_cast<float>(frame.rows) / model.height();
    for (const Detection& detection : detections) {
      if (detection.score < display_threshold) continue;
      const cv::Point p1(std::max(0, static_cast<int>(detection.box.x1 * scale_x)), std::max(0, static_cast<int>(detection.box.y1 * scale_y)));
      const cv::Point p2(std::min(frame.cols - 1, static_cast<int>(detection.box.x2 * scale_x)), std::min(frame.rows - 1, static_cast<int>(detection.box.y2 * scale_y)));
      cv::rectangle(frame, p1, p2, cv::Scalar(0, 255, 0), 2);
      const std::string name = detection.class_id >= 0 && detection.class_id < static_cast<int>(class_names.size())
          ? class_names[detection.class_id] : std::to_string(detection.class_id);
      std::ostringstream text;
      text.setf(std::ios::fixed);
      text.precision(2);
      text << name << " " << detection.score;
      cv::putText(frame, text.str(), cv::Point(p1.x, std::max(20, p1.y - 6)), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    }
    writer.write(frame);
    ++frames;
    if (frames % 50 == 0) std::cout << "video: " << frames << " frames" << std::endl;
  }
  writer.release();
  std::cout << "video complete: " << frames << " frames, display_threshold=" << display_threshold << " -> " << output_path << std::endl;
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (has_flag(argc, argv, "--video")) return run_video(argc, argv);
    const std::string dataset = require_value(argc, argv, "--dataset");
    const std::string exclude_list = require_value(argc, argv, "--exclude-list");
    const std::string silu_model = require_value(argc, argv, "--silu-model");
    const std::string relu_model = require_value(argc, argv, "--relu-model");
    const std::string output_dir = require_value(argc, argv, "--output-dir");
    const std::set<std::string> excluded = load_excluded(exclude_list);
    std::vector<std::string> images;
    for (const std::string& image : list_jpgs(dataset + "/images/train2017")) {
      if (!excluded.count(image)) images.push_back(image);
    }
    if (excluded.size() != 50 || images.size() != 78) throw std::runtime_error("Expected 50 excluded and 78 evaluated images");
    const std::string image_dir = dataset + "/images/train2017";
    const std::string label_dir = dataset + "/labels/train2017";
    const Metrics silu = evaluate_model("silu", silu_model, false, image_dir, label_dir, images, output_dir + "/silu_predictions.json");
    const Metrics relu = evaluate_model("relu", relu_model, true, image_dir, label_dir, images, output_dir + "/relu_predictions.json");
    print_metrics("silu", silu);
    print_metrics("relu", relu);
    write_report(output_dir + "/board_accuracy_metrics.json", silu, relu);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "rknn_eval error: " << error.what() << std::endl;
    return 1;
  }
}

