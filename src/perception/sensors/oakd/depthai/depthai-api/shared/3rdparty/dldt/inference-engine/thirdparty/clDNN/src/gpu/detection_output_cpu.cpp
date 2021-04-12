/*
// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "detection_output_inst.h"
#include "kernel.h"
#include "network_impl.h"
#include "implementation_map.h"
#include "math_utils.h"
#include "cpu_impl_helpers.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <xmmintrin.h>
#include <vector>
#include <utility>

#ifdef FIX_OPENMP_RELEASE_ISSUE
#ifdef OPENMP_FOUND
#include <omp.h>
#endif
#endif

namespace cldnn {
namespace gpu {

namespace {
    using bounding_box = cldnn::cpu::bounding_box;
}  // namespace

/************************ Detection Output CPU ************************/
struct detection_output_cpu : typed_primitive_impl<detection_output> {
    const detection_output_node& outer;

    explicit detection_output_cpu(const detection_output_node& outer) : outer(outer) {}

    static void decode_bounding_box(const bounding_box& prior_bbox,
                                    const std::array<float, PRIOR_BOX_SIZE>& prior_variance,
                                    const prior_box_code_type code_type,
                                    const bool variance_encoded_in_target,
                                    const bounding_box& bbox,
                                    bounding_box* decoded_bbox,
                                    const bool prior_is_normalized,
                                    const size_t image_width,
                                    const size_t image_height,
                                    const bool clip_before_nms) {
        float prior_bbox_xmin = prior_bbox.xmin;
        float prior_bbox_ymin = prior_bbox.ymin;
        float prior_bbox_xmax = prior_bbox.xmax;
        float prior_bbox_ymax = prior_bbox.ymax;

        float bbox_xmin = bbox.xmin;
        float bbox_ymin = bbox.ymin;
        float bbox_xmax = bbox.xmax;
        float bbox_ymax = bbox.ymax;

        if (!prior_is_normalized) {
            prior_bbox_xmin /= image_width;
            prior_bbox_ymin /= image_height;
            prior_bbox_xmax /= image_width;
            prior_bbox_ymax /= image_height;
        }

        switch (code_type) {
            case prior_box_code_type::corner: {
                if (variance_encoded_in_target) {
                    // variance is encoded in target, we simply need to add the offset predictions.
                    decoded_bbox->xmin = prior_bbox_xmin + bbox_xmin;
                    decoded_bbox->ymin = prior_bbox_ymin + bbox_ymin;
                    decoded_bbox->xmax = prior_bbox_xmax + bbox_xmax;
                    decoded_bbox->ymax = prior_bbox_ymax + bbox_ymax;
                } else {
                    // variance is encoded in bbox, we need to scale the offset accordingly.
                    decoded_bbox->xmin = prior_bbox_xmin + prior_variance[0] * bbox_xmin;
                    decoded_bbox->ymin = prior_bbox_ymin + prior_variance[1] * bbox_ymin;
                    decoded_bbox->xmax = prior_bbox_xmax + prior_variance[2] * bbox_xmax;
                    decoded_bbox->ymax = prior_bbox_ymax + prior_variance[3] * bbox_ymax;
                }
                break;
            }
            case prior_box_code_type::center_size: {
                const float prior_width = prior_bbox_xmax - prior_bbox_xmin;
                assert(prior_width > 0);
                const float prior_height = prior_bbox_ymax - prior_bbox_ymin;
                assert(prior_height > 0);
                const float prior_center_x = (prior_bbox_xmin + prior_bbox_xmax) / 2.f;
                const float prior_center_y = (prior_bbox_ymin + prior_bbox_ymax) / 2.f;
                float decode_bbox_center_x, decode_bbox_center_y;
                float decode_bbox_width, decode_bbox_height;
                if (variance_encoded_in_target) {
                    // variance is encoded in target, we simply need to restore the offset predictions.
                    decode_bbox_center_x = bbox_xmin * prior_width + prior_center_x;
                    decode_bbox_center_y = bbox_ymin * prior_height + prior_center_y;
                    decode_bbox_width = (exp(bbox_xmax) * prior_width);
                    decode_bbox_height = (exp(bbox_ymax) * prior_height);
                } else {
                    // variance is encoded in bbox, we need to scale the offset accordingly.
                    decode_bbox_center_x = prior_variance[0] * bbox_xmin * prior_width + prior_center_x;
                    decode_bbox_center_y = prior_variance[1] * bbox_ymin * prior_height + prior_center_y;
                    decode_bbox_width = (exp(prior_variance[2] * bbox_xmax) * prior_width);
                    decode_bbox_height = (exp(prior_variance[3] * bbox_ymax) * prior_height);
                }
                decoded_bbox->xmin = decode_bbox_center_x - decode_bbox_width / 2.0f;
                decoded_bbox->ymin = decode_bbox_center_y - decode_bbox_height / 2.0f;
                decoded_bbox->xmax = decode_bbox_center_x + decode_bbox_width / 2.0f;
                decoded_bbox->ymax = decode_bbox_center_y + decode_bbox_height / 2.0f;
                break;
            }
            case prior_box_code_type::corner_size: {
                const float prior_width = prior_bbox_xmax - prior_bbox_xmin;
                assert(prior_width > 0);
                const float prior_height = prior_bbox_ymax - prior_bbox_ymin;
                assert(prior_height > 0);
                if (variance_encoded_in_target) {
                    // variance is encoded in target, we simply need to add the offset predictions.
                    decoded_bbox->xmin = prior_bbox_xmin + bbox_xmin * prior_width;
                    decoded_bbox->ymin = prior_bbox_ymin + bbox_ymin * prior_height;
                    decoded_bbox->xmax = prior_bbox_xmax + bbox_xmax * prior_width;
                    decoded_bbox->ymax = prior_bbox_ymax + bbox_ymax * prior_height;
                } else {
                    // variance is encoded in bbox, we need to scale the offset accordingly.
                    decoded_bbox->xmin = prior_bbox_xmin + prior_variance[0] * bbox_xmin * prior_width;
                    decoded_bbox->ymin = prior_bbox_ymin + prior_variance[1] * bbox_ymin * prior_height;
                    decoded_bbox->xmax = prior_bbox_xmax + prior_variance[2] * bbox_xmax * prior_width;
                    decoded_bbox->ymax = prior_bbox_ymax + prior_variance[3] * bbox_ymax * prior_height;
                }
                break;
            }
            default: {
                assert(0);
            }
        }

        if (clip_before_nms) {
            decoded_bbox->xmin = std::max(0.0f, std::min(1.0f, decoded_bbox->xmin));
            decoded_bbox->ymin = std::max(0.0f, std::min(1.0f, decoded_bbox->ymin));
            decoded_bbox->xmax = std::max(0.0f, std::min(1.0f, decoded_bbox->xmax));
            decoded_bbox->ymax = std::max(0.0f, std::min(1.0f, decoded_bbox->ymax));
        }
    }

    static void apply_nms(const std::vector<bounding_box>& bboxes,
                          std::vector<std::pair<float, int>>& scores,
                          const float nms_threshold,
                          const float eta,
                          const int top_k) {
        // Sort the scores in descending order and keep top_k scores if needed.
        if ((top_k != -1) && (static_cast<int>(scores.size()) > top_k)) {
            std::partial_sort(scores.begin(),
                              scores.begin() + top_k,
                              scores.end(),
                              [](const std::pair<float, int>& p1, const std::pair<float, int>& p2) {
                                  return (p1.first > p2.first) || (p1.first == p2.first && p1.second < p2.second);
                              });
            scores.resize(top_k);
        } else {
            std::stable_sort(
                scores.begin(),
                scores.end(),
                [](const std::pair<float, int>& p1, const std::pair<float, int>& p2) { return p1.first > p2.first; });
        }

        // NMS
        float adaptive_threshold = nms_threshold;
        int post_nms_count = 0;

        for (auto score_index : scores) {
            const int idx = score_index.second;
            bounding_box box1(bboxes[idx]);
            bool keep = true;
            for (int i = 0; i < post_nms_count; ++i) {
                if (!keep) {
                    break;
                }
                bounding_box box2(bboxes[scores[i].second]);
                bool intersecting = (box1.xmin < box2.xmax) & (box2.xmin < box1.xmax) & (box1.ymin < box2.ymax) &
                                    (box2.ymin < box1.ymax);
                float overlap = 0.0f;
                if (intersecting) {
                    const float intersect_width = std::min(box1.xmax, box2.xmax) - std::max(box1.xmin, box2.xmin);
                    const float intersect_height = std::min(box1.ymax, box2.ymax) - std::max(box1.ymin, box2.ymin);
                    const float intersect_size = intersect_width * intersect_height;
                    overlap = intersect_size / (box1.area() + box2.area() - intersect_size);
                }
                keep = (overlap <= adaptive_threshold);
            }
            if (keep) {
                scores[post_nms_count] = score_index;
                ++post_nms_count;
            }
            if (keep && eta < 1 && adaptive_threshold > 0.5) {
                adaptive_threshold *= eta;
            }
        }
        scores.resize(post_nms_count);  // scores holds only the items that were kept after the NMS.
    }

    template <typename dtype>
    void generate_detections(const detection_output_inst& instance,
                             const int num_of_images,
                             const std::vector<std::vector<std::vector<bounding_box>>>& all_bboxes,
                             std::vector<std::vector<std::vector<std::pair<float, int>>>>& confidences) {
        mem_lock<dtype> lock{instance.output_memory()};
        auto out_ptr = lock.begin();

        const auto& args = instance.argument;
        std::vector<std::vector<std::vector<std::pair<float, int>>>>
            final_detections;  // Per image -> For each label: Pair (score, prior index)
        for (int image = 0; image < num_of_images; ++image) {
            const std::vector<std::vector<bounding_box>>& bboxes_per_image = all_bboxes[image];
            std::vector<std::vector<std::pair<float, int>>>& conf_per_image = confidences[image];
            int num_det = 0;
#ifdef FIX_OPENMP_RELEASE_ISSUE
#ifdef OPENMP_FOUND
            int num_available_threads = omp_get_max_threads();
            // half available threads usage shows the best perf results for both SKL (4c8t) and APL (4c4t) for this part
            // of detection output
            int num_threads_to_use = (omp_in_parallel() == 0) ? num_available_threads / 2 : 1;
#pragma omp parallel for num_threads(num_threads_to_use) reduction(+ : num_det)
#endif
#endif
            for (int cls = 0; cls < static_cast<int>(args.num_classes); ++cls) {
                if (static_cast<int>(cls) == args.background_label_id) {
                    conf_per_image[cls].clear();
                    continue;  // Skip background class.
                }
                std::vector<std::pair<float, int>>& scores = conf_per_image[cls];
                const int label = args.share_location ? 0 : cls;
                apply_nms(bboxes_per_image[label], scores, args.nms_threshold, args.eta, args.top_k);
                num_det += static_cast<int>(scores.size());
            }
            if (num_det > args.keep_top_k) {
                std::vector<std::pair<float, std::pair<int, int>>> score_index_pairs;
                score_index_pairs.reserve(num_det);
                for (int label = 0; label < static_cast<int>(args.num_classes); ++label) {
                    std::vector<std::pair<float, int>>& scores = confidences[image][label];
                    for (std::pair<float, int> score_index : scores) {
                        score_index_pairs.emplace_back(score_index.first, std::make_pair(label, score_index.second));
                    }
                }

                // Keep top k results per image.
                auto sort_function = [](const std::pair<float, std::pair<int, int>>& p1,
                                        const std::pair<float, std::pair<int, int>>& p2) {
                    return p1.first > p2.first;
                };
                if (static_cast<int>(score_index_pairs.size()) > args.keep_top_k) {
                    std::partial_sort(score_index_pairs.begin(),
                                      score_index_pairs.begin() + args.keep_top_k,
                                      score_index_pairs.end(),
                                      sort_function);
                    score_index_pairs.resize(args.keep_top_k);
                } else {
                    std::sort(score_index_pairs.begin(), score_index_pairs.end(), sort_function);
                }

                // Store the new indices.
                std::vector<std::vector<std::pair<float, int>>> new_indices(args.num_classes);
                for (int j = 0; j < static_cast<int>(score_index_pairs.size()); ++j) {
                    int label = score_index_pairs[j].second.first;
                    int idx = score_index_pairs[j].second.second;
                    new_indices[label].emplace_back(score_index_pairs[j].first, idx);
                }
                final_detections.emplace_back(new_indices);
            } else {
                final_detections.emplace_back(confidences[image]);
            }
        }

        int count = 0;
        for (int image = 0; image < num_of_images; ++image) {
            const std::vector<std::vector<bounding_box>>& bboxes_per_image = all_bboxes[image];
            auto& final_detections_per_image = final_detections[image];
            for (int label = 0; label < static_cast<int>(final_detections_per_image.size()); ++label) {
                int loc_label = args.share_location ? 0 : label;
                const std::vector<bounding_box>& bboxes = bboxes_per_image[loc_label];
                const std::vector<std::pair<float, int>>& label_detections = final_detections_per_image[label];
                for (std::pair<float, int> score_prior : label_detections) {
                    out_ptr[count * DETECTION_OUTPUT_ROW_SIZE] = (dtype)static_cast<float>(image);
                    out_ptr[count * DETECTION_OUTPUT_ROW_SIZE + 1] =
                        args.decrease_label_id ? ((dtype)(static_cast<float>(label - 1.0f))) : (dtype)static_cast<float>(label);
                    out_ptr[count * DETECTION_OUTPUT_ROW_SIZE + 2] = (dtype)score_prior.first;
                    const bounding_box& bbox = bboxes[score_prior.second];
                    float xmin = bbox.xmin;
                    float ymin = bbox.ymin;
                    float xmax = bbox.xmax;
                    float ymax = bbox.ymax;

                    if (args.clip_after_nms) {
                        xmin = std::max(0.0f, std::min(1.0f, xmin));
                        ymin = std::max(0.0f, std::min(1.0f, ymin));
                        xmax = std::max(0.0f, std::min(1.0f, xmax));
                        ymax = std::max(0.0f, std::min(1.0f, ymax));
                    }

                    out_ptr[count * DETECTION_OUTPUT_ROW_SIZE + 3] = (dtype)xmin;
                    out_ptr[count * DETECTION_OUTPUT_ROW_SIZE + 4] = (dtype)ymin;
                    out_ptr[count * DETECTION_OUTPUT_ROW_SIZE + 5] = (dtype)xmax;
                    out_ptr[count * DETECTION_OUTPUT_ROW_SIZE + 6] = (dtype)ymax;
                    ++count;
                }
            }
        }

        // In case number of detections is smaller than keep_top_k fill the rest of the buffer with invalid image id
        // (-1).
        while (count < num_of_images * args.keep_top_k) {
            out_ptr[count * DETECTION_OUTPUT_ROW_SIZE] = (dtype)-1.f;
            out_ptr[count * DETECTION_OUTPUT_ROW_SIZE + 1] = (dtype)0.f;
            out_ptr[count * DETECTION_OUTPUT_ROW_SIZE + 2] = (dtype)0.f;
            out_ptr[count * DETECTION_OUTPUT_ROW_SIZE + 3] = (dtype)0.f;
            out_ptr[count * DETECTION_OUTPUT_ROW_SIZE + 4] = (dtype)0.f;
            out_ptr[count * DETECTION_OUTPUT_ROW_SIZE + 5] = (dtype)0.f;
            out_ptr[count * DETECTION_OUTPUT_ROW_SIZE + 6] = (dtype)0.f;
            ++count;
        }
    }

    // Compute the linear index taking the padding into account.
    static inline int get_linear_feature_index(const int batch_id,
                                               const int feature_id,
                                               const int input_buffer_size_f,
                                               const int input_buffer_size_y,
                                               const int input_buffer_size_x,
                                               const int input_padding_lower_y,
                                               const int input_padding_lower_x) {
        // This helper function assumes input layout with x_size = 1 and y_size = 1;
        // Location and confidence inputs should be tensors with size {b,f,1,1}.
        // This is validated in detection output primitive instance creation.

        int input_idx = (batch_id * input_buffer_size_f + feature_id) * input_buffer_size_y * input_buffer_size_x;
        input_idx += input_padding_lower_y * input_buffer_size_x + input_padding_lower_x;

        return input_idx;
    }

    template <typename dtype>
    void extract_locations_per_image(const detection_output_inst& instance,
                                     std::vector<std::vector<std::vector<bounding_box>>>& locations,
                                     const int num_of_priors,
                                     const int num_loc_classes) {
        const bool share_location = instance.argument.share_location;
        auto& input_location = instance.location_memory();
        const int num_of_images = static_cast<int>(locations.size());

        mem_lock<dtype> lock{input_location};
        auto location_data = lock.begin();

        assert(num_of_priors * num_loc_classes * PRIOR_BOX_SIZE == input_location.get_layout().size.feature[0]);

        const auto& input_buffer_size = input_location.get_layout().get_buffer_size();
        const int input_buffer_size_x = input_buffer_size.spatial[0];
        const int input_buffer_size_y = input_buffer_size.spatial[1];
        const int input_buffer_size_f = input_buffer_size.feature[0];
        const auto& input_padding = input_location.get_layout().data_padding;
        const int input_padding_lower_x = input_padding.lower_size().spatial[0];
        const int input_padding_lower_y = input_padding.lower_size().spatial[1];

        for (int image = 0; image < num_of_images; ++image) {
            std::vector<std::vector<bounding_box>>& label_to_bbox = locations[image];
            label_to_bbox.resize(num_loc_classes);
            for (int cls = 0; cls < num_loc_classes; ++cls) {
                int label = share_location ? 0 : cls;
                auto& bboxes = label_to_bbox[label];
                bboxes.resize(num_of_priors);

                for (int prior = 0; prior < num_of_priors; ++prior) {
                    int idx = prior * num_loc_classes * PRIOR_BOX_SIZE;
                    bboxes[prior].xmin = static_cast<float>((location_data[get_linear_feature_index(image,
                                                                                        idx + cls * PRIOR_BOX_SIZE,
                                                                                        input_buffer_size_f,
                                                                                        input_buffer_size_y,
                                                                                        input_buffer_size_x,
                                                                                        input_padding_lower_y,
                                                                                        input_padding_lower_x)]));
                    bboxes[prior].ymin = static_cast<float>((location_data[get_linear_feature_index(image,
                                                                                        idx + cls * PRIOR_BOX_SIZE + 1,
                                                                                        input_buffer_size_f,
                                                                                        input_buffer_size_y,
                                                                                        input_buffer_size_x,
                                                                                        input_padding_lower_y,
                                                                                        input_padding_lower_x)]));
                    bboxes[prior].xmax = static_cast<float>((location_data[get_linear_feature_index(image,
                                                                                        idx + cls * PRIOR_BOX_SIZE + 2,
                                                                                        input_buffer_size_f,
                                                                                        input_buffer_size_y,
                                                                                        input_buffer_size_x,
                                                                                        input_padding_lower_y,
                                                                                        input_padding_lower_x)]));
                    bboxes[prior].ymax = static_cast<float>((location_data[get_linear_feature_index(image,
                                                                                        idx + cls * PRIOR_BOX_SIZE + 3,
                                                                                        input_buffer_size_f,
                                                                                        input_buffer_size_y,
                                                                                        input_buffer_size_x,
                                                                                        input_padding_lower_y,
                                                                                        input_padding_lower_x)]));
                }
            }
        }
    }

    template <typename dtype>
    void extract_prior_boxes_and_variances(const detection_output_inst& instance,
                                           const bool variance_encoded_in_target,
                                           const int32_t prior_info_size,
                                           const int32_t prior_coordinates_offset,
                                           const int32_t images_count,
                                           std::vector<bounding_box>& prior_bboxes,
                                           std::vector<std::array<float, PRIOR_BOX_SIZE>>& prior_variances) {
        auto& input_prior_box = instance.prior_box_memory();
        const int num_of_priors = static_cast<int>(prior_bboxes.size()) / images_count;

        mem_lock<dtype> lock{input_prior_box};
        for (int i = 0; i < images_count; i++) {
            auto prior_box_data =
                lock.begin() + i * num_of_priors * prior_info_size * (variance_encoded_in_target ? 1 : 2);

            for (int prior = 0; prior < num_of_priors; ++prior) {
                int idx = prior * prior_info_size + prior_coordinates_offset;
                prior_bboxes[i * num_of_priors + prior] = bounding_box(static_cast<float>(prior_box_data[idx]),
                                                                       static_cast<float>(prior_box_data[idx + 1]),
                                                                       static_cast<float>(prior_box_data[idx + 2]),
                                                                       static_cast<float>(prior_box_data[idx + 3]));
                idx += num_of_priors * prior_info_size;
                for (int j = 0; j < PRIOR_BOX_SIZE; ++j) {
                    prior_variances[i * num_of_priors + prior][j] =
                        variance_encoded_in_target ? 0.0f : static_cast<float>(prior_box_data[idx + j]);
                }
            }
        }
    }

    template <typename dtype>
    void extract_confidences_per_image(const detection_output_inst& instance,
                                       std::vector<std::vector<std::vector<std::pair<float, int>>>>& confidences,
                                       const int num_of_priors) {
        const int num_classes = instance.argument.num_classes;

        const int num_of_images = static_cast<int>(confidences.size());
        auto& input_confidence = instance.confidence_memory();
        const float confidence_threshold = instance.argument.confidence_threshold;

        mem_lock<dtype> lock{(memory_impl::ptr) &input_confidence};
        auto confidence_data = lock.begin();

        assert(num_of_priors * num_classes == input_confidence.get_layout().size.feature[0]);

        const auto& input_buffer_size = input_confidence.get_layout().get_buffer_size();
        const int input_buffer_size_x = input_buffer_size.spatial[0];
        const int input_buffer_size_y = input_buffer_size.spatial[1];
        const int input_buffer_size_f = input_buffer_size.feature[0];
        const auto& input_padding = input_confidence.get_layout().data_padding;
        const int input_padding_lower_x = input_padding.lower_size().spatial[0];
        const int input_padding_lower_y = input_padding.lower_size().spatial[1];
        const int stride = input_buffer_size_y * input_buffer_size_x;

        for (int image = 0; image < num_of_images; ++image) {
            std::vector<std::vector<std::pair<float, int>>>& label_to_scores = confidences[image];
            label_to_scores.resize(num_classes);
            int idx = get_linear_feature_index(image,
                                               0,
                                               input_buffer_size_f,
                                               input_buffer_size_y,
                                               input_buffer_size_x,
                                               input_padding_lower_y,
                                               input_padding_lower_x);

            if (stride == 1 && std::is_same<dtype, float>::value) {
                float const* confidence_ptr_float = (float const*)(&(*confidence_data));
                confidence_ptr_float += idx;
                __m128 threshold = _mm_load_ps1(&confidence_threshold);
                for (int prior = 0; prior < num_of_priors; ++prior) {
                    int cls = 0;
                    for (; cls + 3 < num_classes; cls += 4) {
                        __m128 scores = _mm_loadu_ps(confidence_ptr_float);
                        confidence_ptr_float += 4;
                        __m128i mask128 = _mm_castps_si128(_mm_cmpgt_ps(scores, threshold));
                        if (_mm_testz_si128(mask128, mask128)) {
                            continue;
                        }
                        int mask = _mm_movemask_ps(_mm_castsi128_ps(mask128));
                        if (mask & 1) {
                            label_to_scores[cls + 0].emplace_back(_mm_cvtss_f32(scores), prior);
                        }
                        if (mask & 2) {
                            int score = _mm_extract_ps(scores, 1);
                            float s = reinterpret_cast<float&>(score);
                            label_to_scores[cls + 1].emplace_back(s, prior);
                        }
                        if (mask & 4) {
                            int score = _mm_extract_ps(scores, 2);
                            float s = reinterpret_cast<float&>(score);
                            label_to_scores[cls + 2].emplace_back(s, prior);
                        }
                        if (mask & 8) {
                            int score = _mm_extract_ps(scores, 3);
                            float s = reinterpret_cast<float&>(score);
                            label_to_scores[cls + 3].emplace_back(s, prior);
                        }
                    }
                    for (; cls < num_classes; ++cls) {
                        float score = *confidence_ptr_float;
                        if (score > confidence_threshold) {
                            label_to_scores[cls].emplace_back(score, prior);
                        }
                        ++confidence_ptr_float;
                    }
                }
            } else {
                for (int prior = 0; prior < num_of_priors; ++prior) {
                    for (int cls = 0; cls < num_classes; ++cls) {
                        float score = static_cast<float>(confidence_data[idx]);
                        if (score > confidence_threshold) {
                            label_to_scores[cls].emplace_back(score, prior);
                        }
                        idx += stride;
                    }
                }
            }
        }
    }

    template <typename dtype>
    void prepare_data(const detection_output_inst& instance,
                      std::vector<std::vector<std::vector<bounding_box>>>& bboxes,
                      std::vector<std::vector<std::vector<std::pair<float, int>>>>& confidences) {
        assert(bboxes.size() == confidences.size());

        const auto& args = instance.argument;

        const int num_of_images = static_cast<int>(bboxes.size());
        const int num_of_priors = instance.prior_box_memory().get_layout().size.spatial[1] / args.prior_info_size;
        const int num_loc_classes = args.share_location ? 1 : args.num_classes;

        // Extract locations per image.
        std::vector<std::vector<std::vector<bounding_box>>> locations(
            num_of_images);  // Per image : label -> bounding boxes.
        extract_locations_per_image<dtype>(instance, locations, num_of_priors, num_loc_classes);

        int32_t batches_in_prior_boxes = instance.prior_box_memory().get_layout().size.batch[0];
        std::vector<bounding_box> prior_bboxes(batches_in_prior_boxes *
                                               num_of_priors);  // Prior-Boxes (identical for all images since we assume
                                                                // all images in a batch are of same dimension).
        std::vector<std::array<float, PRIOR_BOX_SIZE>> prior_variances(
            batches_in_prior_boxes * num_of_priors);  // Variances per prior-box (identical for all images since we
                                                      // assume all images in a batch are of same dimension).
        extract_prior_boxes_and_variances<dtype>(instance,
                                                 args.variance_encoded_in_target,
                                                 args.prior_info_size,
                                                 args.prior_coordinates_offset,
                                                 batches_in_prior_boxes,
                                                 prior_bboxes,
                                                 prior_variances);

        // Create the decoded bounding boxes according to locations predictions and prior-boxes.
        for (int image = 0; image < num_of_images; ++image) {
            std::vector<std::vector<bounding_box>>& bboxes_per_image = bboxes[image];
            bboxes_per_image.resize(num_loc_classes);
            locations[image].resize(num_loc_classes);
            for (int cls = 0; cls < num_loc_classes; ++cls) {
                const int label = args.share_location ? 0 : cls;
                if (!args.share_location && label == args.background_label_id) {
                    continue;  // Skip background class.
                }
                const std::vector<bounding_box>& label_loc_preds = locations[image][label];
                int label_loc_preds_size = static_cast<int>(label_loc_preds.size());

                bboxes_per_image[label].clear();

                for (int i = 0; i < label_loc_preds_size; ++i) {
                    bounding_box decoded_bbox;
                    int32_t pb_offset = (batches_in_prior_boxes > 1) ? (image * num_of_priors + i) : i;
                    int32_t var_offset = (batches_in_prior_boxes > 1) ? (image * num_of_priors + i) : i;
                    decode_bounding_box(prior_bboxes[pb_offset],
                                        prior_variances[var_offset],
                                        args.code_type,
                                        args.variance_encoded_in_target,
                                        label_loc_preds[i],
                                        &decoded_bbox,
                                        args.prior_is_normalized,
                                        args.input_width,
                                        args.input_height,
                                        args.clip_before_nms);
                    bboxes_per_image[label].emplace_back(decoded_bbox);
                }
            }
        }

        // Extract confidences per image.
        extract_confidences_per_image<dtype>(instance, confidences, num_of_priors);
    }

    event_impl::ptr execute_impl(const std::vector<event_impl::ptr>& events, detection_output_inst& instance) override {
        for (auto& a : events) {
            a->wait();
        }

        auto ev = instance.get_network().get_engine().create_user_event(instance.get_network().get_id(), false);

        const int num_of_images = instance.location_memory().get_layout().size.batch[0];  // batch size

        std::vector<std::vector<std::vector<bounding_box>>> bboxes(
            num_of_images);  // Per image : label -> decoded bounding boxes.
        std::vector<std::vector<std::vector<std::pair<float, int>>>> confidences(
            num_of_images);  // Per image : class -> confidences per bounding box.

        if (instance.location_memory().get_layout().data_type == data_types::f32) {
            prepare_data<data_type_to_type<data_types::f32>::type>(instance, bboxes, confidences);

            generate_detections<data_type_to_type<data_types::f32>::type>(instance, num_of_images, bboxes, confidences);
        } else {
            prepare_data<data_type_to_type<data_types::f16>::type>(instance, bboxes, confidences);

            generate_detections<data_type_to_type<data_types::f16>::type>(instance, num_of_images, bboxes, confidences);
        }

        dynamic_cast<cldnn::user_event*>(ev.get())->set();  // set as complete
        // TODO: consider refactoring create_user_event() to return cldnn::user_event*
        return ev;
    }

    static primitive_impl* create(const detection_output_node& arg) { return new detection_output_cpu(arg); }
};

primitive_impl* runDetectOutCpu(const detection_output_node& arg) { return new detection_output_cpu(arg); }

}  // namespace gpu
}  // namespace cldnn
