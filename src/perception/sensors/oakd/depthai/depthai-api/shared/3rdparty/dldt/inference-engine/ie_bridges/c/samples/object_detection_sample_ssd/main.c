// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier : Apache-2.0
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <sys/stat.h>
#include <c_api/ie_c_api.h>
#include "object_detection_sample_ssd.h"
#include <opencv_c_wraper.h>

#ifdef _WIN32
#include "c_w_dirent.h"
#else
#include <dirent.h>
#endif

#define MAX_IMAGE 20

static const char *img_msg = NULL;                                                                                                                                                                        
static const char *input_model = NULL;
static const char *device_name = "CPU";
static const char *custom_cldnn_msg = NULL;
static const char *custom_cpu_library_msg = NULL;
static const char *config_msg = NULL;
static int file_num = 0;
static char **file_paths = NULL;

const char *info = "[ INFO ] ";
const char *warn = "[ WARNING ] ";

int ParseAndCheckCommandLine(int argc, char *argv[]) {
    int opt = 0;
    int help = 0;
    char *string = "hi:m:d:c:l:g:";

    printf("%sParsing input parameters\n", info);

    while ((opt = getopt(argc, argv, string)) != -1) {
        switch(opt) {
            case 'h':
                showUsage();
                help = 1;
                break;
            case 'i':
                img_msg = optarg;
                break;
            case 'm':
                input_model = optarg;
                break;
            case 'd':
                device_name = optarg;
                break;
            case 'c':
                custom_cldnn_msg = optarg;
                break;
            case 'l':
                custom_cpu_library_msg = optarg;
                break;
            case 'f':
                config_msg = optarg;
                break;
            default:
            return -1;
        }
    }

    if (help)
        return -1;
    if (img_msg == NULL) {
        printf("Parameter -i is not set\n");
        return -1;
    }
    if (input_model == NULL) {
        printf("Parameter -m is not set \n");
        return -1;
    }

    return 1;
}

/**
* @brief This function checks input args and existence of specified files in a given folder. Updated the file_paths and file_num.
* @param arg path to a file to be checked for existence
* @return none.
*/
void readInputFilesArgument(const char *arg) {
    struct stat sb;
    int i;
    if (stat(arg, &sb) != 0) {
        printf("%sFile %s cannot be opened!\n", warn, arg);
        return;
    }
    if (S_ISDIR(sb.st_mode)) {
        DIR *dp;
        dp = opendir(arg);
        if (dp == NULL) {
            printf("%sFile %s cannot be opened!\n", warn, arg);
            return;
        }

        struct dirent *ep;
        while (NULL != (ep = readdir(dp))) {
            const char *fileName = ep->d_name;
            if (strcmp(fileName, ".") == 0 || strcmp(fileName, "..") == 0) continue;
            char *file_path = (char *)malloc(strlen(arg) + strlen(ep->d_name) + 1);
            strcpy(file_path, arg);
            strcat(file_path, "/");
            strcat(file_path, ep->d_name);

            if (file_num == 0) {
                file_paths = (char **)malloc(sizeof(char *));
                file_paths[0] = file_path;
                ++file_num;
            } else {
                char **temp = (char **)realloc(file_paths, sizeof(char *) * (file_num +1));
                if (temp) {
                    file_paths = temp;
                    file_paths[file_num++] = file_path;
                } else {
                    for (i = 0; i < file_num; ++i) {
                        free(file_paths[i]);
                    }
                    free(file_paths);
                    file_num = 0;
                }
            }
        }
        closedir(dp);
        dp = NULL;
    } else {
        char *file_path = malloc(strlen(arg));
        strcpy(file_path, arg);
        if (file_num == 0) {
            file_paths = (char **)malloc(sizeof(char *));
        }
        file_paths[file_num++] = file_path;
    }

    if (file_num) {
        printf("%sFiles were added: %d\n", info, file_num);
        for (i = 0; i < file_num; ++i) {
            printf("%s    %s\n", info, file_paths[i]);
        }
    } else {
        printf("%sFiles were added: %d. Too many to display each of them.\n", info, file_num);
    }
}

/**
* @brief This function find -i key in input args. It's necessary to process multiple values for single key
* @return none.
*/
void parseInputFilesArguments(int argc, char **argv) {
    int readArguments = 0, i;
    for (i = 0; i < argc; ++i) {
        if (strcmp(argv[i], "-i") == 0) {
            readArguments = 1;
            continue;
        }
        if (!readArguments) {
            continue;
        }
        if (argv[i][0] == '-') {
            break;
        }
        readInputFilesArgument(argv[i]);
    }
}

/**
* @brief Convert the contents of configuration file to the ie_config_t type.
* @param config_file File path.
* @param comment Separator symbol.
* @return A pointer to the ie_config_t instance.
*/
ie_config_t *parseConfig(const char *config_file, char comment) {
    FILE *file = fopen(config_file, "r");
    if (!file) {
        return NULL;
    }

    ie_config_t *cfg = NULL;
    char key[256], value[256];
    
    if (fscanf(file, "%s", key)!= EOF && fscanf(file, "%s", value) != EOF) {
        char *cfg_name = (char *)malloc(strlen(key));
        char *cfg_value = (char *)malloc(strlen(value));
        strcpy(cfg_name, key);
        strcpy(cfg_value, value);
        ie_config_t *cfg_t = (ie_config_t *)malloc(sizeof(ie_config_t));
        cfg_t->name = cfg_name;
        cfg_t->value = cfg_value;
        cfg_t->next = NULL;
        cfg = cfg_t;
    }
    if (cfg) {
        ie_config_t *cfg_temp = cfg;
        while (fscanf(file, "%s", key)!= EOF && fscanf(file, "%s", value) != EOF) {
            if (strlen(key) == 0 || key[0] == comment) {
                continue;
            }
            char *cfg_name = (char *)malloc(strlen(key));
            char *cfg_value = (char *)malloc(strlen(value));
            strcpy(cfg_name, key);
            strcpy(cfg_value, value);
            ie_config_t *cfg_t = (ie_config_t *)malloc(sizeof(ie_config_t));
            cfg_t->name = cfg_name;
            cfg_t->value = cfg_value;
            cfg_t->next = NULL;
            cfg_temp->next = cfg_t;
            cfg_temp = cfg_temp->next;
        }
    }
    
    return cfg;
}

/**
* @brief Releases memory occupied by config
* @param config A pointer to the config to free memory.
* @return none
*/
void config_free(ie_config_t *config) {
    while (config) {
        ie_config_t *temp = config;
        if (config->name) {
            free((char *)config->name);
            config->name = NULL;
        }
        if(config->value) {
            free((char *)config->value);
            config->value = NULL;
        }
        if(config->next) {
            config = config->next;
        }

        free(temp);
        temp = NULL;
    }
}

/**
* @brief Convert the numbers to char *;
* @param str A pointer to the convered string .
* @param num The number to convert.
* @return none.
*/
void int2str(char *str, int num) {
    int i = 0, j;
    if (num == 0) {
        str[0] = '0';
        str[1] = '\0';
    return;
    }
    
    while (num != 0) {
        str[i++] = num % 10 + '0';
        num = num / 10;
    }

    str[i] = '\0';
    --i;
    for (j = 0; j < i; ++j, --i) {
        char temp =  str[j];
        str[j] = str[i];
        str[i] = temp;
    }
}

int main(int argc, char **argv) {
    /** This sample covers certain topology and cannot be generalized for any object detection one **/
    printf("%sInferenceEngine: \n", info);
    printf("%s\n", ie_c_api_version());

    char **argv_temp =(char **)malloc(sizeof(char *) * argc);
    int i, j;
    for (i = 0; i < argc; ++i) {
        argv_temp[i] = argv[i];
    }
    // --------------------------- 1. Parsing and validation of input args ---------------------------------
    if (ParseAndCheckCommandLine(argc, argv) < 0) {
        return -1;
    }
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 2. Read input -----------------------------------------------------------
    /** This file_paths stores paths to the processed images **/
    parseInputFilesArguments(argc, argv_temp);
    if (!file_num) {
        printf("No suitable images were found\n");
        return -1;
    }
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 3. Load inference engine ------------------------------------------------
    printf("%sLoading Inference Engine\n", info);
    ie_core_t *core = NULL;
    IEStatusCode status = ie_core_create("", &core);
    assert(core);

    ie_core_versions_t ver;
    printf("%sDevice info: \n", info);
    ie_core_get_versions(core, device_name, &ver);
    for (i = 0; i < ver.num_vers; ++i) {
        printf("         %s\n", ver.versions[i].device_name);
        printf("         %s version ......... %zu.%zu\n", ver.versions[i].description, ver.versions[i].major, ver.versions[i].minor);
        printf("         Build ......... %s\n", ver.versions[i].build_number);
    }
    ie_core_versions_free(&ver);

    if (custom_cpu_library_msg) {
        // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
        ie_core_add_extension(core, custom_cpu_library_msg, "CPU");
        printf("%sCPU Extension loaded: %s\n", info, custom_cpu_library_msg);
    }

    if (custom_cldnn_msg) {
        // clDNN Extensions are loaded from an .xml description and OpenCL kernel files
        ie_config_t cfg = {"CONFIG_FILE", custom_cldnn_msg, NULL};
        ie_core_set_config(core, &cfg, "GPU");
        printf("%sGPU Extension loaded: %s\n", info, custom_cldnn_msg);
    }
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 4. Read IR Generated by ModelOptimizer (.xml and .bin files) ------------
    char *input_weight = (char *)malloc(strlen(input_model) + 1);
    strncpy(input_weight, input_model, strlen(input_model)-4);
    input_weight[strlen(input_model)-4] = '\0';
    strcat(input_weight, ".bin");
    printf("%sLoading network files:\n", info);
    printf("\t%s\n", input_model);
    printf("\t%s\n", input_weight);

    ie_network_t *network = NULL;
    ie_core_read_network(core, input_model, input_weight, &network);
    assert(network);
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 5. Prepare input blobs --------------------------------------------------
    printf("%sPreparing input blobs\n", info);

    /** SSD network has one input and one output **/
    size_t input_num = 0;
    status = ie_network_get_inputs_number(network, &input_num);
    if (input_num != 1 && input_num != 2) {
        printf("Sample supports topologies only with 1 or 2 inputs\n");
        return -1;
    }

    /**
     * Some networks have SSD-like output format (ending with DetectionOutput layer), but
     * having 2 inputs as Faster-RCNN: one for image and one for "image info".
     *
     * Although object_datection_sample_ssd's main task is to support clean SSD, it could score
     * the networks with two inputs as well. For such networks imInfoInputName will contain the "second" input name.
     */
    char *imageInputName = NULL, *imInfoInputName = NULL;
    size_t input_width = 0, input_height = 0;

    /** Stores input image **/

    /** Iterating over all input blobs **/
    for (i = 0; i < input_num; ++i) {
        char *name = NULL;
        ie_network_get_input_name(network, i, &name);
        dimensions_t input_dim;
        ie_network_get_input_dims(network, name, &input_dim);

        /** Working with first input tensor that stores image **/
        if(input_dim.ranks == 4) {
            imageInputName = name;
            input_height = input_dim.dims[2];
            input_width = input_dim.dims[3];

            /** Creating first input blob **/
            ie_network_set_input_precision(network, name, U8);
        } else if (input_dim.ranks == 2) {
            imInfoInputName = name;
        
            ie_network_set_input_precision(network, name, FP32);
            if(input_dim.dims[1] != 3 && input_dim.dims[1] != 6) {
                printf("Invalid input info. Should be 3 or 6 values length\n");
                return -1;
            }
        }
    }

    if (imageInputName == NULL) {
        ie_network_get_input_name(network, 0, &imageInputName);

        dimensions_t input_dim;
        ie_network_get_input_dims(network, imageInputName, &input_dim);
        input_height = input_dim.dims[2];
        input_width = input_dim.dims[3];
    }

    /** Collect images data **/
    c_mat_t *originalImages = (c_mat_t *)malloc(file_num * sizeof(c_mat_t));
    c_mat_t *images = (c_mat_t *)malloc(file_num * sizeof(c_mat_t));
    int image_num = 0;
    for (i = 0; i < file_num; ++i) {
        c_mat_t img = {NULL, 0, 0, 0, 0, 0};
        if (image_read(file_paths[i], &img) == -1) {
            printf("%sImage %s cannot be read!\n", warn, file_paths[i]);
            continue;
        }
        /** Store image data **/
        c_mat_t resized_img = {NULL, 0, 0, 0, 0, 0};
        if (input_width == img.mat_width && input_height == img.mat_height) {
            resized_img.mat_data_size = img.mat_data_size;
            resized_img.mat_channels = img.mat_channels;
            resized_img.mat_width = img.mat_width;
            resized_img.mat_height = img.mat_height;
            resized_img.mat_type = img.mat_type;
            resized_img.mat_data = malloc(resized_img.mat_data_size);
            for (j = 0; j < resized_img.mat_data_size; ++j)
                resized_img.mat_data[j] = img.mat_data[j];
        } else {
            printf("%sImage is resized from (%d, %d) to (%zu, %zu)\n", \
            warn, img.mat_width, img.mat_height, input_width, input_height);

            image_resize(&img, &resized_img, (int)input_width, (int)input_height);
        }

        if (resized_img.mat_data) {
            originalImages[image_num] = img;
            images[image_num] = resized_img;
            ++image_num;
        }
    }

    if (!image_num) {
        printf("Valid input images were not found!\n");
        return -1;
    }

    input_shapes_t shapes;
    ie_network_get_input_shapes(network, &shapes);
    shapes.shapes[0].shape.dims[0] = image_num;
    ie_network_reshape(network, shapes);
    ie_network_input_shapes_free(&shapes);

    input_shapes_t shapes2;
    ie_network_get_input_shapes(network, &shapes2);
    size_t batchSize = shapes2.shapes[0].shape.dims[0];
    ie_network_input_shapes_free(&shapes2);
    printf("%sBatch size is %zu\n", info, batchSize);
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 6. Prepare output blobs -------------------------------------------------
    printf("%sPreparing output blobs\n", info);

    size_t output_num = 0;
    ie_network_get_outputs_number(network, &output_num);

    if (!output_num) {
        printf("Can't find a DetectionOutput layer in the topology\n");
        return -1;
    }

    char *output_name = NULL;
    ie_network_get_output_name(network, output_num-1, &output_name);

    dimensions_t output_dim;
    ie_network_get_output_dims(network, output_name, &output_dim);

    if (output_dim.ranks != 4) {
        printf("Incorrect output dimensions for SSD model\n");
        return -1;
    }

    const int maxProposalCount = (int)output_dim.dims[2];
    const int objectSize = (int)output_dim.dims[3];

    if (objectSize != 7) {
        printf("Output item should have 7 as a last dimension\n");
        return -1;
    }

    /** Set the precision of output data provided by the user, should be called before load of the network to the device **/
    ie_network_set_output_precision(network, output_name, FP32);
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 7. Loading model to the device ------------------------------------------
    printf("%sLoading model to the device\n", info);
    ie_executable_network_t *exe_network = NULL;
    if (config_msg) {
        ie_config_t * config = parseConfig(config_msg, '#');
        ie_core_load_network(core, network, device_name, config, &exe_network);
        config_free(config);
    } else {
        ie_config_t cfg = {NULL, NULL, NULL};
        ie_core_load_network(core, network, device_name, &cfg, &exe_network);
    }
    assert(exe_network);

    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 8. Create infer request -------------------------------------------------
    printf("%sCreate infer request\n", info);
    ie_infer_request_t *infer_request = NULL;
    ie_exec_network_create_infer_request(exe_network, &infer_request);
    assert(infer_request);
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 9. Prepare input --------------------------------------------------------


    /** Creating input blob **/
    ie_blob_t *imageInput = NULL;
    ie_infer_request_get_blob(infer_request, imageInputName, &imageInput);
    assert(imageInput);

    /** Filling input tensor with images. First b channel, then g and r channels **/
    dimensions_t input_tensor_dims;
    ie_blob_get_dims(imageInput, &input_tensor_dims);
    size_t num_channels = input_tensor_dims.dims[1];
    size_t image_size = input_tensor_dims.dims[3] * input_tensor_dims.dims[2];

    ie_blob_buffer_t blob_buffer;
    ie_blob_get_buffer(imageInput, &blob_buffer);
    unsigned char *data = (unsigned char *)(blob_buffer.buffer);

    /** Iterate over all input images **/
    int image_id, pid, ch, k;
    for (image_id = 0; image_id < batchSize; ++image_id) {
        /** Iterate over all pixel in image (b,g,r) **/
        for (pid = 0; pid < image_size; ++pid) {
            /** Iterate over all channels **/
            for (ch = 0; ch < num_channels; ++ch) {
                /**          [images stride + channels stride + pixel id ] all in bytes            **/
                data[image_id * image_size * num_channels + ch * image_size + pid] =
                    images[image_id].mat_data[pid * num_channels + ch];
            }
        }
        image_free(&images[image_id]);
    }
    free(images);

    if (imInfoInputName != NULL) {
        ie_blob_t *input2 = NULL;
        ie_infer_request_get_blob(infer_request, imInfoInputName, &input2);

        dimensions_t imInfoDim;
        ie_blob_get_dims(input2, &imInfoDim);
        //Fill input tensor with values 
        ie_blob_buffer_t info_blob_buffer;
        ie_blob_get_buffer(input2, &info_blob_buffer);
        float *p = (float *)(info_blob_buffer.buffer);
        for (image_id = 0; image_id < batchSize; ++image_id) {
            p[image_id * imInfoDim.dims[1] + 0] = (float)input_height;
            p[image_id * imInfoDim.dims[1] + 1] = (float)input_width;
            
            for (k = 2; k < imInfoDim.dims[1]; k++) {
                p[image_id * imInfoDim.dims[1] + k] = 1.0f;  // all scale factors are set to 1.0
            }
        }
    }
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 10. Do inference ---------------------------------------------------------
    printf("%sStart inference\n", info);
    ie_infer_request_infer_async(infer_request);
    ie_infer_request_wait(infer_request, -1);
    // ----------------------------------------------------------------------------------------------------- 

    // --------------------------- 11. Process output -------------------------------------------------------
    printf("%sProcessing output blobs\n", info);

    ie_blob_t *output_blob = NULL;
    ie_infer_request_get_blob(infer_request, output_name, &output_blob);
    assert(output_blob);

    ie_blob_buffer_t output_blob_buffer;
    ie_blob_get_cbuffer(output_blob, &output_blob_buffer);
    const float* detection = (float *)(output_blob_buffer.cbuffer);

    int **classes = (int **)malloc(image_num * sizeof(int *));
    rectangle_t **boxes = (rectangle_t **)malloc(image_num * sizeof(rectangle_t *));
    int *object_num = (int *)malloc(image_num * sizeof(int));
    for ( i = 0; i < image_num; ++i) {
        classes[i] = (int *)malloc(maxProposalCount * sizeof(int));
        boxes[i] = (rectangle_t *)malloc(maxProposalCount * sizeof(rectangle_t));
        object_num[i] = 0;
    }

    /* Each detection has image_id that denotes processed image */
    int curProposal;
    for (curProposal = 0; curProposal < maxProposalCount; curProposal++) {
        image_id = (int)(detection[curProposal * objectSize + 0]);
        if (image_id < 0) {
            break;
        }

        float confidence = detection[curProposal * objectSize + 2];
        int label = (int)(detection[curProposal * objectSize + 1]);
        int xmin = (int)(detection[curProposal * objectSize + 3] * originalImages[image_id].mat_width);
        int ymin = (int)(detection[curProposal * objectSize + 4] * originalImages[image_id].mat_height);
        int xmax = (int)(detection[curProposal * objectSize + 5] * originalImages[image_id].mat_width);
        int ymax = (int)(detection[curProposal * objectSize + 6] * originalImages[image_id].mat_height);

        printf("[%d, %d] element, prob = %f    (%d, %d)-(%d, %d) batch id : %d", \
        curProposal, label, confidence, xmin, ymin, xmax, ymax, image_id);

        if (confidence > 0.5) {
            /** Drawing only objects with >50% probability **/
            classes[image_id][object_num[image_id]] = label;
            boxes[image_id][object_num[image_id]].x_min = xmin;
            boxes[image_id][object_num[image_id]].y_min = ymin;
            boxes[image_id][object_num[image_id]].rect_width = xmax - xmin;
            boxes[image_id][object_num[image_id]].rect_height = ymax - ymin;
            printf(" WILL BE PRINTED!");
            ++object_num[image_id];
        }
        printf("\n");
    }
    /** Adds rectangles to the image and save **/
    int batch_id;
    for (batch_id = 0; batch_id < batchSize; ++batch_id) {
        if (object_num[batch_id] > 0) {
            image_add_rectangles(&originalImages[batch_id], boxes[batch_id], classes[batch_id], object_num[batch_id], 2);
        }
        const char *out = "out_";
        char *img_path = (char *)malloc(strlen(out) + 1);
        char str_num[16] = {0};
        strcpy(img_path, out);
        int2str(str_num, batch_id); 
        strcat(img_path, str_num);
        strcat(img_path, ".bmp");
        image_save(img_path, &originalImages[batch_id]);
        printf("%sImage %s created!\n", info, img_path);
        free(img_path);
        image_free(&originalImages[batch_id]);
    }
    free(originalImages);
    // -----------------------------------------------------------------------------------------------------

    printf("%sExecution successful\n", info);

    for (i = 0; i < image_num; ++i) {
        free(classes[i]);
        free(boxes[i]);
    }
    free(classes);
    free(boxes);
    free(object_num);
    ie_infer_request_free(&infer_request);
    ie_exec_network_free(&exe_network);
    ie_network_free(&network);
    ie_core_free(&core);
    ie_network_name_free(&imageInputName);
    ie_network_name_free(&imInfoInputName);
    ie_network_name_free(&output_name);
    free(input_weight);
    free(argv_temp);
    
    return 0;
}
