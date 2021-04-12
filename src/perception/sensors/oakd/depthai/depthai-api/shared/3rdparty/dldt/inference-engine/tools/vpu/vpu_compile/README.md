# myriad_compile tool

This topic demonstrates how to run the `myriad_compile` tool application, which intended to dump blob for `vpu` plugins of Inference Engine by configuration options.

## How It Works

Upon the start-up, the tool application reads command line parameters and loads a network to the Inference Engine plugin.
Then application exports blob and writes it to the output file.

## Running

Running the application with the <code>-h</code> option yields the following usage message:

```sh
./myriad_compile -h
Inference Engine:
        API version ............ <version>
        Build .................. <build>

myriad_compile [OPTIONS]
[OPTIONS]:
    -h                                       Optional. Print a usage message.
    -m                           <value>     Required. Path to xml model.
    -pp                          <value>     Optional. Path to a plugin folder.
    -o                           <value>     Optional. Path to the output file. Default value: "<model_xml_file>.blob".
    -c                           <value>     Optional. Path to the configuration file. Default value: "config".
    -ip                          <value>     Optional. Specifies precision for all input layers of network. Supported values: FP32, FP16, U8. Default value: FP16.
    -op                          <value>     Optional. Specifies precision for all output layers of network. Supported values: FP32, FP16, U8. Default value: FP16.
    -iop                        "<value>"    Optional. Specifies precision for input/output layers by name.
                                             By default all inputs and outputs have FP16 precision.
                                             Available precisions: FP32, FP16, U8.
                                             Example: -iop "input:FP16, output:FP16".
                                             Notice that quotes are required.
                                             Overwrites precision from ip and op options for specified layers.
    -VPU_MYRIAD_PLATFORM         <value>     Optional. Specifies movidius platform. Supported values: VPU_MYRIAD_2450, VPU_MYRIAD_2480. Overwrites value from config.
                                             This option must be used in order to compile blob without a connected Myriad device.
    -VPU_NUMBER_OF_SHAVES        <value>     Optional. Specifies number of shaves. Should be set with "VPU_NUMBER_OF_CMX_SLICES". Overwrites value from config.
    -VPU_NUMBER_OF_CMX_SLICES    <value>     Optional. Specifies number of CMX slices. Should be set with "VPU_NUMBER_OF_SHAVES". Overwrites value from config.
```

Running the application with the empty list of options yields an error message.

You can use the following command to dump blob using a trained Faster R-CNN network:

```sh
./myriad_compile -m <path_to_model>/model_name.xml
```

## Platform option

You can dump blob without a connected Myriad device.
To do that, you must specify type of movidius platform using the parameter -VPU_MYRIAD_PLATFORM.
Supported values: VPU_MYRIAD_2450, VPU_MYRIAD_2480

## Import and Export functionality

#### Export

You can save a blob file from your application.
To do this, you should call the `Export()` method on the `ExecutableNetwork` object.
`Export()` has the following argument:
* Name of output blob [IN]

Example:

```sh
InferenceEngine::ExecutableNetwork executableNetwork = plugin.LoadNetwork(network,{});
executableNetwork.Export("model_name.blob");
```

#### Import

You can upload blob with network into your application.
To do this, you should call the `ImportNetwork()` method on the `InferencePlugin` object.
`ImportNetwork()` has the following arguments:
* ExecutableNetwork [OUT]
* Path to blob [IN]
* Config options [IN]

Example:

```sh
std::string modelFilename ("model_name.blob");
InferenceEngine::IExecutableNetwork::Ptr importedNetworkPtr;
pluginPtr->ImportNetwork(importedNetworkPtr, modelFilename, {});
```

> **NOTE**: Models should be first converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer).
