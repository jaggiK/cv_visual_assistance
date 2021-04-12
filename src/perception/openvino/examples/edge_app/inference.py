"""
Contains code for working with the Inference Engine.
You'll learn how to implement this code and more in
the related lesson on the topic.
"""

import os
import time

from openvino.inference_engine import IENetwork, IECore


class Network:
    """
    Load and store information for working with the Inference Engine,
    and any loaded models.
    """

    def __init__(self):
        self.plugin = None
        self.input_blob = None
        self.exec_network = None
        self.num_requests = 2

    def load_model(self, model, device="CPU", cpu_extension=None):
        """
        Load the model given IR files.
        Defaults to CPU as device for use in the workspace.
        Synchronous requests made within.
        """
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        print(model_bin)
        # Initialize the plugin
        self.plugin = IECore()
        print(self.plugin.available_devices)
        # TODO (Jagadish) : below code is not required for latest v2020, so remove when confident
        # Add a CPU extension, if applicable
        # if cpu_extension and "CPU" in device:
        #     self.plugin.add_extension(cpu_extension, device)

        print("---extracting output")
        # Read the IR as a IENetwork
        network = IENetwork(model=model_xml, weights=model_bin)
        # self.plugin.set_config({'VPU_HW_STAGES_OPTIMIZATION': 'YES'}, "MYRIAD")
        # Load the IENetwork into the plugin
        self.exec_network = self.plugin.load_network(network, device_name=device)
        self.num_requests = self.exec_network.get_metric("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
        self.exec_network = self.plugin.load_network(network, device, num_requests=self.num_requests)

        # Get the input layer
        self.input_blob = next(iter(network.inputs))

        # Return the input shape (to determine preprocessing)
        return network.inputs[self.input_blob].shape

    def sync_inference(self, image):
        """
        Makes a synchronous inference request, given an input image.
        """
        self.exec_network.infer({self.input_blob: image})
        return

    def async_inference(self, image, request_id):
        '''
            Performs asynchronous inference
            Returns the `exec_net`
            '''
        self.exec_network.start_async(request_id=request_id, inputs={self.input_blob: image})
        return self.exec_network

    def wait(self, request_id):
        '''
        Checks the status of the inference request.
        '''
        status = self.exec_network.requests[request_id].wait(-1)
        return status

    def extract_output(self):
        """
        Returns a list of the results for the output layer of the network.
        """
        return self.exec_network.requests[0].outputs

    def extract_output_async(self, request_id):
        """
        Returns a list of the results for the output layer of the network.
        """
        return self.exec_network.requests[request_id].outputs
