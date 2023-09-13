import onnx
onnx_model = onnx.load("model/nuScenes_3Dtracking.onnx")
graph = onnx_model.graph
nodes = graph.node
for i in range(len(nodes)):
    if(nodes[i].op_type == "Plugin"):
        nodes[i].op_type = "DCNv2"
onnx.save(onnx_model,"model/nuScenes_3Dtracking.onnx")