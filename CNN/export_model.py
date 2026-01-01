import torch;
from model import CNN;
# import openvino as ov;
import utils;

def export_to_onnx(model: torch.nn.Module, device: torch.device, filepath: str = "model.onnx"):
    model.eval();
    dummy_input: torch.tensor = torch.randn(1, 3, 32, 32);
    torch.onnx.export(
        model, dummy_input, filepath,
        export_params=True,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    );
    print(f"ONNX model saved to {filepath}");
    return ;

# def convert_to_openvino(onnx_path, xml_path="model.xml"):
#     ov_model = ov.convert_model(onnx_path);
#     ov.save_model(ov_model, xml_path);
#     print(f"OpenVINO IR saved to {xml_path}");
#     return ;

def main():
    device: torch.device = utils.load_device();
    model: torch.nn.Module = utils.load_model("best_model_state.pth", device);
    export_to_onnx(model, "model.onnx");
#    convert_to_openvino("model.onnx");
    return ;

if __name__ == "__main__":
    main();
