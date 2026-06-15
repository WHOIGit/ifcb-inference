import argparse

import onnx
from onnx import TensorProto, helper, shape_inference

# Ops that act as the final classification head; their data input is the
# penultimate-layer embedding (the global-pooled feature vector).
_HEAD_OPS = ("Gemm", "MatMul")


def detect_embedding_tensor(graph):
    """Return the name of the embedding tensor: the data input of the last
    classification-head op (Gemm/MatMul), i.e. its input that is not a weight
    or bias initializer."""
    initializers = {init.name for init in graph.initializer}
    for node in reversed(graph.node):
        if node.op_type in _HEAD_OPS:
            data_inputs = [name for name in node.input if name not in initializers]
            if not data_inputs:
                continue
            return data_inputs[0]
    raise ValueError(
        f"Could not find a classification-head op ({'/'.join(_HEAD_OPS)}) to "
        "auto-detect the embedding tensor. Pass --tensor-name explicitly."
    )


def _infer_embedding_dim(graph, tensor_name):
    """Best-effort embedding dimension D from the head op's weight initializer
    (its second dim). Returns None if it cannot be determined."""
    initializers = {init.name: init for init in graph.initializer}
    for node in graph.node:
        if node.op_type in _HEAD_OPS and tensor_name in node.input:
            for name in node.input:
                if name in initializers:
                    dims = initializers[name].dims
                    if len(dims) == 2:
                        # Gemm fc.weight is [num_classes, D]; D is the input width.
                        return dims[1]
    return None


def add_embedding_output(model_path: str, output_path: str, tensor_name=None):
    """Add the penultimate-layer embedding tensor to an ONNX model's declared
    graph outputs, producing a dual-head model whose ``session.run(None, ...)``
    returns ``[logits, embedding]``.

    Parameters:
        model_path (str): Path to the input ONNX model.
        output_path (str): Path to save the modified ONNX model.
        tensor_name (str or None): Name of the embedding tensor. If None, it is
            auto-detected as the data input of the final Gemm/MatMul.
    """
    model = onnx.load(model_path)
    graph = model.graph

    if tensor_name is None:
        tensor_name = detect_embedding_tensor(graph)
    print(f"Embedding tensor: '{tensor_name}'")

    existing_outputs = {out.name for out in graph.output}
    if tensor_name in existing_outputs:
        print(f"'{tensor_name}' is already a graph output; saving unchanged copy.")
        onnx.save(model, output_path)
        print(f"model saved to: {output_path}")
        return

    # Prefer shape-inferred value_info so the output carries shape/type metadata.
    inferred = shape_inference.infer_shapes(model)
    value_info = None
    for vi in inferred.graph.value_info:
        if vi.name == tensor_name:
            value_info = vi
            break

    if value_info is None:
        dim = _infer_embedding_dim(graph, tensor_name)
        shape = ["batch_size", dim] if dim is not None else ["batch_size", 0]
        value_info = helper.make_tensor_value_info(
            tensor_name, TensorProto.FLOAT, shape
        )
        print(f"No inferred value_info; declared shape {shape}.")

    graph.output.append(value_info)
    onnx.save(model, output_path)
    print(
        f"Added embedding output '{tensor_name}'. "
        f"graph.output is now: {[out.name for out in graph.output]}"
    )
    print(f"model saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Add the penultimate-layer embedding tensor to an ONNX "
        "model's graph outputs (dual-head: [logits, embedding])."
    )
    parser.add_argument("model_path", help="Path to input ONNX model")
    parser.add_argument("output_path", help="Path to save the modified ONNX model")
    parser.add_argument(
        "--tensor-name",
        default=None,
        help="Embedding tensor name. Default: auto-detect (data input of the "
        "final Gemm/MatMul).",
    )
    args = parser.parse_args()
    add_embedding_output(args.model_path, args.output_path, args.tensor_name)


if __name__ == "__main__":
    main()
