import numpy as np
import onnxruntime as ort
from PIL import Image

def get_face_embedding(img_path: str, onnx_path: str) -> np.ndarray:
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_node = sess.get_inputs()[0]
    input_name = in_node.name
    in_shape = in_node.shape
    print("模型输入shape:", in_shape)

    # 自动解析模型需要的 H W C
    # 两种常见布局：
    # 1. NCHW: [1, 3, 160, 160]  batch, channel, h, w
    # 2. NHWC: [1, 160, 160, 3]  batch, h, w, channel
    if in_shape[1] == 3:
        # NCHW
        _, C, target_h, target_w = in_shape
        transpose_order = (2, 0, 1)
    elif in_shape[-1] == 3:
        # NHWC
        _, target_h, target_w, C = in_shape
        transpose_order = None
    else:
        raise Exception("模型输入非3通道RGB，不兼容")

    # 加载人脸并缩放
    img = Image.open(img_path).convert("RGB")
    img = img.resize((target_w, target_h))
    img_np = np.array(img, dtype=np.float32)

    # FaceNet 归一化 [-1, 1]
    img_np = (img_np - 127.5) / 128.0

    # 通道转换
    if transpose_order is not None:
        img_np = np.transpose(img_np, transpose_order)
    # 增加batch维度
    input_tensor = np.expand_dims(img_np, axis=0)

    # 推理
    outputs = sess.run(None, {input_name: input_tensor})
    feat = outputs[0]

    # 标准FaceNet输出 [1,128]，取出向量
    embedding = feat[0].flatten()
    # L2归一化
    norm = np.linalg.norm(embedding)
    if norm > 1e-6:
        embedding = embedding / norm
    return embedding


def cos_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.dot(v1, v2))


# ========== 调用示例 ==========
if __name__ == "__main__":
    vec = get_face_embedding("./crop_face_1.jpg", "./facenet.onnx")
    print("特征向量shape:", vec.shape)  # 输出 (128,)
    print("向量前10维：", vec[:10])

    # 一对一比对
    res = get_face_embedding("./crop_face_2.jpg", "./facenet.onnx")
    print("一对一比对结果: ", cos_sim(vec, res))

    # 一对一比对
    res = get_face_embedding("./crop_face_3.jpg", "./facenet.onnx")
    print("一对多比对结果: ", cos_sim(vec, res))