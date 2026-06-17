import cv2
import numpy as np
import onnxruntime as ort
import os
import argparse
from typing import List, Tuple, Optional, Dict

# 确保 box_utils.py 在同目录下（UltraFace 后处理依赖）
from box_utils import predict
ort.set_default_logger_severity(3)


# ==============================================
# 1. 人脸检测类：UltraFaceDetector
# ==============================================
class UltraFaceDetector:
    """基于 UltraFace ONNX 模型的人脸检测器"""

    def __init__(self, model_path: str, use_gpu: bool = False):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name

    @staticmethod
    def scale(box):
        """将检测框缩放为正方形，便于后续人脸识别"""
        width = box[2] - box[0]
        height = box[3] - box[1]
        maximum = max(width, height)
        dx = int((maximum - width) / 2)
        dy = int((maximum - height) / 2)
        return [box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy]

    def detect(self, img: np.ndarray, threshold: float = 0.7) -> Tuple[np.ndarray, Optional[list], np.ndarray]:
        """检测人脸，返回 bboxes, kpss(UltraFace无关键点返回None), probs"""
        orig_h, orig_w = img.shape[:2]

        # 预处理
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 480))
        image_mean = np.array([127, 127, 127])
        image = (image - image_mean) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0).astype(np.float32)

        # 推理
        confidences, boxes = self.sess.run(None, {self.input_name: image})

        # 后处理 (依赖 box_utils.predict)
        boxes, labels, probs = predict(orig_w, orig_h, confidences, boxes, threshold)
        return boxes, None, probs

    def draw_detections(self, img: np.ndarray, bboxes: np.ndarray, kpss=None, probs=None) -> np.ndarray:
        """在原图上绘制检测框和置信度"""
        img_draw = img.copy()
        draw_color = (255, 128, 0)
        for i, box in enumerate(bboxes):
            square_box = self.scale(box)
            x1, y1, x2, y2 = map(int, square_box)
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), draw_color, 2)
            if probs is not None:
                cv2.putText(img_draw, f"{probs[i]:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 2)
        return img_draw


# ==============================================
# 2. 人脸编码类：FaceNetEncoder
# ==============================================
class FaceNetEncoder:
    """基于 FaceNet ONNX 模型的人脸特征提取器（支持内存直传）"""

    def __init__(self, model_path: str, use_gpu: bool = False):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu else ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(model_path, providers=providers)
        in_node = self.sess.get_inputs()[0]
        self.input_name = in_node.name
        in_shape = in_node.shape

        # 自动解析模型需要的 H W C 布局
        if in_shape[1] == 3:  # NCHW
            _, self.C, self.target_h, self.target_w = in_shape
            self.transpose_order = (2, 0, 1)
        elif in_shape[-1] == 3:  # NHWC
            _, self.target_h, self.target_w, self.C = in_shape
            self.transpose_order = None
        else:
            raise Exception("模型输入非3通道RGB，不兼容")

    @staticmethod
    def crop_image(image, box):
        """根据坐标裁剪图像，防止越界"""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = max(0, int(box[0])), max(0, int(box[1])), min(w, int(box[2])), min(h, int(box[3]))
        return image[y1:y2, x1:x2]

    def encode(self, img: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """直接从内存中的原图和bbox提取 Embedding，零磁盘IO"""
        # 1. 裁剪为正方形
        square_box = UltraFaceDetector.scale(bbox)
        face_crop = self.crop_image(img, square_box)

        # 2. BGR 转 RGB (cv2 默认 BGR，FaceNet 需要 RGB)
        img_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        # 3. 缩放到模型需要的尺寸
        img_resized = cv2.resize(img_rgb, (self.target_w, self.target_h))

        # 4. 归一化到 [-1, 1]
        img_np = img_resized.astype(np.float32)
        img_np = (img_np - 127.5) / 128.0

        # 5. 通道转换 (如果是 NCHW 格式)
        if self.transpose_order is not None:
            img_np = np.transpose(img_np, self.transpose_order)

        # 6. 增加 batch 维度并推理
        input_tensor = np.expand_dims(img_np, axis=0)
        outputs = self.sess.run(None, {self.input_name: input_tensor})

        # 7. 取出向量并进行 L2 归一化
        embedding = outputs[0][0].flatten()
        norm = np.linalg.norm(embedding)
        if norm > 1e-6:
            embedding = embedding / norm

        return embedding


# ==============================================
# 3. 人脸库管理类：FaceDatabase
# ==============================================
class FaceDatabase:
    """人脸库管理类，支持本地持久化和1:N余弦相似度比对"""

    def __init__(self, similarity_threshold: float = 0.6):
        self.threshold = similarity_threshold
        self.faces: Dict[str, np.ndarray] = {}  # key:人名 value:特征向量

    def add_face(self, name: str, feature: np.ndarray) -> None:
        if name in self.faces:
            print(f"提示: 人脸 [{name}] 已存在，将覆盖原有特征")
        self.faces[name] = feature

    def remove_face(self, name: str) -> bool:
        if name in self.faces:
            del self.faces[name]
            return True
        return False

    def list_faces(self) -> List[str]:
        return list(self.faces.keys())

    def save(self, save_path: str) -> None:
        if not self.faces:
            print("提示: 人脸库为空，无需保存")
            return
        names = list(self.faces.keys())
        features = np.stack([self.faces[name] for name in names])
        np.savez(save_path, names=names, features=features)
        print(f"人脸库已保存到: {save_path}，共 {len(names)} 张人脸")

    def load(self, load_path: str) -> None:
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"人脸库文件不存在: {load_path}")
        data = np.load(load_path, allow_pickle=True)
        names = data["names"].tolist()
        features = data["features"]
        self.faces = {name: feat for name, feat in zip(names, features)}
        print(f"人脸库加载成功，共 {len(names)} 张人脸")

    @staticmethod
    def cosine_similarity(feat1: np.ndarray, feat2: np.ndarray) -> float:
        return float(np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2)))

    def match(self, target_feature: np.ndarray) -> Tuple[Optional[str], float, bool]:
        if not self.faces:
            return None, 0.0, False

        max_sim = -1.0
        best_name = None
        for name, feat in self.faces.items():
            sim = self.cosine_similarity(target_feature, feat)
            if sim > max_sim:
                max_sim = sim
                best_name = name

        is_match = max_sim >= self.threshold
        return best_name, max_sim, is_match


# ==============================================
# 4. 整合系统类：FaceRecognitionSystem
# ==============================================
class FaceRecognitionSystem:
    """人脸识别完整系统，整合检测、编码、比对全流程"""

    def __init__(self, det_model_path: str, rec_model_path: str, similarity_threshold: float = 0.6, use_gpu: bool = False):
        self.detector = UltraFaceDetector(det_model_path, use_gpu=use_gpu)
        self.encoder = FaceNetEncoder(rec_model_path, use_gpu=use_gpu)
        self.database = FaceDatabase(similarity_threshold)

    def detect_and_draw(self, img_path: str, save_path: Optional[str] = None) -> np.ndarray:
        img = cv2.imread(img_path)
        if img is None: raise FileNotFoundError(f"无法读取图像: {img_path}")

        bboxes, kpss, probs = self.detector.detect(img)
        print(f"检测到 {len(bboxes)} 张人脸")
        img_draw = self.detector.draw_detections(img, bboxes, kpss, probs)

        if save_path:
            cv2.imwrite(save_path, img_draw)
            print(f"检测结果已保存到: {save_path}")
        return img_draw

    def get_face_encodings(self, img_path: str) -> List[Dict]:
        img = cv2.imread(img_path)
        if img is None: raise FileNotFoundError(f"无法读取图像: {img_path}")

        bboxes, kpss, probs = self.detector.detect(img)
        if len(bboxes) == 0:
            print("未检测到人脸")
            return []

        encodings = []
        for i in range(len(bboxes)):
            feat = self.encoder.encode(img, bboxes[i])
            encodings.append({"bbox": bboxes[i], "probability": probs[i], "feature": feat})

        print(f"成功提取 {len(encodings)} 张人脸的特征向量")
        return encodings

    def verify_face_in_database(self, img_path: str, use_largest_face: bool = True) -> Dict:
        img = cv2.imread(img_path)
        if img is None: raise FileNotFoundError(f"无法读取图像: {img_path}")

        bboxes, kpss, probs = self.detector.detect(img)
        if len(bboxes) == 0:
            return {"success": False, "message": "未检测到人脸"}

        if use_largest_face:
            areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in bboxes]
            target_idx = int(np.argmax(areas))
            target_bbox = bboxes[target_idx]
        else:
            target_bbox = bboxes[0]

        target_feat = self.encoder.encode(img, target_bbox)
        best_name, similarity, is_match = self.database.match(target_feat)

        return {
            "success": True, "is_match": is_match, "best_match": best_name,
            "similarity": similarity, "threshold": self.database.threshold, "face_count": len(bboxes)
        }


def get_first_face_images(dataset_root: str) -> list:
    """
    遍历 dataset 文件夹，获取每个人名文件夹下的第一张人脸照片
    返回：列表，每个元素是 (人名, 图片绝对路径) 元组
    """
    # 最终结果列表
    result = []

    # 常见图片后缀（可自行添加）
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.ppm', '.tiff')

    # 遍历 dataset 下的所有子文件夹（人名）
    for person_name in os.listdir(dataset_root):
        person_dir = os.path.join(dataset_root, person_name)

        # 只处理文件夹，跳过文件
        if not os.path.isdir(person_dir):
            continue

        # 获取该人文件夹下所有图片文件
        image_files = []
        for filename in os.listdir(person_dir):
            if filename.lower().endswith(IMG_EXTENSIONS):
                image_files.append(filename)

        # 如果没有图片，跳过
        if not image_files:
            print(f"⚠️  文件夹 {person_name} 中未找到图片，已跳过")
            continue

        # 取第一张图片
        first_image = image_files[0]
        first_image_path = os.path.abspath(os.path.join(person_dir, first_image))

        # 存入元组 → 加入列表
        result.append((person_name, first_image_path))

    return result


def test_face_detect(test_img, save_path="det_result.jpg"):
    print("=" * 60)
    print("【功能1】人脸检测与绘制")
    face_sys = FaceRecognitionSystem(det_model_path="./version-RFB-640.onnx", rec_model_path="./facenet.onnx")
    face_sys.detect_and_draw(test_img, save_path)
    print("save_path:", save_path)


def test_face_encoding(test_img):
    print("=" * 60)
    print("【功能2】人脸编码")
    face_sys = FaceRecognitionSystem(det_model_path="./version-RFB-640.onnx", rec_model_path="./facenet.onnx")
    encodings = face_sys.get_face_encodings(test_img)
    if encodings:
        print(f"特征向量维度: {encodings[0]['feature'].shape}")
        print(f"特征前5个值: {encodings[0]['feature'][:5].round(4)}")
    # for encoding in encodings:
    #     print(encoding)
    return encodings


def test_face_encoding_save(test_imgs_path="/Users/zh/yunqi/TorchAttacks/model/datasets", db_path = "face_database.npz"):
    print("\n" + "=" * 60)
    print("【功能3】人脸库构建与保存")
    face_sys = FaceRecognitionSystem(det_model_path="./version-RFB-640.onnx", rec_model_path="./facenet.onnx")
    person_list = [
        ("Person_A", "/Users/zh/yunqi/TorchAttacks/model/datasets/Abdullah_Ahmad_Badawi/Abdullah_Ahmad_Badawi_0001.jpg"),
        ("Person_B", "/Users/zh/yunqi/TorchAttacks/model/datasets/Abdullah_Gul/Abdullah_Gul_0001.jpg")
    ]

    # 获取结果
    person_list = get_first_face_images(test_imgs_path)

    # 打印查看
    print(f"\n✅ 共获取到 {len(person_list)} 个人的照片")
    print("\n前 5 条数据示例：")
    for item in person_list[:5]:
        print(item)

    print("正在构建人脸库...")
    for name, img_path in person_list:
        if os.path.exists(img_path):
            face_info = face_sys.get_face_encodings(img_path)
            if face_info:
                face_sys.database.add_face(name, face_info[0]["feature"])
        else:
            print(f"跳过不存在的注册图片: {img_path}")

    print(f"\n当前人脸库成员: {face_sys.database.list_faces()}")

    # 3.2 保存与加载演示
    face_sys.database.save(db_path)
    # face_sys.database.faces = {}
    # face_sys.database.load(db_path)
    # print(f"\n当前人脸库成员: {face_sys.database.list_faces()}")

    # 检查没有存到库中的人脸
    # person_names = []
    # for name in os.listdir(test_imgs_path):
    #     full_path = os.path.join(test_imgs_path, name)
    #     if os.path.isdir(full_path):
    #         person_names.append(name)
    # unregistered = [x for x in person_names if x not in face_sys.database.list_faces()]
    # print(unregistered)

    # result = face_sys.verify_face_in_database(args.test_img)


def test_face_compare(test_img):
    print("\n" + "=" * 60)
    print("【功能4】人脸库1:N比对")
    face_sys = FaceRecognitionSystem(det_model_path="./version-RFB-640.onnx", rec_model_path="./facenet.onnx")
    face_sys.database.load("face_database.npz")
    result = face_sys.verify_face_in_database(test_img)
    print(result)
    # print("\n比对结果:")
    # if result["success"]:
    #     if result["is_match"]:
    #         print(f"✅ 匹配成功！最匹配: {result['best_match']}，相似度: {result['similarity']:.4f}")
    #     else:
    #         print(f"❌ 匹配失败。最相似: {result['best_match']}，相似度: {result['similarity']:.4f}")
    #         print(f"阈值: {result['threshold']}，该人脸不在库中")
    # else:
    #     print(f"❌ 比对失败: {result['message']}")

# ==============================================
# 5. 使用示例与测试
# ==============================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--det_model", type=str, default="./version-RFB-640.onnx", help="UltraFace检测模型路径")
    parser.add_argument("--rec_model", type=str, default="./facenet.onnx", help="FaceNet识别模型路径")
    parser.add_argument("--test_img", type=str, default="/Users/zh/yunqi/TorchAttacks/model/datasets/Abdullah_Gul/Abdullah_Gul_0019.jpg", help="测试图片路径")
    args = parser.parse_args()

    # 初始化系统
    face_sys = FaceRecognitionSystem(
        det_model_path=args.det_model,
        rec_model_path=args.rec_model,
        similarity_threshold=0.6,  # FaceNet 余弦相似度阈值通常设为 0.6 左右
        use_gpu=False
    )

    # ===================== 1. 人脸检测并绘制框 =====================
    test_face_detect(args.test_img)

    # ===================== 2. 提取人脸编码向量 =====================
    test_face_encoding(args.test_img)

    # ===================== 3. 人脸库构建 =====================
    test_face_encoding_save()

    # ===================== 4. 人脸库1:N比对 =====================
    test_face_compare(args.test_img)
