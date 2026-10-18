"""Export yolov8n-pose.pt to optimized runtimes for CPU inference.

产出的模型供运行时加载（config.MODEL_PATH，加载时传 OpenVINO 目录）：
  models/yolov8n-pose_openvino_model/        FP16 OpenVINO IR（默认部署用）
  models/yolov8n-pose_int8_openvino_model/   INT8 混合精度（NNCF PTQ，可选）

用法：
  python tools/export_model.py                     # 导出 FP16 OpenVINO IR
  python tools/export_model.py --int8 --calib-dir <dir>   # 再导出 INT8 混合精度

依赖：ultralytics>=8.3, openvino（FP16）；nncf（INT8，见 requirements-export.txt）。
校准数据：必须来自业务真实画面（本项目实际处理的视频抽帧/快照），
不要用无关图片，否则 INT8 量化精度会严重下降。
采集校准图：python tools/collect_calib.py   # 从业务视频抽帧生成 models/calib/
"""
import argparse
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO / "models"
SRC_PT = MODELS_DIR / "yolov8n-pose.pt"

FP16_DIR = MODELS_DIR / "yolov8n-pose_openvino_model"
INT8_DIR = MODELS_DIR / "yolov8n-pose_int8_openvino_model"
FP16_XML = FP16_DIR / "yolov8n-pose.xml"
INT8_XML = INT8_DIR / "yolov8n-pose.xml"


def _ensure_source():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if not SRC_PT.exists():
        cand = REPO / "yolov8n-pose.pt"
        if cand.exists():
            shutil.copy(cand, SRC_PT)
    if not SRC_PT.exists():
        sys.exit(f"未找到源模型 yolov8n-pose.pt，请先放到 {SRC_PT}")


def _load():
    from ultralytics import YOLO
    return YOLO(str(SRC_PT))


def export_fp16():
    """原生 ultralytics OpenVINO 导出（FP16 IR，opset 17，含 metadata）。"""
    if FP16_XML.exists() and FP16_XML.stat().st_size > 0:
        print(f"[export] FP16 IR 已存在：{FP16_DIR}，跳过。删除该目录可重新导出")
        return FP16_DIR
    print("[export] 导出 OpenVINO IR (FP16, opset 17, CPU, 动态 batch)...")
    model = _load()
    # dynamic=True：batch 维度动态，支持运行时批推理（detect_objects 按 16
    # 帧/批推理提速）。静态 batch=1 会导致批推理报
    # "got [16,3,640,640] expecting [1,3,640,640]"。
    out = model.export(format="openvino", opset=17, half=True,
                       dynamic=True, device="cpu", verbose=True)
    print(f"[export] FP16 IR 已保存：{out}")
    return Path(out)


def _make_calib_yaml(calib_dir, out_yaml):
    """生成 INT8 校准数据集的 data.yaml（path + val 指向图片目录）。"""
    calib_dir = Path(calib_dir).resolve()
    imgs = [p for p in calib_dir.iterdir() if p.suffix.lower() in
            (".jpg", ".jpeg", ".png", ".bmp")]
    if not imgs:
        sys.exit(f"[export] 校准目录无可用图片：{calib_dir}")
    out_yaml = Path(out_yaml)
    out_yaml.write_text(
        f"path: {calib_dir.as_posix()}\n"
        f"train: .\n"
        f"val: .\n"
        f"kpt_shape: [17, 3]\n"   # COCO-pose：17 关键点，(x, y, visible)
        f"names:\n  0: person\n", encoding="utf-8")
    return out_yaml


def export_int8(calib_dir):
    """ultralytics 原生 INT8 导出：NNCF PTQ，MIXED preset +
    自动 ignored_scope（Pose 头的 Add/Sub/Mul/Div/dfl/Sigmoid 保留 FP16）。"""
    if INT8_XML.exists() and INT8_XML.stat().st_size > 0:
        print(f"[export] INT8 IR 已存在：{INT8_DIR}，跳过。删除该目录可重新导出")
        return INT8_DIR
    print(f"[export] 准备 INT8 校准数据：{calib_dir}")
    yaml_path = _make_calib_yaml(calib_dir, MODELS_DIR / "calib_data.yaml")
    print("[export] NNCF 混合精度量化 (INT8, MIXED preset)...")
    model = _load()
    # 注意：int8 与 half 互斥（ultralytics 内部 half 会被强制置 False）。
    # INT8 模型文件本身就是 FP16 权重 + INT8 激活，无需也不能传 half=True。
    # dynamic=True 保证批推理可用（与 FP16 一致）。
    out = model.export(format="openvino", int8=True, half=False,
                       opset=17, dynamic=True, device="cpu",
                       data=str(yaml_path), verbose=True)
    print(f"[export] INT8 IR 已保存：{out}")
    return Path(out)


def verify(dst):
    """用一张真实图跑一次推理，确认 IR 能正常加载并出 person 检测框。"""
    from ultralytics import YOLO
    test_img = None
    for cand in (MODELS_DIR / "calib", REPO / "devtest", REPO):
        if not cand.exists():
            continue
        for p in sorted(cand.glob("*.jpg")):
            test_img = str(p)
            break
        if test_img:
            break
    if not test_img:
        print("[verify] 无测试图，跳过")
        return
    m = YOLO(str(dst))
    r = m(test_img, classes=[0], conf=0.3, verbose=False)[0]
    n = len(r.boxes)
    kp = len(r.keypoints.xy) if r.keypoints is not None else 0
    print(f"[verify] {dst.name}: person boxes={n}, keypoints={kp} "
          f"({'正常' if n > 0 else '无检测，注意检查'})")


def main():
    ap = argparse.ArgumentParser(description="Export yolov8n-pose to OpenVINO")
    ap.add_argument("--int8", action="store_true", help="额外导出 INT8 混合精度")
    ap.add_argument("--calib-dir", default=None,
                    help="INT8 校准图目录（缺省用 models/calib）")
    ap.add_argument("--verify", action="store_true", default=True,
                    help="导出后跑一次推理校验")
    args = ap.parse_args()

    _ensure_source()
    fp16_dir = export_fp16()
    if args.verify:
        verify(fp16_dir)

    if args.int8:
        calib_dir = Path(args.calib_dir) if args.calib_dir else (
            MODELS_DIR / "calib")
        if not calib_dir.exists():
            sys.exit("[export] 未找到校准图目录：请用 --calib-dir 指定业务真实画面目录")
        int8_dir = export_int8(calib_dir)
        if args.verify:
            verify(int8_dir)

    print("\n[export] 完成。部署时将 models/ 拷贝到容器 /app/models（Dockerfile 已处理）。")


if __name__ == "__main__":
    main()
