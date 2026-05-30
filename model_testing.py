import os
from contextlib import nullcontext

import cv2
import numpy as np
import torch
import torch.nn as nn
from decord import VideoReader, cpu
from pytorchvideo.models.hub import slowfast_r50

# ============ CONFIGURATION ============
data_path = "/home/himanshus/Projects/fight_detection_model"

# Try different possible folder names
possible_fight_folders = ["fight_Test", "fight_test", "Fight_Test", "fight", "Fight"]
possible_normal_folders = ["normal_video", "normal", "Normal", "normal_videos"]

fight_path = None
normal_path = None

for folder in possible_fight_folders:
    test_path = os.path.join(data_path, folder)
    if os.path.exists(test_path):
        fight_path = test_path
        print(f"Found fight folder: {folder}")
        break

for folder in possible_normal_folders:
    test_path = os.path.join(data_path, folder)
    if os.path.exists(test_path):
        normal_path = test_path
        print(f"Found normal folder: {folder}")
        break

if fight_path is None:
    print(f"❌ Could not find fight folder in {data_path}")
    print(
        f"   Contents: {os.listdir(data_path) if os.path.exists(data_path) else 'Path does not exist'}"
    )
    exit(1)

if normal_path is None:
    print(f"❌ Could not find normal folder in {data_path}")
    print(
        f"   Contents: {os.listdir(data_path) if os.path.exists(data_path) else 'Path does not exist'}"
    )
    exit(1)

t = 32
a = 4
st = t // a
sz = 224
mn = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
sd = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)


# ============ MODEL DEFINITION ============
def parent(m, path):
    x = m
    if path == "":
        return x
    for s in path.split("."):
        x = getattr(x, s)
    return x


def replace_head(m, c=2):
    last = None
    for n, mod in m.named_modules():
        if isinstance(mod, nn.Linear):
            last = n

    if last is not None:
        p = last.rsplit(".", 1)[0] if "." in last else ""
        a = last.split(".")[-1]
        par = parent(m, p)
        old = getattr(par, a)
        if isinstance(old, nn.Linear):
            setattr(par, a, nn.Linear(old.in_features, c, bias=old.bias is not None))
            return True

    if hasattr(m, "blocks") and len(m.blocks) > 0:
        h = m.blocks[-1]
        if hasattr(h, "proj"):
            pr = h.proj
            if isinstance(pr, nn.Linear):
                h.proj = nn.Linear(pr.in_features, c, bias=pr.bias is not None)
                return True
            if isinstance(pr, nn.Sequential):
                for j in range(len(pr) - 1, -1, -1):
                    if isinstance(pr[j], nn.Linear):
                        old = pr[j]
                        pr[j] = nn.Linear(old.in_features, c, bias=old.bias is not None)
                        return True
    return False


class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = slowfast_r50(pretrained=True)
        ok = replace_head(self.m, 2)

        for n, p in self.m.named_parameters():
            p.requires_grad = False

        for n, p in self.m.named_parameters():
            if "blocks.4" in n or "blocks.5" in n or "blocks.6" in n or "proj" in n:
                p.requires_grad = True

        if not ok:
            raise RuntimeError("head replace failed")

    def forward(self, x):
        return self.m(x)


# ============ VIDEO PREDICTION FUNCTION ============
def predict_video(video_path, model, device="cuda"):
    model.eval()

    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        n = len(vr)

        if n >= t:
            s = (n - t) // 2
            idx = np.linspace(s, s + t - 1, t)
        else:
            idx = np.linspace(0, max(n - 1, 0), t)

        idx = np.clip(np.round(idx).astype(np.int64), 0, max(n - 1, 0))
        v = vr.get_batch(idx).asnumpy()

        v = np.stack(
            [cv2.resize(f, (sz, sz), interpolation=cv2.INTER_LINEAR) for f in v],
            axis=0,
        )
        v = np.ascontiguousarray(v)

        v = torch.from_numpy(v).float().permute(3, 0, 1, 2).contiguous() / 255.0
        v = (v - mn) / sd

        fast = v.unsqueeze(0).to(device)
        slow = v[:, ::a, :, :].unsqueeze(0).to(device)

        if slow.shape[2] < st:
            pad = slow[:, :, -1:, :, :].repeat(1, 1, st - slow.shape[2], 1, 1)
            slow = torch.cat([slow, pad], dim=2)
        elif slow.shape[2] > st:
            slow = slow[:, :, :st, :, :]

        with torch.no_grad():
            with torch.amp.autocast("cuda") if device == "cuda" else nullcontext():
                outputs = model([slow, fast])
                probs = torch.softmax(outputs, dim=1)
                pred = outputs.argmax(1).item()
                confidence = probs[0, pred].item()

        return pred, confidence

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None, None


# ============ RECURSIVELY FIND ALL VIDEOS ============
def find_videos(folder_path):
    """Recursively find all video files in folder and subfolders"""
    video_files = []
    video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".MP4", ".AVI")

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(video_extensions):
                video_files.append(os.path.join(root, file))

    return video_files


# ============ TEST ALL VIDEOS ============
def test_model():
    print("=" * 70)
    print("FIGHT DETECTION MODEL - FINAL TEST")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n📱 Device: {device.upper()}")

    print("\n📦 Loading model...")
    model = M().to(device)

    checkpoint_path = "best_slowfast_safe.pth"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Model file '{checkpoint_path}' not found!")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", "unknown")
        acc = checkpoint.get("accuracy", "unknown")
        print(f"✅ Loaded model from epoch {epoch} (accuracy: {acc}%)")
    else:
        model.load_state_dict(checkpoint)
        print("✅ Loaded best model weights")

    model.eval()

    # Find all videos
    print(f"\n📁 Scanning for videos...")
    fight_videos = find_videos(fight_path)
    normal_videos = find_videos(normal_path)

    print(f"   Found {len(fight_videos)} fight videos")
    print(f"   Found {len(normal_videos)} normal videos")

    # Test fight videos
    print("\n" + "=" * 70)
    print("📹 TESTING FIGHT VIDEOS")
    print("=" * 70)

    fight_correct = 0
    fight_results = []

    for i, video_path in enumerate(fight_videos):
        video_name = os.path.basename(video_path)
        pred, conf = predict_video(video_path, model, device)

        if pred is not None:
            is_correct = pred == 1
            if is_correct:
                fight_correct += 1

            prediction = "FIGHT" if pred == 1 else "NORMAL"
            status = "✓" if is_correct else "✗"

            fight_results.append(
                {
                    "file": video_name,
                    "path": video_path,
                    "prediction": prediction,
                    "confidence": conf,
                    "correct": is_correct,
                }
            )

            print(f"  {status} {video_name[:50]:<50} → {prediction:<6} ({conf:.3f})")

        # Print progress every 20 videos
        if (i + 1) % 20 == 0:
            print(f"  ... Progress: {i + 1}/{len(fight_videos)}")

    # Test normal videos
    print("\n" + "=" * 70)
    print("📹 TESTING NORMAL VIDEOS")
    print("=" * 70)

    normal_correct = 0
    normal_results = []

    for i, video_path in enumerate(normal_videos):
        video_name = os.path.basename(video_path)
        pred, conf = predict_video(video_path, model, device)

        if pred is not None:
            is_correct = pred == 0
            if is_correct:
                normal_correct += 1

            prediction = "FIGHT" if pred == 1 else "NORMAL"
            status = "✓" if is_correct else "✗"

            normal_results.append(
                {
                    "file": video_name,
                    "path": video_path,
                    "prediction": prediction,
                    "confidence": conf,
                    "correct": is_correct,
                }
            )

            print(f"  {status} {video_name[:50]:<50} → {prediction:<6} ({conf:.3f})")

        # Print progress every 20 videos
        if (i + 1) % 20 == 0:
            print(f"  ... Progress: {i + 1}/{len(normal_videos)}")

    # Final Summary
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS SUMMARY")
    print("=" * 70)

    total_correct = fight_correct + normal_correct
    total_videos = len(fight_videos) + len(normal_videos)
    overall_acc = (total_correct / total_videos * 100) if total_videos > 0 else 0

    print(f"\n{'Category':<15} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print("-" * 50)
    if len(fight_videos) > 0:
        print(
            f"{'Fight Videos':<15} {fight_correct:<10} {len(fight_videos):<10} {fight_correct / len(fight_videos) * 100:.1f}%"
        )
    if len(normal_videos) > 0:
        print(
            f"{'Normal Videos':<15} {normal_correct:<10} {len(normal_videos):<10} {normal_correct / len(normal_videos) * 100:.1f}%"
        )
    print("-" * 50)
    print(f"{'TOTAL':<15} {total_correct:<10} {total_videos:<10} {overall_acc:.1f}%")

    # Show misclassified
    print("\n" + "=" * 70)
    print("❌ MISCLASSIFIED VIDEOS")
    print("=" * 70)

    misclassified = []
    for r in fight_results:
        if not r["correct"]:
            misclassified.append(("FIGHT (missed)", r["file"], r["confidence"]))
    for r in normal_results:
        if not r["correct"]:
            misclassified.append(("NORMAL (false alarm)", r["file"], r["confidence"]))

    if misclassified:
        print(f"\nTotal misclassified: {len(misclassified)}")
        for label, file, conf in misclassified[:20]:
            print(f"  {label:<20} → {file[:50]} (conf: {conf:.3f})")
    else:
        print("\n  🎉 PERFECT! No misclassifications!")

    print("\n" + "=" * 70)
    print("✅ TEST COMPLETE")
    print("=" * 70)

    return overall_acc


if __name__ == "__main__":
    test_model()
