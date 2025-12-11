from huggingface_hub import snapshot_download
import os
import sys
import time

print("=" * 60)
print("开始下载数据集...")
print("仓库: QingyanBai/Ditto-1M")
print("目标目录: ./Ditto-1M")
print("=" * 60)

# 检查 Hugging Face 环境变量
hf_endpoint = os.environ.get('HF_ENDPOINT', 'https://huggingface.co')
print(f"使用 Hugging Face 端点: {hf_endpoint}")

# 如果设置了镜像，提示用户
if 'mirror' in hf_endpoint.lower() or 'hf-mirror' in hf_endpoint.lower():
    print("检测到使用镜像站点")
else:
    print("提示: 如果下载速度慢，可以设置环境变量:")
    print("  export HF_ENDPOINT=https://hf-mirror.com")

start_time = time.time()

try:
    snapshot_download(
        repo_id="QingyanBai/Ditto-1M",
        repo_type="dataset",
        local_dir="./Ditto-1M/tests", 
        allow_patterns=["mini_test_videos/*"],
        resume_download=True,  # 支持断点续传
        local_dir_use_symlinks=False,  # 不使用符号链接，避免权限问题
        # max_workers=8,  # 增加并发数以提升下载速度（香港网络通常可以支持更多并发）
        tqdm_class=None,  # 使用默认的进度条显示
    )
    elapsed_time = time.time() - start_time
    print("=" * 60)
    print(f"下载完成！总耗时: {elapsed_time/60:.2f} 分钟")
    print("=" * 60)
except KeyboardInterrupt:
    print("\n下载被用户中断")
    sys.exit(1)
except Exception as e:
    print(f"\n下载过程中出现错误: {e}")
    print("\n可能的解决方案:")
    print("1. 检查网络连接是否稳定")
    print("2. 检查是否有足够的磁盘空间（需要约 21.5GB+）")
    print("3. 尝试使用镜像站点:")
    print("   export HF_ENDPOINT=https://hf-mirror.com")
    print("4. 如果之前下载中断，删除 .incomplete 文件后重试")
    print("5. 检查防火墙或代理设置")
    sys.exit(1)