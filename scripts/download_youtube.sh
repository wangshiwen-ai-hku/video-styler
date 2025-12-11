#!/bin/bash
# 文件名: yt-download-and-rename.sh
# 用法: ./yt-download-and-rename.sh "https://www.youtube.com/watch?v=xxxx"

set -euo pipefail   # 安全模式

URL="$1"
DEST_DIR="./Ditto-1M/tests/youtube"

# 如果目录不存在就创建
mkdir -p "$DEST_DIR"

echo "正在下载: $URL → $DEST_DIR"

yt-dlp \
  --remote-components ejs:github \
  -f "bestvideo[height<=1080]+bestaudio/best" \
  --merge-output-format mp4 \
  -P "$DEST_DIR" \
  $URL

echo "下载完成，开始重命名..."

# 进入目标目录
cd "$DEST_DIR"

# 方法二（更稳健版）：找当前最大数字，然后继续往后排（适合批量多次运行）
# 取消下面注释即可替换上面那段
max=0
for f in *.mp4; do
    num=$(echo "$f" | grep -oE '^[0-9]+' || echo 0)
    (( num > max )) && max=$num
done

for f in *.mp4; do
    if [[ $f == [0-9]*.mp4 ]]; then
        continue  # 已经是数字命名，跳过
    fi
    ((max++))
    mv -v "$f" "${max}.mp4"
done


echo "全部搞定！当前目录文件："
ls -lht *.mp4 | head -20