#!/bin/sh

CONFIG_FILE="./config_files/hitnet.yaml"

i=3
while [ $i -lt 10000 ]
do
  # 格式化 OUTPUT_DIR 為四位數 (0000 ~ 9999)
  OUTPUT_NUM=$(printf "%04d" $i)

  # 修改 OUTPUT_DIR 的最後數字
  sed -i "s|OUTPUT_DIR: .*|OUTPUT_DIR: \"data/output/hitnet_pose_transformer_20250324_seed_${OUTPUT_NUM}\"|" "$CONFIG_FILE"

  # 執行 python 指令，seed 直接使用數值格式
  python train_net.py --config-file "$CONFIG_FILE" --seed $i

  # 增加計數
  i=$((i + 1))
done
