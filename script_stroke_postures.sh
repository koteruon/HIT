#!/bin/sh

CONFIG_FILE="./config_files/hitnet_stroke_postures.yaml"

i=7
while [ $i -lt 50 ]
do
  # 格式化 OUTPUT_DIR 為四位數 (0000 ~ 9999)
  OUTPUT_NUM=$(printf "%02d" $i)

  # 修改 OUTPUT_DIR 的最後數字
  # sed -i "s|OUTPUT_DIR: .*|OUTPUT_DIR: \"data/output/hitnet_pose_transformer_with_skateformer_20250318_${OUTPUT_NUM}\"|" "$CONFIG_FILE"
  sed -i "s|OUTPUT_DIR: .*|OUTPUT_DIR: \"data/output/hitnet_pose_transformer_stroke_postures_20250326_seed_${OUTPUT_NUM}\"|" "$CONFIG_FILE"

  # 執行 python 指令，seed 直接使用數值格式
  python train_net.py --config-file "$CONFIG_FILE" --seed $i

  # 增加計數
  i=$((i + 1))
done
