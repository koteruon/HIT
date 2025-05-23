#!/bin/sh

CONFIG_FILE="./config_files/hitnet_pretrain_skateformer.yaml"

i=17
while [ $i -lt 10000 ]
do
  # 格式化 OUTPUT_DIR 為四位數 (0000 ~ 9999)
  OUTPUT_NUM=$(printf "%04d" $i)

  # 修改 OUTPUT_DIR 的最後數字
  # sed -i "s|OUTPUT_DIR: .*|OUTPUT_DIR: \"data/output/hitnet_pose_transformer_with_pretrain_skateformer_20250318_${OUTPUT_NUM}\"|" "$CONFIG_FILE"
  sed -i "s|OUTPUT_DIR: .*|OUTPUT_DIR: \"data/output/hitnet_pose_transformer_with_pretrain_skateformer_20250521_seed_${OUTPUT_NUM}\"|" "$CONFIG_FILE"

  # 執行 python 指令，seed 直接使用數值格式
  python train_net.py --config-file "$CONFIG_FILE" --seed $i

  # 增加計數
  i=$((i + 1))
done
