modelscope download \
  --dataset eugenelyj96/blinkflow \
  --include "train/*" \
  --local_dir ./blinkflow
cat train.zip.part-* > train.zip
unzip train.zip
