export HOME_PREFIX=/your/local/data/path


export OBJAVERSE_HOUSES_BASE_DIR=${HOME_PREFIX}/houses/objaverse_houses
export OBJAVERSE_HOUSES_DIR=${HOME_PREFIX}/houses/objaverse_houses/houses_2023_07_28
export OBJAVERSE_DATA_BASE_DIR=${HOME_PREFIX}/assets/objaverse_houses
export OBJAVERSE_DATA_DIR=${HOME_PREFIX}/assets/objaverse_assets/2023_07_28
export OBJAVERSE_ANNOTATIONS_PATH=${HOME_PREFIX}/assets/objaverse_assets/2023_07_28/annotations.json.gz


echo "Download objaverse assets and annotation"
if [ ! -f $OBJAVERSE_DATA_BASE_DIR/2023_07_28/annotations.json.gz ] ; then
  python -m objathor.dataset.download_annotations --version 2023_07_28 --path $OBJAVERSE_DATA_BASE_DIR
else
  echo "Annotations already downloaded"
fi

if [ ! -d $OBJAVERSE_DATA_BASE_DIR/2023_07_28/assets ] ; then
  python -m objathor.dataset.download_assets --version 2023_07_28 --path $OBJAVERSE_DATA_BASE_DIR
else
  echo "Assets already downloaded"
fi

echo "Download objaverse houses"
if [ ! -f $OBJAVERSE_HOUSES_BASE_DIR/houses_2023_07_28/val.jsonl.gz ] ; then
  python download_objaverse_houses.py --save_dir $OBJAVERSE_HOUSES_BASE_DIR --subset val
else
  echo "Houses already downloaded"
fi
