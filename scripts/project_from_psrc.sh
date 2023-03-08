DUMP_FOLDER='./raw'
SRC_DATA_FOLDER=$DUMP_FOLDER/projection_from_psrc
mkdir -p $SRC_DATA_FOLDER
mkdir -p $SRC_DATA_FOLDER/cache

python data/project_from_psrc.py \
--dataset-name-or-paths glue glue glue glue glue \
--dataset-configs cola sst2 mrpc qqp stsb \
--prompt-templates-configs None None None None None \
--cache-dir $SRC_DATA_FOLDER/cache \
--output-dir $SRC_DATA_FOLDER \
--highlight-variables \
--add-source-metadata \
--num-proc 16