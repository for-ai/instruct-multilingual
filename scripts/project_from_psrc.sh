# The native format with a lots of metadata
DUMP_FOLDER='./raw'
SRC_DATA_FOLDER=$DUMP_FOLDER/projection_from_psrc
mkdir -p $SRC_DATA_FOLDER
mkdir -p $SRC_DATA_FOLDER/cache

python data/project_from_psrc.py \
--dataset-name-or-paths nq_open \
--dataset-configs None \
--prompt-templates-configs None \
--cache-dir $SRC_DATA_FOLDER/cache \
--output-dir $SRC_DATA_FOLDER \
--highlight-variables \
--add-source-metadata \
--num-proc 16

# The xP3 format
DUMP_FOLDER='./raw'
SRC_DATA_FOLDER=$DUMP_FOLDER/projection_from_psrc
mkdir -p $SRC_DATA_FOLDER
mkdir -p $SRC_DATA_FOLDER/cache

python data/project_from_psrc.py \
--dataset-name-or-paths nq_open \
--dataset-configs None \
--prompt-templates-configs None \
--cache-dir $SRC_DATA_FOLDER/cache \
--output-dir $SRC_DATA_FOLDER \
--xp3-format \
--num-proc 16