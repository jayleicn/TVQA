#!/usr/bin/env bash

mkdir -p data
cd data

# download
echo [log] Start downloading files
wget http://tvqa.cs.unc.edu/files/tvqa_qa_release.tar.gz -q --show-progress
wget http://tvqa.cs.unc.edu/files/tvqa_subtitles.tar.gz -q --show-progress
wget http://tvqa.cs.unc.edu/files/frm_cnt_cache.tar.gz -q --show-progress
wget http://tvqa.cs.unc.edu/files/det_visual_concepts.pickle.tar.gz -q --show-progress
wget http://tvqa.cs.unc.edu/files/tvqa_data.md5 -q --show-progress


# check
if ! md5sum -c tvqa_data.md5; then
    echo [Log] Found corrupted file, please re-download the files.
    exit 1
else
    echo [Log] All files are complete.
fi

# uncompress
echo [Log] Uncompressing data
tar -xzf tvqa_qa_release.tar.gz
tar -xzf tvqa_subtitles.tar.gz
tar -xzf frm_cnt_cache.tar.gz
tar -xzf det_visual_concepts.pickle.tar.gz

# external files
echo -n "Do you wish to download GloVe pretrained word vectors (822MB) as well? (Y/N)"
read answer
if [ "$answer" != "${answer#[Yy]}" ] ;then
    echo Yes
    wget http://nlp.stanford.edu/data/glove.6B.zip -q --show-progress
    unzip glove.6B.zip
else
    echo No
fi

