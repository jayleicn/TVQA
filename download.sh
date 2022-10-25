#!/usr/bin/env bash

mkdir -p data
cd data

# download
echo [log] Start downloading files
wget https://nlp.cs.unc.edu/data/jielei/tvqa/files/tvqa_qa_release.tar.gz -q --show-progress
wget https://nlp.cs.unc.edu/data/jielei/tvqa/files/tvqa_subtitles.tar.gz -q --show-progress
wget https://nlp.cs.unc.edu/data/jielei/tvqa/files/frm_cnt_cache.tar.gz -q --show-progress
wget https://nlp.cs.unc.edu/data/jielei/tvqa/files/det_visual_concepts_hq.pickle.tar.gz -q --show-progress
wget https://nlp.cs.unc.edu/data/jielei/tvqa/files/tvqa_data.md5 -q --show-progress


# uncompress
echo [Log] Uncompressing data
tar -xzf tvqa_qa_release.tar.gz
tar -xzf tvqa_subtitles.tar.gz
tar -xzf frm_cnt_cache.tar.gz
tar -xzf det_visual_concepts_hq.pickle.tar.gz

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

