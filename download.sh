#!/usr/bin/env bash

mkdir -p data
cd data

# download
wget http://tvqa.cs.unc.edu/files/tvqa_qa_release.tar.gz -q --show-progress
wget http://tvqa.cs.unc.edu/files/tvqa_subtitles.tar.gz -q --show-progress
wget http://tvqa.cs.unc.edu/files/frm_cnt_cache.tar.gz -q --show-progress
wget http://tvqa.cs.unc.edu/files/tvqa_data.md5 -q --show-progress

# check
if ! md5sum -c tvqa_data.md5; then
    exit 1
fi

# uncompress
tar -xzf tvqa_qa_release.tar.gz
tar -xzf tvqa_subtitles.tar.gz
tar -xzf frm_cnt_cache.tar.gz
