#!/bin/bash
echo "Unpacking data to directory..."
tar -xzvf data_en_es.tar.gz
mkdir data_en_es
mv en_es.* data_en_es/
tar -xzvf data_es_en.tar.gz
mkdir data_es_en
mv es_en.* data_es_en/
tar -xzvf data_fr_en.tar.gz
mkdir data_fr_en
mv fr_en.* data_fr_en/
echo "Done!"