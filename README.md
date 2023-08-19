# KPCE

Code for our ACL 2023 paper: <a href='https://arxiv.org/abs/2305.01876'>Causality-aware Concept Extraction based on Knowledge-guided Prompting</a>

## Requirements
- torch == 1.4.0
- transformers == 4.2.0

## Dataset
sample dataset is release on dataset/

### format
- Chinese data: abstract \t 0/1 strings for the start position \t 0/1 strings for the end position
```
袁希治 1946年生，湖北汉阳人。二级演员。1966年毕业于湖北省戏曲学校楚剧科演员专业。	0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0	0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 人物
```
- English data: abstract \t start position \t answer length \t topic
```
California is a state in the Pacific Region of the United States of America. With 39.5 million residents across a total area of about 163,696 square miles 423,970 km2, California is the most populous U.S. state and the third-largest by area, and is also the world's thirty-fourth most populous subnational entity. California is also the most populated subnational entity in North America, and has its state capital in Sacramento. The Greater Los Angeles area and the San Francisco Bay Area are the nation's second- and fifth-most populous urban regions, with 18.7 million and 9.7 million residents respectively. Los Angeles is California's most populous city, and the country's second-most populous, after New York City. California also has the nation's most populous county, Los Angeles County, and its largest county by area, San Bernardino County. The City and County of San Francisco is both the country's second most densely populated major city after New York City and the fifth most densely populated county, behind only four of the five New York City boroughs.	654	4	city	location
```

## Run on sample data
```
python main.py
```

## results
the results which are labeled by the volunteers are release on results/

## Reference
- google-research bert: <https://github.com/google-research/bert>
- transformer: <https://github.com/huggingface/transformers>
- tensorflow: <https://github.com/tensorflow/tensorflow>

## Citation

If you find our paper or resources useful, please kindly cite our paper. If you have any questions, please [contact us](mailto:syyuan21@m.fudan.edu.cn)!

```latex
@inproceedings{yuan-etal-2023-causality,
    title = "Causality-aware Concept Extraction based on Knowledge-guided Prompting",
    author = "Yuan, Siyu  and
      Yang, Deqing  and
      Liu, Jinxi  and
      Tian, Shuyu  and
      Liang, Jiaqing  and
      Xiao, Yanghua  and
      Xie, Rui",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.514",
    doi = "10.18653/v1/2023.acl-long.514",
    pages = "9255--9272",
    abstract = "Concepts benefit natural language understanding but are far from complete in existing knowledge graphs (KGs). Recently, pre-trained language models (PLMs) have been widely used in text-based concept extraction (CE). However, PLMs tend to mine the co-occurrence associations from massive corpus as pre-trained knowledge rather than the real causal effect between tokens. As a result, the pre-trained knowledge confounds PLMs to extract biased concepts based on spurious co-occurrence correlations, inevitably resulting in low precision. In this paper, through the lens of a Structural Causal Model (SCM), we propose equipping the PLM-based extractor with a knowledge-guided prompt as an intervention to alleviate concept bias. The prompt adopts the topic of the given entity from the existing knowledge in KGs to mitigate the spurious co-occurrence correlations between entities and biased concepts. Our extensive experiments on representative multilingual KG datasets justify that our proposed prompt can effectively alleviate concept bias and improve the performance of PLM-based CE models.",
}
```
