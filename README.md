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
