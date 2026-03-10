# Day 11 Dataset Integration Summary

- Dataset kind: **mimiciv**
- Input root: `D:\Article\_local\Datasets\physionet.org\files\mimiciv\3.1\hosp`
- Split level: **patient** (`subject_id`)
- Random seed: **42**

## Split overview

### overall
- records: 545825
- unique patients: 223397
- unique admissions: 545825
- avg seq len: 50.41
- median seq len: 36.00
- p95 seq len: 147.00
- avg diagnosis / procedure / medication per record: 11.66 / 1.57 / 37.18

### train
- records: 380922
- unique patients: 156377
- unique admissions: 380922
- avg seq len: 50.45
- median seq len: 36.00
- p95 seq len: 147.00
- avg diagnosis / procedure / medication per record: 11.65 / 1.58 / 37.22

### val
- records: 55061
- unique patients: 22339
- unique admissions: 55061
- avg seq len: 50.60
- median seq len: 36.00
- p95 seq len: 147.00
- avg diagnosis / procedure / medication per record: 11.68 / 1.58 / 37.33

### test
- records: 109842
- unique patients: 44681
- unique admissions: 109842
- avg seq len: 50.19
- median seq len: 36.00
- p95 seq len: 147.00
- avg diagnosis / procedure / medication per record: 11.68 / 1.57 / 36.94
