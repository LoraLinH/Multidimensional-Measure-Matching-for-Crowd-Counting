# Multidimensional Measure Matching for Crowd Counting
Extension of IJCAI 2021 paper 'Direct Measure Matching for Crowd Counting'

## Train
1. Dowload Dataset JHU++ or UCF-QNRF.
2. Preprocess them by 'preprocess_dataset.py' in [Link](https://github.com/LoraLinH/Boosting-Crowd-Counting-via-Multifaceted-Attention) or 'preprocess_dataset_ucf.py'.
3. Dowload the pretrained transformer backbone pvt_v2_b3 from [PVT](https://github.com/whai362/PVT/tree/v2/classification) and put it in directory 'models'.
4. Change the path to where your data and models are located in 'Train.py'.
5. Run 'Train.py'.
6. Wait patiently and happily for the program to finish.
7. Then you will get a good counting model!

## Test
1. Dowload Dataset JHU++ or UCF-QNRF.
2. Preprocess them by 'preprocess_dataset.py' in [Link](https://github.com/LoraLinH/Boosting-Crowd-Counting-via-Multifaceted-Attention) or 'preprocess_dataset_ucf.py'.
3. JHU Model [Link](https://drive.google.com/file/d/12VRSZ5K8QDS1ZssoiZWJcHAAn9j8na3J/view?usp=sharing); UCF Model [Link](https://drive.google.com/file/d/1ghl33nr_at18g-99BmlhWvvywncJ176Z/view?usp=sharing)
4. Dowload the pretrained transformer backbone pvt_v2_b3 from [PVT](https://github.com/whai362/PVT/tree/v2/classification) and put it in directory 'models'.
5. Change the path to where your data and models are located in 'Test.py'.
6. Run 'Test.py'.
