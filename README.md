# Georgios branch

Here I'll be pushing my staff. Please don't merge with main.
Experiments on semi-supervised learning for automatic speech recognition.

Architectures (to be experimented):
- LSTM
- Bi-directional LSTM
- Deep LSTM

Dataset:
- TIMIT

### Activate virtual environment

NO VIRTUAL ENVIRONMENT HERE.

### Install dependencies

```bash
pip install -r requirements.txt
```

### Install dataset
```bash
wget https://data.deepai.org/timit.zip
unzip timit.zip
rm timit.zip
```

## Running

### Test
In MacOS with Python3 this should work well:
```bash
python app_test.py
```

This can be used to plot the result after running MFCC for the TIMIT dataset.
```bash
python3 app_test.py --n_fft=512
```
