# The Levenchuk Post Project

![tLPp](img/logo.png)

Status: under development.

Text Generative application based on char-rnn models.

Tensorflow is required.


Usage:

0. Get raw data.
```
cd src
python download --nb 4
```

Will result in two files in save-dir:

- post-list.txt -- urls for every public entry in selected Livejournal

- dump.txt -- livejournal dump in the following format:

 ```
 <post href='lj.com/xxxx.html'> Title, just for easy dropping
 Content in that post with http://some-links.google.com.
 </post>
 ```


1. Prepare data.

```
python prepare.py
```

Will result in three files:

- train.npy -- encoded train set
- test.npy -- encoded test set
- vocab.json -- encoding table

2. Train baseline model:

Not ready
```
python train.py
```

3. Run the bot:
Not ready



 
