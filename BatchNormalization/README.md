# CUDA_BatchNormalization

Batch Normalization layer, which is frequently used in artificial neural networks, was created using CUDA.

## What is Batch Normalization ?

<a href="https://www.codecogs.com/eqnedit.php?latex=\mu&space;B\leftarrow&space;\frac{1}{m}\sum_{m}^{i=1}x_{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\mu&space;B\leftarrow&space;\frac{1}{m}\sum_{m}^{i=1}x_{i}" title="\mu B\leftarrow \frac{1}{m}\sum_{m}^{i=1}x_{i}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\sigma_{B}&space;^{2}\leftarrow&space;\frac{1}{m}\sum_{i=1}^{m}\left&space;(&space;x_{i}&space;-&space;\mu_{B}\right&space;)^{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sigma_{B}&space;^{2}\leftarrow&space;\frac{1}{m}\sum_{i=1}^{m}\left&space;(&space;x_{i}&space;-&space;\mu_{B}\right&space;)^{2}" title="\sigma_{B} ^{2}\leftarrow \frac{1}{m}\sum_{i=1}^{m}\left ( x_{i} - \mu_{B}\right )^{2}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=\hat{x}_{i}\leftarrow&space;\frac{x_{i}&space;-&space;\mu&space;_{B}}{\sqrt[]{\sigma&space;_{B}^{2}&space;&plus;&space;\epsilon&space;}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{x}_{i}\leftarrow&space;\frac{x_{i}&space;-&space;\mu&space;_{B}}{\sqrt[]{\sigma&space;_{B}^{2}&space;&plus;&space;\epsilon&space;}}" title="\hat{x}_{i}\leftarrow \frac{x_{i} - \mu _{B}}{\sqrt[]{\sigma _{B}^{2} + \epsilon }}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=y_{i}\leftarrow&space;\gamma&space;x\hat{}_{i}&space;&plus;&space;\beta&space;\equiv&space;BN_{\gamma,\beta}&space;\left&space;(&space;x_{i}&space;\right&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_{i}\leftarrow&space;\gamma&space;x\hat{}_{i}&space;&plus;&space;\beta&space;\equiv&space;BN_{\gamma,\beta}&space;\left&space;(&space;x_{i}&space;\right&space;)" title="y_{i}\leftarrow \gamma x\hat{}_{i} + \beta \equiv BN_{\gamma,\beta} \left ( x_{i} \right )" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=x\hat{}_{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x\hat{}_{i}" title="x\hat{}_{i}" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=y_{i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_{i}" title="y_{i}" /></a> part runs parallel