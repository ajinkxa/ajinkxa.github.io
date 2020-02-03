This is a fun project that I did to try out the pretrained models on datastream created from google images. It was a good practice in building data pipelines, creating image transforms, importing the pretrained models and optimizing the learning rate. 

I searched for images of some of the famous football players on google images and fed the first 200 images to this model to see how well the model is able to train on an uncurated noisy data.

Recognizing players can be an useful application in sport analytics. A lot of in-game metrics like xG and xA scores, heatmaps depend on some kind of deep learning algorithms running in the backend.

# **Setting up the libraries**


This step mounts the google drive in working enironment which would be used as a backend storage.


```python
from google.colab import drive
drive.mount('/content/gdrive')
```

    Drive already mounted at /content/gdrive; to attempt to forcibly remount, call drive.mount("/content/gdrive", force_remount=True).
    


```python
from fastai.vision import *
```


```python
import warnings
warnings.filterwarnings("ignore")
```

# **Creating data pipeline**


```python
ls
```

    [0m[01;34mgdrive[0m/  [01;34msample_data[0m/
    


```python
cd gdrive/My Drive/fastai_player_recog/
```

    /content/gdrive/My Drive/fastai_player_recog
    


```python
folder = 'cr'
file = 'cr.csv'
```


```python
folder = 'messi'
file = 'messi.csv'
```


```python
folder = 'suarez'
file = 'suarez.csv'
```


```python
path = Path('data/players')
dest = path/folder
dest.mkdir(parents=True, exist_ok=True)
```


```python
path.ls
```




    <bound method <lambda> of PosixPath('data/players')>




```python
classes = ['cr','messi','suarez']
```


```python
download_images(path/file, dest, max_pics=200)
```






```python
for c in classes:
    print(c)
    verify_images(path/c, delete=True, max_size=500)
```

    cr
    





    messi
    





    suarez
    





    cannot identify image file <_io.BufferedReader name='data/players/suarez/00000070.jpg'>
    cannot identify image file <_io.BufferedReader name='data/players/suarez/00000096.png'>
    

# **Modeling - Data Transformation**

**Transforms**


```python
path = Path('data/players/')
```


```python
path
```




    PosixPath('data/players')




```python
transforms = get_transforms()
```


```python
np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=transforms, size=224, bs=32).normalize(imagenet_stats)
```


```python
data.classes
```




    ['cr', 'messi', 'suarez']




```python
data.show_batch(rows=3, figsize=(7,8))
```


<img src="images/pr1.png?raw=true"/>



As you can see above, the dataset is uncurated with lot of noisy pictures. For example, the last picture is a sketch. There are several such images that are inaccurate/improper but we shall be using them to induce noise in our models.


```python
data.classes, data.c, len(data.train_ds), len(data.valid_ds)
```




    (['cr', 'messi', 'suarez'], 3, 383, 95)



# **Training**

As seen below, we are able to achieve around 87%-89% accuracy by optimizing the learning rate. If we now remove the noise from the data, we shall quite easily be able to achieve an even higher accuracy. It'd be interesting to see which images are misclassified the most.


```python
learn = cnn_learner(data, models.resnet152, metrics=error_rate)
```


```python
learn.fit_one_cycle(30)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.153895</td>
      <td>0.486600</td>
      <td>0.147368</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.176932</td>
      <td>0.453679</td>
      <td>0.147368</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.186371</td>
      <td>0.438741</td>
      <td>0.136842</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.176964</td>
      <td>0.498095</td>
      <td>0.126316</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.187868</td>
      <td>0.495250</td>
      <td>0.157895</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.173565</td>
      <td>0.450809</td>
      <td>0.147368</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.167043</td>
      <td>0.444867</td>
      <td>0.115789</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.161417</td>
      <td>0.477105</td>
      <td>0.178947</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.164916</td>
      <td>0.773907</td>
      <td>0.168421</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.157456</td>
      <td>0.554971</td>
      <td>0.189474</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.157130</td>
      <td>0.592353</td>
      <td>0.136842</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.168478</td>
      <td>0.647615</td>
      <td>0.136842</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.172660</td>
      <td>0.529567</td>
      <td>0.168421</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.178256</td>
      <td>0.557514</td>
      <td>0.157895</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.172853</td>
      <td>0.423964</td>
      <td>0.168421</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.170725</td>
      <td>0.382557</td>
      <td>0.115789</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.156468</td>
      <td>0.471120</td>
      <td>0.147368</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.133140</td>
      <td>0.533500</td>
      <td>0.136842</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.115539</td>
      <td>0.595399</td>
      <td>0.147368</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>19</td>
      <td>0.108522</td>
      <td>0.631604</td>
      <td>0.136842</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>20</td>
      <td>0.105572</td>
      <td>0.566845</td>
      <td>0.147368</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>21</td>
      <td>0.103867</td>
      <td>0.629708</td>
      <td>0.157895</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>22</td>
      <td>0.096924</td>
      <td>0.590672</td>
      <td>0.147368</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>23</td>
      <td>0.091466</td>
      <td>0.573798</td>
      <td>0.136842</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>24</td>
      <td>0.080457</td>
      <td>0.582823</td>
      <td>0.136842</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>25</td>
      <td>0.077717</td>
      <td>0.551564</td>
      <td>0.136842</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>26</td>
      <td>0.075651</td>
      <td>0.545016</td>
      <td>0.126316</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>27</td>
      <td>0.064027</td>
      <td>0.561220</td>
      <td>0.126316</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>28</td>
      <td>0.060929</td>
      <td>0.569121</td>
      <td>0.126316</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>29</td>
      <td>0.054775</td>
      <td>0.578649</td>
      <td>0.126316</td>
      <td>00:09</td>
    </tr>
  </tbody>
</table>



```python
learn.save('stage-1')
```


```python
learn.unfreeze()
```


```python
learn.lr_find(stop_div=False, num_it=200)
```



    <div>
        <style>
            /* Turns off some styling */
            progress {
                /* gets rid of default border in Firefox and Opera. */
                border: none;
                /* Needs to be in here for Safari polyfill so background images work as expected. */
                background-size: auto;
            }
            .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
                background: #F44336;
            }
        </style>
      <progress value='18' class='' max='19', style='width:300px; height:20px; vertical-align: middle;'></progress>
      94.74% [18/19 02:59<00:09]
    </div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.029879</td>
      <td>#na#</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.039062</td>
      <td>#na#</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.040193</td>
      <td>#na#</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.038014</td>
      <td>#na#</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.036690</td>
      <td>#na#</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.050070</td>
      <td>#na#</td>
      <td>00:10</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.065584</td>
      <td>#na#</td>
      <td>00:10</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.091199</td>
      <td>#na#</td>
      <td>00:10</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.370187</td>
      <td>#na#</td>
      <td>00:10</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.688983</td>
      <td>#na#</td>
      <td>00:10</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.859709</td>
      <td>#na#</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.963872</td>
      <td>#na#</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>12</td>
      <td>1.173712</td>
      <td>#na#</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>13</td>
      <td>1.679814</td>
      <td>#na#</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>14</td>
      <td>2.981434</td>
      <td>#na#</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>15</td>
      <td>14.075550</td>
      <td>#na#</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>16</td>
      <td>116.549461</td>
      <td>#na#</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>17</td>
      <td>650.312622</td>
      <td>#na#</td>
      <td>00:09</td>
    </tr>
  </tbody>
</table><p>

    <div>
        <style>
            /* Turns off some styling */
            progress {
                /* gets rid of default border in Firefox and Opera. */
                border: none;
                /* Needs to be in here for Safari polyfill so background images work as expected. */
                background-size: auto;
            }
            .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
                background: #F44336;
            }
        </style>
      <progress value='1' class='' max='11', style='width:300px; height:20px; vertical-align: middle;'></progress>
      9.09% [1/11 00:01<00:17 735.8074]
    </div>



    LR Finder is complete, type {learner_name}.recorder.plot() to see the graph.
    


```python
learn.recorder.plot()
```


<img src="images/pr2.png?raw=true"/>



```python
learn.fit_one_cycle(10, max_lr=slice(1e-03,1e-06))

```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.067066</td>
      <td>0.845543</td>
      <td>0.178947</td>
      <td>00:11</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.042421</td>
      <td>1.085019</td>
      <td>0.231579</td>
      <td>00:11</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.106151</td>
      <td>1.190116</td>
      <td>0.284211</td>
      <td>00:11</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.217121</td>
      <td>5.310340</td>
      <td>0.610526</td>
      <td>00:11</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.304309</td>
      <td>3.054987</td>
      <td>0.578947</td>
      <td>00:11</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.348517</td>
      <td>1.629803</td>
      <td>0.368421</td>
      <td>00:11</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.353958</td>
      <td>0.877336</td>
      <td>0.200000</td>
      <td>00:11</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.319123</td>
      <td>0.604664</td>
      <td>0.168421</td>
      <td>00:11</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.277269</td>
      <td>0.568268</td>
      <td>0.136842</td>
      <td>00:11</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.257772</td>
      <td>0.594630</td>
      <td>0.126316</td>
      <td>00:11</td>
    </tr>
  </tbody>
</table>



```python
learn.save('stage-2')
```


```python
learn.freeze_to(100)
```


```python
learn.fit_one_cycle(30)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.199813</td>
      <td>0.610031</td>
      <td>0.136842</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.169803</td>
      <td>0.593857</td>
      <td>0.136842</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.145744</td>
      <td>0.590429</td>
      <td>0.136842</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.139687</td>
      <td>0.596360</td>
      <td>0.147368</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.137717</td>
      <td>0.611183</td>
      <td>0.136842</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.122393</td>
      <td>0.663416</td>
      <td>0.136842</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.124853</td>
      <td>0.701337</td>
      <td>0.126316</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.136023</td>
      <td>0.690178</td>
      <td>0.136842</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.134810</td>
      <td>0.711177</td>
      <td>0.136842</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.133105</td>
      <td>0.759611</td>
      <td>0.136842</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.122329</td>
      <td>0.687816</td>
      <td>0.157895</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.121954</td>
      <td>0.623489</td>
      <td>0.157895</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.108721</td>
      <td>0.719655</td>
      <td>0.157895</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.099113</td>
      <td>0.656762</td>
      <td>0.147368</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.094180</td>
      <td>0.653667</td>
      <td>0.157895</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.089522</td>
      <td>0.598895</td>
      <td>0.115789</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.082002</td>
      <td>0.629596</td>
      <td>0.126316</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.093078</td>
      <td>0.576757</td>
      <td>0.115789</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.085515</td>
      <td>0.558922</td>
      <td>0.115789</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>19</td>
      <td>0.082467</td>
      <td>0.533377</td>
      <td>0.115789</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>20</td>
      <td>0.076356</td>
      <td>0.555016</td>
      <td>0.126316</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>21</td>
      <td>0.073800</td>
      <td>0.568873</td>
      <td>0.126316</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>22</td>
      <td>0.078990</td>
      <td>0.541078</td>
      <td>0.115789</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>23</td>
      <td>0.076296</td>
      <td>0.547799</td>
      <td>0.115789</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>24</td>
      <td>0.077476</td>
      <td>0.534094</td>
      <td>0.126316</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>25</td>
      <td>0.071813</td>
      <td>0.546575</td>
      <td>0.115789</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>26</td>
      <td>0.065507</td>
      <td>0.554435</td>
      <td>0.115789</td>
      <td>00:09</td>
    </tr>
    <tr>
      <td>27</td>
      <td>0.063554</td>
      <td>0.564250</td>
      <td>0.126316</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>28</td>
      <td>0.058297</td>
      <td>0.568706</td>
      <td>0.126316</td>
      <td>00:08</td>
    </tr>
    <tr>
      <td>29</td>
      <td>0.055142</td>
      <td>0.556144</td>
      <td>0.126316</td>
      <td>00:08</td>
    </tr>
  </tbody>
</table>



```python
interp = ClassificationInterpretation.from_learner(learn)
```





Below shown are the images with highest losses. As you can see, most of it is noise. For example, the third image is an edited version, 6th and 9th images are graphic designs and not raw images.


```python
interp.plot_top_losses(9, figsize=(15,11))
```


<img src="images/pr3.png?raw=true"/>
<img src="images/pr4.png?raw=true"/>




```python
interp.plot_confusion_matrix()
```


<img src="images/pr5.png?raw=true"/>


The confusion matrix thus shows that the model is quite adept at classifying the images. It currently shows an accuracy of 87%-89%. The accuracy can further be improved by 
- increasing the data
- inceasing the computation effort
- removing the noise from data


```python

```
