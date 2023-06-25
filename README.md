<div align= >

# <img align=center height=60px src="https://media1.giphy.com/media/YBypkW5Mlfz4znhCDD/giphy.gif?cid=ecf05e47pm5rbpvyte0zxbifp808lf554oysuxgmqjfaz04c&ep=v1_stickers_search&rid=giphy.gif&ct=s"> Hand Gesture Recognition


</div>
<div align="center">
   <img align="center" height="350px"  src="https://cdn.dribbble.com/users/662638/screenshots/4803914/thumbsupdribs2.gif" alt="logo">
   <br>

   ### ”Hello, Let us help you to recognize hand Gesture 🖐️“
</div>

<p align="center"> 
    <br> 
</p>

## <img align= center width=50px height=50px src="https://thumbs.gfycat.com/HeftyDescriptiveChimneyswift-size_restricted.gif"> Table of Contents

- <a href ="#about"> 📙 Overview</a>
- <a href ="#Started"> 💻 Get Started</a>
- <a href ="#Pipeline"> ⛓️ Project Pipeline</a>
- <a href ="#Modules">🤖  Modules</a>
    - <a href="#Preprocessing">🔁 Preprocessing Module</a>
    - <a href="#Feature">💪 Feature Extraction Module</a>
    - <a href="#Selection">✅ Model Selection</a>
    - <a href="#Performance">👌 Performance Module</a>
- <a href ="#Contributors"> ✨ Contributors</a>
- <a href ="#License"> 🔒 License</a>
<hr style="background-color: #4b4c60"></hr>

<a id = "about"></a>

## <img align="center"  height =50px src="https://user-images.githubusercontent.com/71986226/154076110-1233d7a8-92c2-4d79-82c1-30e278aa518a.gif"> Overview

<ul>
<li> Hand Gesture Recognition is machine learning project aim to recognize hand number from 0 => 5</li>
<li> This project ranked <strong>1st place</strong> among 21 teams with 81% accuracy, tested on unseen dataset.</li>

<li> Built using <a href="https://docs.python.org/3/">Python</a>.</li>
<li> Using Libraries</li>
<ul>
<li>opencv</li>
<li>numpy</li>
<li>matplotlib</li>
<li>pandas</li>
<li>scikit_image</li>
<li>scikit_learn</li>
<li>seaborn</li>
<li>joblib</li>
</ul>
<li>You can view
<a href="https://www.kaggle.com/datasets/evernext10/hand-gesture-of-the-colombian-sign-language">Data Set</a> which using to train module</li>
<li>You can view <a href="https://github.com/AbdelrahmanHamdyy/Hand-Gesture-Recognition/blob/main/Report.pdf"> Report</a></li>
</ul>
<hr style="background-color: #4b4c60"></hr>
<a id = "Started"></a>

## <img  align= center width=50px height=50px src="https://c.tenor.com/HgX89Yku5V4AAAAi/to-the-moon.gif"> How To Run

- First install the  <a href="https://github.com/AbdelrahmanHamdyy/Hand-Gesture-Recognition/blob/main/requirements.txt">needed packages</a>.</li> 

```sh
pip install -r requirements.txt
```

- Add the needed data to test the model in the `data` folder next to the `src` folder

```sh
├───data
├───src
│   ├───main.py
│   └───predict.py
...
```

- Navigate to the src directory

```sh
cd src
```

- Run the `main.py` file

```python
python main.py
```
- output 2 files <a href="https://github.com/AbdelrahmanHamdyy/Hand-Gesture-Recognition/blob/main/src/result.txt">"result.txt"</a> & <a href="https://github.com/AbdelrahmanHamdyy/Hand-Gesture-Recognition/blob/main/src/time.txt">"time.txt"</a>

<table>
<tr>
<th>File</th>
<th>Description</th>
</tr>
<tr>
<td>result</td>
<td>output classification result of every image by order </td>
</tr>
<tr>
<td>Time</td>
<td>output needed time to classifier image </td>
</tr>
</table>

<hr style="background-color: #4b4c60"></hr>

<a id = "Pipeline"></a>

## <img  align= center width=60px src="https://media3.giphy.com/media/JpHBXrvMkAiWdRemW8/giphy.gif?cid=ecf05e47foyhm8nq26e7mg0is4r80fd5m2khgyyfnn3ih5dc&ep=v1_stickers_search&rid=giphy.gif&ct=s"> Project Pipeline
<ol>
<li>📷 Read images</li>
<li>🔁 Preprocessing</li>
<li>💪 Get features</li>
<li>🪓 Split Training and Test Data</li>
<li>✅ Calculate accuracy</li>
<li>👌 Calculate performance analysis</li>

</ol>

<hr style="background-color: #4b4c60"></hr>
<a id = "Modules"></a>

## <img  align= center width=60px src="https://media0.giphy.com/media/j3nq3JkXp0bkFXcNlE/giphy.gif?cid=ecf05e47cftu8uth80woqhyl1kr7oy4m7zaihotdf9twrcaa&ep=v1_stickers_search&rid=giphy.gif&ct=s"> Modules
<a id = "Preprocessing"></a>

### <img align= center width=50px src="https://media0.giphy.com/media/321AaGDATXT8dq4MDC/giphy.gif?cid=ecf05e47r2eazdcsf8tqp6diz0z2o24gcho6yy4kj4lu6ctb&ep=v1_stickers_search&rid=giphy.gif&ct=s">Preprocessing Module
<ol>
<li>Apply gamma correction to adjust lighting</li>
<li>Segmentations</li>
<li>Convert image to YCbCr color space</li>
<li>Skin masking</li>
<li>Convert image to grayscale</li>
<li>Convert original image to gray scale</li>
<li>Erosion</li>
<li>Dilation</li>
<li>Draw left & right borders</li>
<li>Region Filling using Contours</li>
<li>Erosion</li>
<li>Masking eroded the image with the original one</li>
<li>Crop image to fit the hand exactly</li>
<li>Augmentation each image to enhance accuracy </li>
</ol>
<table>
<thead>
<th>Input</th>
<th>Output</th>
</thead>
<tr>
<td  align="center" width="15%"><img src="https://github-production-user-asset-6210df.s3.amazonaws.com/71986226/248538350-642e99c2-23f8-4dc3-b8d3-5a1694a195de.JPG?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20230625%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230625T075118Z&X-Amz-Expires=300&X-Amz-Signature=5921d041ff25c2e838ae1e83806a41c240ccbfad03721acf286e7d4b8d23138b&X-Amz-SignedHeaders=host&actor_id=71986226&key_id=0&repo_id=627202383"></td>
<td  align="center" width="15%"><img  height="250px" 
src="https://github-production-user-asset-6210df.s3.amazonaws.com/71986226/248538333-f40559d1-9460-4e07-a652-cc92d55105bb.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20230625%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230625T075129Z&X-Amz-Expires=300&X-Amz-Signature=9f4d8dd38b6353796773c89675f18529eadc41c94c555bccffecd956d4608e28&X-Amz-SignedHeaders=host&actor_id=71986226&key_id=0&repo_id=627202383"></td>
</tr>
<tr>
<td width="15%"><img src="https://github-production-user-asset-6210df.s3.amazonaws.com/71986226/248538378-20fbfc6e-8cb9-49de-a1c2-81bc5e0a1da9.JPG?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20230625%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230625T075102Z&X-Amz-Expires=300&X-Amz-Signature=0ef1ab6dc0829a076c648a2736247ce900ee560e17fabe5d688f954864fa5a7f&X-Amz-SignedHeaders=host&actor_id=71986226&key_id=0&repo_id=627202383"></td>
<td  align="center" width="15%"><img height="250px"src="https://github-production-user-asset-6210df.s3.amazonaws.com/71986226/248538368-943a9466-7065-4f1a-98a0-548827702d85.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20230625%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230625T075048Z&X-Amz-Expires=300&X-Amz-Signature=174a41530bcefee3daacd27fbb1ddf592f4fbdaf5f4b3132a2bc5c5331b755cf&X-Amz-SignedHeaders=host&actor_id=71986226&key_id=0&repo_id=627202383"></td>
</tr>
</table>
<a id = "Feature"></a>

### <img align= center height=60px src="https://media0.giphy.com/media/fw9KH5k7W2BVb78Wkq/200w.webp?cid=ecf05e472gayvziprwm50vr429mjzkk6lic31u4tegu821k7&ep=v1_stickers_search&rid=200w.webp&ct=s">Feature Extraction Module
<ol>
<li> Enter each image on Histogram of Oriented Gradients 
(HOG)</li>
<ol>
<li>Resizing</li>
<li>Gradient Computation</li>
<li>Cell Division</li>
<li>Orientation Binning</li>
<li>Histogram Calculations</li>
<li>Block Normalization</li>
<li>Feature Vector</li>
</ol>
<li>  Append array of features of each image in a list</li>
</ol>
<a id = "Selection"></a>

### <img align= center height=60px src="https://media0.giphy.com/media/YqJxBFX7cOPQSFO6gv/200w.webp?cid=ecf05e47q2pctv46mon3iqculvvgg8k8bruy7d5or1kf1jh8&ep=v1_stickers_search&rid=200w.webp&ct=s">Model Selection
<ol>
<li> Fitting training data and labeling into <strong>SVM model</strong></li>
<li>Dumping model</li>
<li>Getting classified data</li>
</ol>


### <img align= center height=60px src="https://media2.giphy.com/media/uhQuegHFqkVYuFMXMQ/giphy.gif?cid=ecf05e47s7yfhenqvyko8mhsuci17skn4q8fdik83l0k6j1m&ep=v1_stickers_search&rid=giphy.gif&ct=s">Performance Module
<strong>Calculate Confusion Matrix</strong>
<br>
<div align="center">
<img  width="75%" src="https://github-production-user-asset-6210df.s3.amazonaws.com/71986226/248539905-67634a34-4b23-46a4-8ba5-9bcacf1b4e3b.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWNJYAX4CSVEH53A%2F20230625%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20230625T081756Z&X-Amz-Expires=300&X-Amz-Signature=8a20205cbe6d10ae8f6d7451e395dd0b04defb64d7dba8a9c2c38344a0d35f7c&X-Amz-SignedHeaders=host&actor_id=71986226&key_id=0&repo_id=627202383">
</div>
<hr style="background-color: #4b4c60"></hr>

<a id ="Contributors"></a>

## <img  align="center" width= 70px height =55px src="https://media0.giphy.com/media/Xy702eMOiGGPzk4Zkd/giphy.gif?cid=ecf05e475vmf48k83bvzye3w2m2xl03iyem3tkuw2krpkb7k&rid=giphy.gif&ct=s"> Contributors 

<table align="center" >
  <tr>
    <td align="center"><a href="https://github.com/AbdelrahmanHamdyy"><img src="https://avatars.githubusercontent.com/u/67989900?v=4" width="150;" alt=""/><br /><sub><b>Abdelrahman Hamdy</b></sub></a><br /></td>
      <td align="center"><a href="https://github.com/ZiadSheriif" ><img src="https://avatars.githubusercontent.com/u/78238570?v=4" width="150;" alt=""/><br /><sub><b>Ziad Sherif</b></sub></a><br />
    </td>
       <td align="center"><a href="https://github.com/ZeyadTarekk"><img src="https://avatars.githubusercontent.com/u/76125650?v=4" width="150;" alt=""/><br /><sub><b>Zeyad Tarek</b></sub></a><br /></td>
     <td align="center"><a href="https://github.com/EslamAsHhraf"><img src="https://avatars.githubusercontent.com/u/71986226?v=4" width="150;" alt=""/><br /><sub><b>Eslam Ashraf</b></sub></a><br /></td>
  </tr>
</table>



<a id ="License"></a>

## 🔒 License

> **Note**: This software is licensed under MIT License, See [License](https://github.com/AbdelrahmanHamdyy/Hand-Gesture-Recognition/blob/main/LICENSE) for more information ©AbdelrahmanHamdyy.
