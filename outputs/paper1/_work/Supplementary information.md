# Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis

Fujin Wang a,b,∗, Zhi Zhaia,b,∗, Zhibin Zhaoa,b,∗∗, Yi Dia,b, Xuefeng Chena,b,∗∗

aNational and Local Joint Engineering Research Center of Equipment Operation Safety and Intelligent Monitoring, Xi’an Jiaotong University, Xi’an, 710049, PR China

bSchool of Mechanical Engineering, Xi’an Jiaotong University, Xi’an, 710049, PR China

# Battery Parameters

Chemical: $\mathrm { L i N i } _ { 0 . 5 } \mathrm { C o } _ { 0 . 2 } \mathrm { M n } _ { 0 . 3 } \mathrm { O } _ { 2 }$

Size: $\Phi 1 8 . 3 _ { - 0 . 2 } ^ { + 0 . 2 } \times 6 4 . 9 _ { - 0 . 3 } ^ { + 0 . 3 }$ mm

Weight: < 46g

Nominal capacity: 2000 mAh

Lower cut-off voltage: 2.5 V

Upper cut-off voltage: 4.2 V

Nominal voltage: 3.6 V

Figure S.1: The battery degradation platform in our laboratory. All batteries cycled to failure in 40-channel ACTS-5V10A-GGS-D at room temperature.

# Supplementary Note 1. Dataset description

We use batch 1 to batch 6 to represent the 6 charging and discharging protocols, respectively. All batches except batch 2 consist of 8 batteries, while batch 2 contains 15 batteries. The voltage, current, temperature, and discharge capacity curves of the first battery in each batch are given in Figure S.4.

# Supplementary Note 1.1. batch 1

The batteries in batch 1 were cycled under a fixed charging and discharging strategy. All batteries were charged to 4.2 V at 2 C with constant voltage and constant current (CC-CV) mode and then discharged to 2.5 V at 1 C.

# Supplementary Note 1.2. batch 2

Batch 2 contains 15 batteries, and its charging and discharging strategy is similar to batch 1. All batteries were charged to 4.2 V at 3 C with CC-CV mode and then discharged to 2.5 V at 1 C.

# Supplementary Note 1.3. batch 3

Batch 3 has a more complex protocol than that of the first two batches. All batteries were charged at 2 C with CC-CV mode. Then they were discharged to 2.5 V with a current value of x C, where $x \in \{ 0 . 5 , 1 , 2 , 3 , 5 \}$ .

# Supplementary Note 1.4. batch 4

Batch 4 is similar to batch 3. The batteries were charged at 2 C with CC-CV mode and then discharged to 3.0 V with the same current as batch 3.

# Supplementary Note 1.5. batch 5

Batch 5 follows the random walking strategy, thereby the entire process of charging and discharging are more closely with real-life usage. Specifically, all cells are charged to 4.2 V at 1 C with CC-CV mode and then discharge to 3.0 V. The discharge current is a random integer in the range of [2, 8] ampere and the duration is in the range of [2, 6] minutes.

# Supplementary Note 1.6. batch 6

In batch 6, we simulated the charging and discharging strategy of a satellite in geosynchronous earth orbit (GEO). The batteries of GEO satellites only supply power during the shadow period of the earth, and the depth-of-discharge (DOD) is generally less than 80%. The duration of each discharge is determined by the duration of the Earth’s shadow. Therefore, the discharge duration of each cycle is different, and the DOD is also different.

Specifically, GEO satellites experience the Earth’s shadow during the spring and autumn equinoxes each year, approximately 23 days before and after the equinoxes, resulting in a total duration of approximately 46 days for each occurrence. During the period of the Earth’s shadow, the duration of the shadow varies in a regular pattern every day, as depicted in Table S.1 and Figure A.12. The duration of the shadow initially increases gradually before decreasing. An illustration of the discharge capacity curve of battery 1 in batch 6 is given in Figure S.3.

Table S.1: The discharge duration of each cycle of the GEO satellite in the Earth’s shadow period [1]. 

<table><tr><td rowspan="2">Cycle number</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td><td>13</td><td>14</td><td>15</td><td>16</td><td>17</td><td>18</td><td>19</td><td>20</td><td>21</td><td>22</td><td>23</td></tr><tr><td>46</td><td>45</td><td>44</td><td>43</td><td>42</td><td>41</td><td>40</td><td>39</td><td>38</td><td>37</td><td>36</td><td>35</td><td>34</td><td>33</td><td>32</td><td>31</td><td>30</td><td>29</td><td>28</td><td>27</td><td>26</td><td>25</td><td>24</td></tr><tr><td>Discharge duration</td><td>5</td><td>20</td><td>34</td><td>41</td><td>46</td><td>50</td><td>54</td><td>56</td><td>58</td><td>60</td><td>62</td><td>64</td><td>68</td><td>69</td><td>70</td><td>71</td><td>72</td><td>72</td><td>72</td><td>72</td><td>72</td><td>72</td><td>72</td></tr></table>

Figure S.2: The discharge duration of each cycle of the GEO satellite in the Earth’s shadow period.

Figure S.3: An illustration of the discharge capacity curve of battery 1 in batch 6 during the whole life cycles.

Figure S.4: The voltage, current, temperature, and discharge capacity curves during the whole life cycles of the XJTU battery dataset. Each row represents the curve of the first battery in the corresponding batch. From batch 3 to batch 6, the experimental settings are all incomplete discharge protocols, so we measure the discharge capacity after every few cycles, and finally interpolate the discharge capacity curve to obtain the red degradation trajectory curve in the figure.

# Supplementary Note 2. Feature extraction

For the XJTU and TJU datasets, the voltage curve whose value is in the range [4.0, 4.2] V is selected, and for the MIT and HUST datasets, the voltage range is [3.4, 3.6] V. Suppose the time range corresponding to the selected data is $[ t _ { \mathrm { s t a r t } } , t _ { \mathrm { e n d } } ]$ , the mean, standard deviation, kurtosis, skewness, charging time, accumulated charge, curve slope, and curve entropy are calculated as:

$$
\bar {x} = \frac {1}{n} \sum_ {i = 1} ^ {n} \mathbf {x} (i), \tag {1}
$$

$$
\sigma = \sqrt {\frac {1}{n - 1} \sum_ {i = 1} ^ {n} \left(\mathbf {x} (i) - \bar {x}\right) ^ {2}}, \tag {2}
$$

$$
\text { kurtosis } = \frac {\sum_ {i = 1} ^ {n} (\mathbf {x} (i) - \bar {x}) ^ {4}}{(n - 1) \sigma^ {4}}, \tag {3}
$$

$$
\mathrm{skewness} = \frac {\sum_ {i = 1} ^ {n} \left(\mathbf {x} (i) - \bar {x}\right) ^ {3}}{(n - 1) \sigma^ {3}}, \tag {4}
$$

$$
\Delta t = t _ {\mathrm{end}} - t _ {\mathrm{start}}, \tag {5}
$$

$$
\Delta Q = \int_ {t _ {\mathrm{start}}} ^ {t _ {\mathrm{end}}} I \cdot d t, \tag {6}
$$

$$
\mathrm{slope} = \frac {V _ {\mathrm{end}} - V _ {\mathrm{start}}}{t _ {\mathrm{end}} - t _ {\mathrm{start}}}, \tag {7}
$$

$$
\text { entropy } = - \sum_ {i = 1} ^ {n} p _ {i} \cdot \log (p _ {i}), \tag {8}
$$

where $p _ { i }$ is normalized value of a curve. For current curve with value between [0.5, 0.1] A, the features are calculated in the same way as above.

Table S.2: The number of test batteries in each dataset. 

<table><tr><td>Dataset</td><td>Batch</td><td>Total number</td><td>Number of test set</td></tr><tr><td rowspan="6">XJTU</td><td>1</td><td>8</td><td>2</td></tr><tr><td>2</td><td>15</td><td>3</td></tr><tr><td>3</td><td>8</td><td>2</td></tr><tr><td>4</td><td>8</td><td>2</td></tr><tr><td>5</td><td>8</td><td>2</td></tr><tr><td>6</td><td>8</td><td>2</td></tr><tr><td rowspan="3">TJU</td><td>1</td><td>66</td><td>13</td></tr><tr><td>2</td><td>55</td><td>11</td></tr><tr><td>3</td><td>9</td><td>2</td></tr><tr><td>HUST</td><td>-</td><td>77</td><td>20</td></tr><tr><td>MIT</td><td>-</td><td>125</td><td>23</td></tr></table>

Note: In the paper [2], the MIT dataset includes 124 batteries. In fact, we found that a total of 135 batteries were included in the 3 MATLAB files, because some of the data were not included in the paper [2] for some errors. In our study, we used a total of 125 batteries by removing some battery data that encountered errors during feature extraction. The battery ID is the number of the data in the MATLAB file. More details can be found in our code: code link.

Table S.3: The estimation errors of the proposed PINN (ours), multi-layer perceptron (MLP) and convolutional neural network (CNN) on the XJTU battery dataset. MAPE is the mean absolute percentage error and RMSE is the root mean square error. All values are averages from 10 experiments. 

<table><tr><td rowspan="2">Batch</td><td rowspan="2">Battery</td><td colspan="2">Ours</td><td colspan="2">MLP</td><td colspan="2">CNN</td></tr><tr><td>MAPE</td><td>RMSE</td><td>MAPE</td><td>RMSE</td><td>MAPE</td><td>RMSE</td></tr><tr><td rowspan="2">1</td><td>battery-4</td><td>0.0071</td><td>0.0105</td><td>0.0276</td><td>0.0295</td><td>0.0187</td><td>0.0240</td></tr><tr><td>battery-8</td><td>0.0070</td><td>0.0084</td><td>0.0244</td><td>0.0259</td><td>0.0352</td><td>0.0419</td></tr><tr><td rowspan="3">2</td><td>battery-4</td><td>0.0145</td><td>0.0156</td><td>0.0310</td><td>0.0354</td><td>0.0359</td><td>0.0410</td></tr><tr><td>battery-8</td><td>0.0106</td><td>0.0112</td><td>0.0331</td><td>0.0348</td><td>0.0243</td><td>0.0298</td></tr><tr><td>battery-14</td><td>0.0087</td><td>0.0098</td><td>0.0185</td><td>0.0209</td><td>0.0290</td><td>0.0346</td></tr><tr><td rowspan="2">3</td><td>battery-4</td><td>0.0102</td><td>0.0120</td><td>0.0190</td><td>0.0222</td><td>0.0197</td><td>0.0232</td></tr><tr><td>battery-8</td><td>0.0070</td><td>0.0080</td><td>0.0232</td><td>0.0251</td><td>0.0157</td><td>0.0192</td></tr><tr><td rowspan="2">4</td><td>battery-4</td><td>0.0063</td><td>0.0099</td><td>0.0185</td><td>0.0226</td><td>0.0140</td><td>0.0175</td></tr><tr><td>battery-8</td><td>0.0080</td><td>0.0112</td><td>0.0215</td><td>0.0244</td><td>0.0160</td><td>0.0203</td></tr><tr><td rowspan="2">5</td><td>battery-4</td><td>0.0094</td><td>0.0113</td><td>0.0154</td><td>0.0174</td><td>0.0371</td><td>0.0447</td></tr><tr><td>battery-8</td><td>0.0116</td><td>0.0157</td><td>0.0212</td><td>0.0260</td><td>0.0328</td><td>0.0460</td></tr><tr><td rowspan="2">6</td><td>battery-4</td><td>0.0081</td><td>0.0130</td><td>0.0225</td><td>0.0268</td><td>0.0150</td><td>0.0200</td></tr><tr><td>battery-8</td><td>0.0046</td><td>0.0063</td><td>0.0184</td><td>0.0215</td><td>0.0148</td><td>0.0188</td></tr></table>

Table S.4: The estimation errors of the proposed PINN (ours), multi-layer perceptron (MLP) and convolutional neural network (CNN) on the TJU battery dataset. MAPE is the mean absolute percentage error and RMSE is the root mean square error. All values are averages from 10 experiments. 

<table><tr><td rowspan="2">Batch</td><td rowspan="2">Battery</td><td colspan="2">Ours</td><td colspan="2">MLP</td><td colspan="2">CNN</td></tr><tr><td>MAPE</td><td>RMSE</td><td>MAPE</td><td>RMSE</td><td>MAPE</td><td>RMSE</td></tr><tr><td rowspan="18">1</td><td>CY25-025_1-#5</td><td>0.0197</td><td>0.0180</td><td>0.0227</td><td>0.0211</td><td>0.0177</td><td>0.0178</td></tr><tr><td>CY25-05_1-#10</td><td>0.0119</td><td>0.0124</td><td>0.0167</td><td>0.0165</td><td>0.0202</td><td>0.0207</td></tr><tr><td>CY25-05_1-#16</td><td>0.0099</td><td>0.0101</td><td>0.0113</td><td>0.0116</td><td>0.0149</td><td>0.0159</td></tr><tr><td>CY25-05_1-#2</td><td>0.0100</td><td>0.0098</td><td>0.0123</td><td>0.0122</td><td>0.0161</td><td>0.0172</td></tr><tr><td>CY25-05_1-#8</td><td>0.0470</td><td>0.0436</td><td>0.0548</td><td>0.0489</td><td>0.0294</td><td>0.0325</td></tr><tr><td>CY25-1_1-#3</td><td>0.0146</td><td>0.0154</td><td>0.0188</td><td>0.0202</td><td>0.0374</td><td>0.0395</td></tr><tr><td>CY25-1_1-#9</td><td>0.0123</td><td>0.0139</td><td>0.0155</td><td>0.0174</td><td>0.0161</td><td>0.0183</td></tr><tr><td>CY45-05_1-#1</td><td>0.0309</td><td>0.0264</td><td>0.0362</td><td>0.0310</td><td>0.0245</td><td>0.0234</td></tr><tr><td>CY45-05_1-#15</td><td>0.0113</td><td>0.0111</td><td>0.0177</td><td>0.0170</td><td>0.0151</td><td>0.0158</td></tr><tr><td>CY45-05_1-#19</td><td>0.0098</td><td>0.0093</td><td>0.0144</td><td>0.0139</td><td>0.0128</td><td>0.0125</td></tr><tr><td>CY45-05_1-#24</td><td>0.0086</td><td>0.0086</td><td>0.0143</td><td>0.0139</td><td>0.0114</td><td>0.0118</td></tr><tr><td>CY45-05_1-#28</td><td>0.0080</td><td>0.0080</td><td>0.0106</td><td>0.0109</td><td>0.0086</td><td>0.0089</td></tr><tr><td>CY45-05_1-#8</td><td>0.0190</td><td>0.0185</td><td>0.0223</td><td>0.0213</td><td>0.0335</td><td>0.0358</td></tr><tr><td>CY25-05_1-#12</td><td>0.0234</td><td>0.0261</td><td>0.0236</td><td>0.0267</td><td>0.0179</td><td>0.0201</td></tr><tr><td>CY25-05_1-#16</td><td>0.0133</td><td>0.0164</td><td>0.0144</td><td>0.0166</td><td>0.0143</td><td>0.0158</td></tr><tr><td>CY25-05_1-#21</td><td>0.0198</td><td>0.0232</td><td>0.0271</td><td>0.0291</td><td>0.0281</td><td>0.0286</td></tr><tr><td>CY25-05_1-#4</td><td>0.0190</td><td>0.0230</td><td>0.0231</td><td>0.0264</td><td>0.0232</td><td>0.0236</td></tr><tr><td>CY35-05_1-#1</td><td>0.0119</td><td>0.0108</td><td>0.0135</td><td>0.0125</td><td>0.0105</td><td>0.0106</td></tr><tr><td rowspan="6">2</td><td>CY45-05_1-#1</td><td>0.0089</td><td>0.0090</td><td>0.0125</td><td>0.0124</td><td>0.0132</td><td>0.0134</td></tr><tr><td>CY45-05_1-#15</td><td>0.0099</td><td>0.0110</td><td>0.0114</td><td>0.0118</td><td>0.0160</td><td>0.0163</td></tr><tr><td>CY45-05_1-#19</td><td>0.0043</td><td>0.0050</td><td>0.0065</td><td>0.0069</td><td>0.0085</td><td>0.0086</td></tr><tr><td>CY45-05_1-#24</td><td>0.0047</td><td>0.0045</td><td>0.0077</td><td>0.0072</td><td>0.0075</td><td>0.0074</td></tr><tr><td>CY45-05_1-#28</td><td>0.0042</td><td>0.0040</td><td>0.0094</td><td>0.0088</td><td>0.0066</td><td>0.0067</td></tr><tr><td>CY45-05_1-#8</td><td>0.0111</td><td>0.0116</td><td>0.0151</td><td>0.0150</td><td>0.0121</td><td>0.0123</td></tr><tr><td rowspan="2">3</td><td>CY25-05_2-#2</td><td>0.0069</td><td>0.0070</td><td>0.0133</td><td>0.0133</td><td>0.0107</td><td>0.0110</td></tr><tr><td>CY25-05_4-#3</td><td>0.0090</td><td>0.0088</td><td>0.0166</td><td>0.0156</td><td>0.0140</td><td>0.0141</td></tr></table>

Table S.5: The estimation errors of the proposed PINN (ours), multi-layer perceptron (MLP) and convolutional neural network (CNN) on the HUST battery dataset. MAPE is the mean absolute percentage error and RMSE is the root mean square error. All values are averages from 10 experiments. 

<table><tr><td rowspan="2">Battery</td><td colspan="2">Ours</td><td colspan="2">MLP</td><td colspan="2">CNN</td></tr><tr><td>MAPE</td><td>RMSE</td><td>MAPE</td><td>RMSE</td><td>MAPE</td><td>RMSE</td></tr><tr><td>1-4</td><td>0.0070</td><td>0.0082</td><td>0.0071</td><td>0.0088</td><td>0.0070</td><td>0.0086</td></tr><tr><td>1-8</td><td>0.0061</td><td>0.0065</td><td>0.0070</td><td>0.0076</td><td>0.0061</td><td>0.0067</td></tr><tr><td>2-4</td><td>0.0078</td><td>0.0083</td><td>0.0073</td><td>0.0080</td><td>0.0073</td><td>0.0082</td></tr><tr><td>2-8</td><td>0.0053</td><td>0.0062</td><td>0.0049</td><td>0.0059</td><td>0.0073</td><td>0.0088</td></tr><tr><td>3-4</td><td>0.0080</td><td>0.0096</td><td>0.0090</td><td>0.0104</td><td>0.0104</td><td>0.0121</td></tr><tr><td>3-8</td><td>0.0103</td><td>0.0110</td><td>0.0112</td><td>0.0124</td><td>0.0098</td><td>0.0107</td></tr><tr><td>4-4</td><td>0.0084</td><td>0.0091</td><td>0.0079</td><td>0.0088</td><td>0.0073</td><td>0.0083</td></tr><tr><td>4-8</td><td>0.0090</td><td>0.0103</td><td>0.0090</td><td>0.0103</td><td>0.0077</td><td>0.0090</td></tr><tr><td>5-4</td><td>0.0056</td><td>0.0068</td><td>0.0063</td><td>0.0075</td><td>0.0050</td><td>0.0061</td></tr><tr><td>5-7</td><td>0.0053</td><td>0.0064</td><td>0.0051</td><td>0.0063</td><td>0.0058</td><td>0.0068</td></tr><tr><td>6-4</td><td>0.0119</td><td>0.0124</td><td>0.0115</td><td>0.0121</td><td>0.0098</td><td>0.0116</td></tr><tr><td>6-8</td><td>0.0123</td><td>0.0132</td><td>0.0125</td><td>0.0137</td><td>0.0112</td><td>0.0145</td></tr><tr><td>7-4</td><td>0.0058</td><td>0.0068</td><td>0.0067</td><td>0.0079</td><td>0.0063</td><td>0.0072</td></tr><tr><td>7-8</td><td>0.0069</td><td>0.0075</td><td>0.0077</td><td>0.0082</td><td>0.0063</td><td>0.0074</td></tr><tr><td>8-4</td><td>0.0056</td><td>0.0068</td><td>0.0053</td><td>0.0064</td><td>0.0069</td><td>0.0083</td></tr><tr><td>8-8</td><td>0.0099</td><td>0.0106</td><td>0.0110</td><td>0.0117</td><td>0.0074</td><td>0.0082</td></tr><tr><td>9-4</td><td>0.0145</td><td>0.0149</td><td>0.0145</td><td>0.0150</td><td>0.0102</td><td>0.0114</td></tr><tr><td>9-8</td><td>0.0036</td><td>0.0042</td><td>0.0033</td><td>0.0040</td><td>0.0046</td><td>0.0055</td></tr><tr><td>10-4</td><td>0.0090</td><td>0.0105</td><td>0.0084</td><td>0.0097</td><td>0.0068</td><td>0.0085</td></tr><tr><td>10-8</td><td>0.0035</td><td>0.0042</td><td>0.0042</td><td>0.0051</td><td>0.0047</td><td>0.0056</td></tr></table>

Table S.6: The estimation errors of the proposed PINN (ours), multi-layer perceptron (MLP) and convolutional neural network (CNN) on the MIT battery dataset. MAPE is the mean absolute percentage error and RMSE is the root mean square error. All values are averages from 10 experiments. 

<table><tr><td rowspan="2">Date</td><td rowspan="2">Battery</td><td colspan="2">Ours</td><td colspan="2">MLP</td><td colspan="2">CNN</td></tr><tr><td>MAPE</td><td>RMSE</td><td>MAPE</td><td>RMSE</td><td>MAPE</td><td>RMSE</td></tr><tr><td rowspan="10">2017/5/12</td><td>battery-5</td><td>0.0091</td><td>0.0109</td><td>0.0044</td><td>0.0050</td><td>0.0069</td><td>0.0077</td></tr><tr><td>battery-10</td><td>0.0032</td><td>0.0037</td><td>0.0059</td><td>0.0070</td><td>0.0057</td><td>0.0062</td></tr><tr><td>battery-15</td><td>0.0069</td><td>0.0073</td><td>0.0053</td><td>0.0057</td><td>0.0070</td><td>0.0077</td></tr><tr><td>battery-20</td><td>0.0036</td><td>0.0040</td><td>0.0068</td><td>0.0075</td><td>0.0065</td><td>0.0070</td></tr><tr><td>battery-25</td><td>0.0073</td><td>0.0076</td><td>0.0038</td><td>0.0047</td><td>0.0044</td><td>0.0053</td></tr><tr><td>battery-30</td><td>0.0038</td><td>0.0046</td><td>0.0028</td><td>0.0033</td><td>0.0036</td><td>0.0042</td></tr><tr><td>battery-35</td><td>0.0034</td><td>0.0037</td><td>0.0062</td><td>0.0065</td><td>0.0057</td><td>0.0062</td></tr><tr><td>battery-40</td><td>0.0043</td><td>0.0046</td><td>0.0090</td><td>0.0098</td><td>0.0059</td><td>0.0068</td></tr><tr><td>battery-45</td><td>0.0062</td><td>0.0073</td><td>0.0099</td><td>0.0124</td><td>0.0096</td><td>0.0129</td></tr><tr><td>battery-5</td><td>0.0104</td><td>0.0103</td><td>0.0023</td><td>0.0027</td><td>0.0039</td><td>0.0045</td></tr><tr><td rowspan="8">2017/6/30</td><td>battery-15</td><td>0.0022</td><td>0.0026</td><td>0.0243</td><td>0.0240</td><td>0.0163</td><td>0.0173</td></tr><tr><td>battery-20</td><td>0.0183</td><td>0.0187</td><td>0.0126</td><td>0.0194</td><td>0.0086</td><td>0.0132</td></tr><tr><td>battery-25</td><td>0.0093</td><td>0.0155</td><td>0.0055</td><td>0.0059</td><td>0.0036</td><td>0.0042</td></tr><tr><td>battery-30</td><td>0.0056</td><td>0.0060</td><td>0.0080</td><td>0.0096</td><td>0.0078</td><td>0.0098</td></tr><tr><td>battery-35</td><td>0.0072</td><td>0.0093</td><td>0.0044</td><td>0.0051</td><td>0.0040</td><td>0.0048</td></tr><tr><td>battery-40</td><td>0.0041</td><td>0.0049</td><td>0.0073</td><td>0.0080</td><td>0.0041</td><td>0.0048</td></tr><tr><td>battery-45</td><td>0.0059</td><td>0.0067</td><td>0.0139</td><td>0.0133</td><td>0.0055</td><td>0.0060</td></tr><tr><td>battery-10</td><td>0.0040</td><td>0.0053</td><td>0.0055</td><td>0.0061</td><td>0.0049</td><td>0.0059</td></tr><tr><td rowspan="5">2018/4/12</td><td>battery-15</td><td>0.0069</td><td>0.0070</td><td>0.0084</td><td>0.0084</td><td>0.0057</td><td>0.0062</td></tr><tr><td>battery-25</td><td>0.0106</td><td>0.0106</td><td>0.0099</td><td>0.0100</td><td>0.0121</td><td>0.0123</td></tr><tr><td>battery-30</td><td>0.0021</td><td>0.0024</td><td>0.0033</td><td>0.0037</td><td>0.0026</td><td>0.0031</td></tr><tr><td>battery-40</td><td>0.0124</td><td>0.0122</td><td>0.0133</td><td>0.0130</td><td>0.0109</td><td>0.0109</td></tr><tr><td>battery-45</td><td>0.0067</td><td>0.0075</td><td>0.0083</td><td>0.0086</td><td>0.0053</td><td>0.0064</td></tr></table>

Table S.7: The results of 3 models (the proposed PINN (Ours), multilayer perceptron (MLP), and convolutional neural network (CNN)) in small sample experiments on XJTU battery dataset. MAPE is mean absolute percentage error and RMSE is root mean square error. All values are averaged from 10 experiments. 

<table><tr><td rowspan="2">Batch</td><td rowspan="2">Train Batteries</td><td colspan="2">Ours</td><td colspan="2">MLP</td><td colspan="2">CNN</td></tr><tr><td>MAPE</td><td>RMSE</td><td>MAPE</td><td>RMSE</td><td>MAPE</td><td>RMSE</td></tr><tr><td rowspan="4">1</td><td>1</td><td>0.0141</td><td>0.0184</td><td>0.0343</td><td>0.0390</td><td>0.0929</td><td>0.0949</td></tr><tr><td>2</td><td>0.0105</td><td>0.0134</td><td>0.0267</td><td>0.0304</td><td>0.0728</td><td>0.0826</td></tr><tr><td>3</td><td>0.0069</td><td>0.0096</td><td>0.0347</td><td>0.0383</td><td>0.0548</td><td>0.0666</td></tr><tr><td>4</td><td>0.0056</td><td>0.0076</td><td>0.0292</td><td>0.0327</td><td>0.0560</td><td>0.0647</td></tr><tr><td rowspan="4">2</td><td>1</td><td>0.0954</td><td>0.0516</td><td>0.5480</td><td>0.1204</td><td>3.3605</td><td>0.2915</td></tr><tr><td>2</td><td>0.0197</td><td>0.0262</td><td>0.0648</td><td>0.0629</td><td>0.2045</td><td>0.1469</td></tr><tr><td>3</td><td>0.0106</td><td>0.0119</td><td>0.0264</td><td>0.0308</td><td>0.1066</td><td>0.1165</td></tr><tr><td>4</td><td>0.0115</td><td>0.0130</td><td>0.0284</td><td>0.0329</td><td>0.0744</td><td>0.0862</td></tr><tr><td rowspan="4">3</td><td>1</td><td>0.0240</td><td>0.0261</td><td>0.0336</td><td>0.0379</td><td>0.1579</td><td>0.1520</td></tr><tr><td>2</td><td>0.0096</td><td>0.0112</td><td>0.0270</td><td>0.0312</td><td>0.0551</td><td>0.0659</td></tr><tr><td>3</td><td>0.0090</td><td>0.0105</td><td>0.0273</td><td>0.0304</td><td>0.0374</td><td>0.0450</td></tr><tr><td>4</td><td>0.0088</td><td>0.0102</td><td>0.0291</td><td>0.0324</td><td>0.0277</td><td>0.0340</td></tr><tr><td rowspan="4">4</td><td>1</td><td>0.0168</td><td>0.0199</td><td>0.0261</td><td>0.0301</td><td>0.1075</td><td>0.1130</td></tr><tr><td>2</td><td>0.0075</td><td>0.0103</td><td>0.0218</td><td>0.0251</td><td>0.0623</td><td>0.0726</td></tr><tr><td>3</td><td>0.0076</td><td>0.0106</td><td>0.0222</td><td>0.0251</td><td>0.0344</td><td>0.0427</td></tr><tr><td>4</td><td>0.0070</td><td>0.0097</td><td>0.0205</td><td>0.0230</td><td>0.0238</td><td>0.0299</td></tr><tr><td rowspan="4">5</td><td>1</td><td>0.2519</td><td>0.1479</td><td>0.6330</td><td>0.1553</td><td>0.4337</td><td>0.2315</td></tr><tr><td>2</td><td>0.0141</td><td>0.0184</td><td>0.0306</td><td>0.0392</td><td>0.1193</td><td>0.1210</td></tr><tr><td>3</td><td>0.0157</td><td>0.0189</td><td>0.0303</td><td>0.0390</td><td>0.1084</td><td>0.1136</td></tr><tr><td>4</td><td>0.0132</td><td>0.0168</td><td>0.0264</td><td>0.0349</td><td>0.1133</td><td>0.0971</td></tr><tr><td rowspan="4">6</td><td>1</td><td>0.0103</td><td>0.0135</td><td>0.0282</td><td>0.0351</td><td>0.2007</td><td>0.1577</td></tr><tr><td>2</td><td>0.0096</td><td>0.0120</td><td>0.0196</td><td>0.0231</td><td>0.0499</td><td>0.0626</td></tr><tr><td>3</td><td>0.0061</td><td>0.0088</td><td>0.0218</td><td>0.0264</td><td>0.0341</td><td>0.0437</td></tr><tr><td>4</td><td>0.0072</td><td>0.0107</td><td>0.0205</td><td>0.0241</td><td>0.0247</td><td>0.0319</td></tr></table>

Figure S.5: An illustration of test root mean square error (RMSE) distributions for 3 models (the proposed PINN (Ours), multi-layer perceptron (MLP), and convolutional neural network (CNN)) on XJTU battery dataset. Each error bar contains 10 points (10 experiment) and is marked with mean and standard deviation lines. The legends have been added only on the last subplot. The “1 battery” in the legend means that we only use the data of 1 battery to train the model. Others are similar. As the number of batteries increases, the performance of the 3 models is getting better. However, our method still performs best among them.

# Supplementary Note 3. Physics-informed neural network

Table S.8: The details of proposed PINN, multi-layer perceptron (MLP), and multi-layer perceptron (CNN). Sin refers to the sine function. BasicBlack is similar to that in ResNet [3], which consists of Conv1d, BatchNorm1d, ReLU, Conv1d, and BatchNorm1d. 

<table><tr><td>Model</td><td>Module</td><td>Layer</td><td>Input size</td><td>Output size</td><td>Inference Param.</td><td>Num.</td><td>Inference time/1000 sample</td></tr><tr><td rowspan="8">PINN</td><td rowspan="5"> $\mathcal{F}(\cdot)$ </td><td>Linear+Sin</td><td>17</td><td>60</td><td></td><td></td><td></td></tr><tr><td>Linear+Sin</td><td>60</td><td>60</td><td></td><td></td><td></td></tr><tr><td>Linear</td><td>60</td><td>32</td><td>7781</td><td></td><td>5.81e-04</td></tr><tr><td>Linear+Sin</td><td>32</td><td>32</td><td></td><td></td><td></td></tr><tr><td>Linear</td><td>32</td><td>1</td><td></td><td></td><td></td></tr><tr><td rowspan="3"> $\mathcal{G}(\cdot)$ </td><td>Linear+Sin</td><td>35</td><td>60</td><td></td><td></td><td></td></tr><tr><td>Linear+Sin</td><td>60</td><td>60</td><td></td><td></td><td></td></tr><tr><td>Linear</td><td>60</td><td>1</td><td></td><td></td><td></td></tr><tr><td rowspan="5">MLP</td><td rowspan="5"></td><td>Linear+Sin</td><td>17</td><td>60</td><td></td><td></td><td></td></tr><tr><td>Linear+Sin</td><td>60</td><td>60</td><td></td><td></td><td></td></tr><tr><td>Linear</td><td>60</td><td>32</td><td>7781</td><td></td><td>5.64e-04</td></tr><tr><td>Linear+Sin</td><td>32</td><td>32</td><td></td><td></td><td></td></tr><tr><td>Linear</td><td>32</td><td>1</td><td></td><td></td><td></td></tr><tr><td rowspan="6">CNN</td><td rowspan="6"></td><td>BasicBlock</td><td>(1,17)</td><td>(8,17)</td><td></td><td></td><td></td></tr><tr><td>BasicBlock</td><td>(8,17)</td><td>(16,9)</td><td></td><td></td><td></td></tr><tr><td>BasicBlock</td><td>(16,9)</td><td>(24,5)</td><td></td><td></td><td></td></tr><tr><td>BasicBlock</td><td>(24,5)</td><td>(16,5)</td><td>8465</td><td></td><td>1.29e-02</td></tr><tr><td>BasicBlock</td><td>(16,5)</td><td>(8,5)</td><td></td><td></td><td></td></tr><tr><td>Linear</td><td>8*5</td><td>1</td><td></td><td></td><td></td></tr></table>

Note: The values in the ”Inference time/1000 sample” column represents the the time, in seconds, spent in inference per 1000 samples. Specifically, we set the batch size to 1000, count the time spent on 1000 forward inferences, and then take the average. Since the number of parameters of all three model is small, we do not use GPU for acceleration. Three models were implemented in Pytorch 1.7.1 on Intel Core i5-10400F CPU @ 2.90 GHz.

The parameters and structure of the proposed PINN, MLP, and CNN are given in Table S.8. The proposed PINN was learned by minimizing the loss defined in Equation (9) of the Manuscript File, and the Adam optimizer was used in the training phase. In the process of hyperparameter tuning, the Grid Search strategy was adopted to optimize the hypterparameter, including the number of PINN layers, the number of neurons in each layer, and α and $\beta .$ The trade-off parameters α and $\beta$ are set to 0.7 and 20 for XJTU battery dataset, 1 and 50 for TJU and MIT datasets, and 0.5 and 80 for HUST dataset. We set batch size to 256 for XJTU battery dataset and 512 for other 3 datasets. More details can be found in our codes.

# References

[1] D. Zhao, Z. Zhou, S. Tang, Y. Cao, J. Wang, P. Zhang, Y. Zhang, Online estimation of satellite lithium-ion battery capacity based on approximate belief rule base and hidden markov model, Energy 256 (2022) 124632.   
[2] K. A. Severson, P. M. Attia, N. Jin, N. Perkins, B. Jiang, Z. Yang, M. H. Chen, M. Aykol, P. K. Herring, D. Fraggedakis, et al., Data-driven prediction of battery cycle life before capacity degradation, Nature Energy 4 (5) (2019) 383–391.   
[3] K. He, X. Zhang, S. Ren, J. Sun, Deep residual learning for image recognition, in: Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 770–778.
