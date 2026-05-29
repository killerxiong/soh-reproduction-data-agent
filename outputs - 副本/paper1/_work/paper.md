# Physics-informed neural network for lithiumion battery degradation stable modeling and prognosis

Received: 28 September 2023

Accepted: 7 May 2024

Published online: 21 May 2024

Check for updates

Fujin Wang 1,2,3, Zhi Zhai 1,2,3, Zhibin Zhao 1,2 , Yi Di1,2 & Xuefeng Chen1,2

Accurate state-of-health (SOH) estimation is critical for reliable and safe operation of lithium-ion batteries. However, reliable and stable battery SOH estimation remains challenging due to diverse battery types and operating conditions. In this paper, we propose a physics-informed neural network (PINN) for accurate and stable estimation of battery SOH. Specifically, we model the attributes that affect the battery degradation from the perspective of empirical degradation and state space equations, and utilize neural networks to capture battery degradation dynamics. A general feature extraction method is designed to extract statistical features from a short period of data before the battery is fully charged, enabling our method applicable to different battery types and charge/discharge protocols. Additionally, we generate a comprehensive dataset consisting of 55 lithium-nickel-cobalt-manganeseoxide (NCM) batteries. Combined with three other datasets from different manufacturers, we use a total of 387 batteries with 310,705 samples to validate our method. The mean absolute percentage error (MAPE) is 0.87%. Our proposed PINN has demonstrated remarkable performance in regular experiments, small sample experiments, and transfer experiments when compared to alternative neural networks. This study highlights the promise of physicsinformed machine learning for battery degradation modeling and SOH estimation.

In recent years, the number of lithium-ion batteries is growing at an alarming rate in the whole society, which is an unprecedented impetus to the popularization of renewable energy equipment. With the advantages of high energy density, low self-discharge rate, and long service life1 , lithium-ion batteries have become the main energy storage devices in portable electronic devices, electric vehicles, aerospace, and many other fields2–8 . In 2019, the global shipments of lithium-ion batteries for new energy vehicles alone reached 116.6 GWh9 . It is estimated that by 2025, the global lithium-ion batteries installed capacity will reach 800 GWh, and the market value will reach 91.8 billion dollars10. The explosive growth of lithium-ion batteries has brought convenience to people’s lives, however, its aging and health management have also attracted people’s concerns and attention. The aging of lithium-ion batteries is an important issue, and their performance will decline with time until it fails. To ensure long-term, safe, and continuous operation, lithium-ion batteries must be properly maintained and controlled, which includes state-of-health (SOH) assessments. The SOH of a battery is defined as the ratio of the current available capacity to the initial capacity, which can be used as an indicator to measure battery degradation11. When the SOH drops to 80%, the battery reaches its first service life. Batteries that have reached their first service life can still be used in fields such as energy

1 National and Local Joint Engineering Research Center of Equipment Operation Safety and Intelligent Monitoring, Xi’an Jiaotong University, Xi’an 710049 Shaanxi, PR China. 2 School of Mechanical Engineering, Xi’an Jiaotong University, Xi’an 710049 Shaanxi, PR China. 3 These authors contributed equally: Fujin Wang, Zhi Zhai. e-mail: zhaozhibin@xjtu.edu.cn; chenxf@mail.xjtu.edu.cn

storage power stations for secondary utilization. Therefore, it is particularly important to accurately estimate the SOH of the battery.

In recent years, various SOH estimation methods of lithium-ion batteries have been proposed, which greatly advance the development of this field12–15. However, accurately estimating SOH is still a challenging problem. Generally, the capacity can be obtained from a complete discharge curve from an upper cut-off voltage to a lower cut-off voltage via the ampere-hour integration, thereby obtaining SOH. In actual application, it is difficult to obtain a complete charge or discharge curve because the battery is rarely fully discharged. Some scholars estimate SOH by establishing battery aging models. Baghdadi et al.16 proposed a physics-based approach based on Dakin’s degradation method to simulate the linear degradation process of batteries. Considering that time-varying temperature conditions have an important impact on the discharge capacity and aging law of lithium-ion batteries, Xu et al.17 proposed a stochastic degradation rate model based on the Arrhenius temperature model and established an aging model of lithium-ion batteries under time-varying temperature conditions based on the Wiener process. Dong et al.18 proposed a physics-based model that combines chemical and mechanical degradation mechanisms to predict capacity fade by simulating the formation and growth of solid electrolyte interphase (SEI). Lui et al.19 proposed a physicsbased approach to predict the capacity of lithium-ion batteries by modeling degradation mechanisms such as losses of active materials of the positive and negative electrodes and the loss of lithium inventory.

Given the difficulty in establishing physical models and the difficulty in obtaining complete discharge capacity, many studies have used data-driven methods20–24 to estimate SOH based on current and voltage curves during charge and discharge. Commonly used datadriven methods include linear regression25, support vector machines26, Gaussian process regression27, deep neural networks28,29, etc. Xia et al.30 extracted features from incremental capacity (IC) curves and differential voltage (DV) curves to estimate SOH. Wang et al.31 extracted valuable health indicators from electrochemical impedance spectroscopy (EIS) as input for Gaussian process regression to estimate SOH. Data-driven methods do not require physical knowledge and only focus on the relationship between input and output, so the extraction of degenerated features is a key part of data-driven methods, which largely determines the performance of the SOH estimation.

However, challenges still stand in the way of developing reliable, accurate, and general SOH estimation methods14,21. Physics-based models are stable and accurate, but batteries with different chemical compositions require different model parameters, and the models have high computational costs32. The data-driven models have high accuracy and efficiency, but its generalizability depends on the extracted features and have poor stability14,33. For instance, due to the high usage variability, existing methods30,34,35 need to extract specific features for different datasets or different working conditions, leading to the fact that models are dataset-specific, resulting in a waste of computing resources. The promising prospect of physics-informed neural network (PINN)36,37 lies in amalgamating the strengths of physics-based and data-driven approaches, potentially addressing the aforementioned challenges. Due to the consideration of physical information, PINN can use relatively less data to train the model, and the model is more stable. It is a promising approach in the field of battery prognosis and diagnostics. Aykol et al.38 classified battery modeling methods that combine physical knowledge and machine learning into five categories, including three Sequential Integration methods, A1–A3, and two Hybrid methods, B1–B2. Among them, an obvious feature of the Sequential Integration method is that the physical model and the machine learning model are standalone, while the Hybrid method fuses the two together. Within this framework, some works has been published39–43. Nascimento et al.39 directly implemented the numerical integration of principle-based governing equations through recurrent neural networks to simulate the dynamic response of the battery. Wang et al.42 proposed a battery neural network (BattNN) for discharge voltage prediction based on the equivalent circuit model (ECM). Hofmann et al.43 used the pseudo-twodimensional (P2D) Newman model to generate data at different health status points and combined it with experimental data and field data to train the neural network model, which takes advantage of the correlation between internal states and measurable SOH. According to the categories proposed by Aykol et al.38, these methods belong to the A240,41,43 ${ \bf A } 2 ^ { 4 0 , \bar { 4 } 1 , 4 3 }$ and A339,42 categories.

In fact, the Sequential Integration method is relatively straightforward to implement because the physical model and the machine learning model are standalone, making it a practical near-term strategy for battery modeling. Essentially, machine learning models in Sequential Integration method are not subject to physical constraints. The Hybrid methods, on the other hand, are more fundamental as they truly integrate the primary governing equations for battery modeling with data-driven methods. However, due to the complex physical equations contain numerous parameters and are difficult to solve, there are few publications that implement Hybrid methods for SOH estimation. Recent review38 pointed out that Hybrid methods will become the dominant method in the long term, but it is still an open research question.

In this work, we proposed a PINN for battery SOH estimation, which belong to the B2 architecture. This approach achieves true integration of governing equations and neural networks, resulting in stable and precise SOH estimation. Unlike existing PINN approaches, we also validated its advancements in small sample learning and transfer learning among batteries with different chemistries and charge/discharge profiles. Specifically, first, considering the complexity of the electrochemical equations, it hinders the development of B2-type PINNs. Therefore, we model battery degradation dynamics from the perspective of empirical degradation and state space equations, and utilize neural networks to capture battery degradation dynamics. Second, to make the model more general, we develop a new feature extraction method. The discharge process of a battery is userspecific, and the battery is rarely fully discharged. In contrast, once charging starts, the probability of full charge is high, and it is more fixed and regular. Therefore, we extract features from a short period of data before the battery is fully charged. Third, to verify our method, we carried out battery degradation experiments and released a dataset containing run-to-failure data from 55 batteries. In addition, we also verified our method on other three large-scale datasets with different chemical compositions and charge/discharge protocols, proving the superiority and versatility of our method. We also perform the further task of estimating SOH by transferring degradation knowledge from one dataset to another. These datasets for performing transfer tasks contain batteries with different chemistries and charge/discharge protocols. The results illustrate the effectiveness and generality of the proposed PINN in SOH estimation.

# Results

# Framework overview and flowchart

We developed a PINN for lithium-ion battery SOH estimation, and its flowchart is shown in Fig. 1. Our method is designed for more general, reliable, stable, and high-precision SOH estimation by considering the dynamic behavior of battery degradation as well as the degradation trend.

In the data preprocessing stage (shown in Fig. 1b), statistical features are extracted from a short period of data before the battery is fully charged as the input of the model, which ensures that this period of data exists in most battery datasets, and solves the problem of nonuniversal features in existing studies. Therefore, our method is applicable to batteries with different chemistries and charge/discharge protocols.

In the SOH estimation stage, due to the complexity of electrochemical equations, there is currently no good way to integrate them with the neural networks. In this work, we modeled the attributes that affect the battery degradation from the perspective of the empirical degradation and state space equation, and utilized neural networks to approximate the established degradation model, effectively achieving the integration of governing equations and neural networks. The proposed PINN consists of two parts: a solution function ( ⋅ ) that maps ffeatures to SOH and a nonlinear function ( ⋅ ) that models battery gdegradation dynamic behaviors, as shown in Fig. 1c. The solution ( ⋅ ), fmodeling the relationship between features and SOH, is expressed as i = ( i , xi ), where i represents time, xi represents the extracted feature

Fig. 1 | The flowchart of the proposed PINN for lithium-ion battery SOH estimation. a The lithium-ion batteries may have different chemistries (e.g., lithium nickel-cobalt-manganate (NCM), lithium nickel-cobalt-aluminate (NCA), and lithium iron phosphate (LFP), etc.). Different users have personalized battery discharge strategies resulting in different degradation trajectories. b An illustration of the selected data for feature extraction. We extracted features from a short period of data before the battery is fully charged. These features are used as the inputs of the proposed PINN to estimate SOH. The upper figures are the curves from the 10th cycle, and lower figures are all the curves during the entire life cycle. Aging of the battery and changes in charge/discharge protocols cause the curves to shift. c The structure of the proposed PINN, where  and ^ represent the true and estimated SOH,  and x represent cycle and features, the superscript  represents sample t iindex, and the subscripts  and x represent the corresponding partial derivatives. The functions ( ⋅ ) and $g ( \cdot )$ respectively model the mapping between features to SOH and the degradation dynamics of the battery, and $\mathcal { F } ( \cdot )$ and represent the neural networks used to approximate ( ⋅ ) and ( ⋅ ) (see section “Methods” for more details).

Table 1 | The chemical components and basic experiment conditions for four datasets 

<table><tr><td>Dataset</td><td>Batch</td><td>Chemical component</td><td>Nominal capacity (mAh)</td><td>Cut-off voltage (V)</td><td>Experiment temperature (°C)</td><td>Number of cells</td></tr><tr><td>XJTU</td><td>1,2,3,4,5,6</td><td> $LiNi_{0.5}Co_{0.2}Mn_{0.3}O_2$ </td><td>2000</td><td>2.5–4.2</td><td>Room temperature</td><td>55</td></tr><tr><td>TJU</td><td>1</td><td> $Li_{0.86}Ni_{0.86}Co_{0.11}Al_{0.03}O_2$ </td><td>3500</td><td>2.65–4.2</td><td>25,35,45</td><td>66</td></tr><tr><td></td><td>2</td><td> $Li_{0.84}Ni_{0.83}Co_{0.11}Mn_{0.07}O_2$ </td><td>3500</td><td>2.5–4.2</td><td>25,35,45</td><td>55</td></tr><tr><td></td><td>3</td><td>Blend of 42 (3) wt.%  $LiNiCoMnO_2$  and 58 (3) wt.%  $LiNiCoAlO_2$ </td><td>2500</td><td>2.5–4.2</td><td>25</td><td>9</td></tr><tr><td>MIT</td><td>-</td><td> $LiFePO_4$ </td><td>1100</td><td>2.0–3.6</td><td>30</td><td>125</td></tr><tr><td>HUST</td><td>-</td><td> $LiFePO_4$ </td><td>1100</td><td>2.0–3.6</td><td>30</td><td>77</td></tr></table>

The charge/discharge protocol varies among different datasets.

vector, and i denotes the SOH of the cycle . The nonlinear function $g ( \cdot )$ u i models the SOH decay rate of the battery. Since $f ( \cdot )$ and $g ( \cdot )$ are g f gaffected by many factors in reality and their explicit expressions are unknown, they are replaced by small fully connected neural networks, denote as $\mathcal F ( \cdot )$ and $\mathcal { G } ( \cdot )$ . During training, we consider data term loss $\mathcal { L } _ { \mathrm { d a t a } } ,$ , monotonicity loss ${ \mathcal { L } } _ { { \mathrm { m o n o } } } ,$ and loss $\mathcal { L } _ { \mathrm { P D E } }$ constrained by the degradation equation described by partial differential equation. They minimize the errors between the predicted and the true values, while making the model follow the properties of monotonicity of the degradation trajectory and satisfy the constraints of the established degradation model.

To validate the superiority of the proposed PINN, we conducted small sample experiments and transfer experiments. During the transfer experiments, we froze  and fine-tuned  on datasets with different chemical compositions. The experimental outcomes demonstrate that the proposed PINN framework can effectively capture the dynamics of battery degradation. Our study combines knowledge of the battery degradation with neural networks and achieves promising results. This study highlights the promise of physics-informed neural network for battery degradation modeling and SOH estimation (more methodological details can be found in the “Methods” section).

# Data generation

To cover different battery types and chemistries, we employ 310,705 samples of 387 batteries from 4 different large-scale datasets for validation. The first dataset comes from the battery degradation experiments we conducted for this study, the other three datasets are well-known public datasets from Zhu et al.44, Ye et al.45, and Severson et al.25. For convenience, we refer to the four datasets as the XJTU battery dataset, TJU dataset44, HUST dataset45, and MIT dataset25. The basic information of the four datasets is given in Table 1.

We developed a battery degradation experiment in this study, as shown in Fig. S1. A total of 55 batteries manufactured by LISHEN $\begin{array} { r } { ( \mathsf { L i N i } _ { 0 . 5 } \mathsf { C o } _ { 0 . 2 } \mathsf { M n } _ { 0 . 3 } \mathsf { O } _ { 2 } , } \end{array}$ , 2000 mAh nominal capacity, and 3.6 V nominal voltage, the cut-off voltages of charging and discharging are 4.2 V and 2.5 V) were cycled to failure under 6 charge/discharge protocols at the room temperature. The protocols include fixed charging and discharging, random discharging with a fixed current in different cycles, random walking, and the charging and discharging strategy of a satellite in geosynchronous earth orbit (GEO). We use batch 1 to batch 6 to represent the 6 charge/discharge protocols, respectively. The degradation trajectories are shown in Fig. 2. More details about our dataset can be found in Supplementary Note 1.

Fig. 2 | The degradation trajectories of the XJTU battery dataset. There are 6 batches (55 batteries) in total, all batches contain 8 batteries except batch 2 which contains 15 batteries. The charge/discharge protocols are different among batches. See Supplementary Note 1 for more details.

The TJU dataset contains three types of battery: NCA battery (3500 mAh nominal capacity and 2.65–4.2 V cut-off voltage), NCM battery (3500 mAh nominal capacity and 2.5–4.2 V cut-off voltages), and NCM + NCA battery (2500 mAh nominal capacity and 2.5–4.2 V cut-off voltage). These batteries are cycled in a temperature-controlled chamber with different temperatures and different charge current rates. Candidate sets for temperatures include 25, 35, and $4 5 ^ { \circ } \mathrm { C }$ . Current rates ranging from 0.25 C to 4 C were used. We use batch 1, batch 2, and batch 3 to represent NCA, NCM, and NCM + NCA batteries, respectively.

The HUST dataset contains data from 77 LFP/graphite cells under 77 different multi-stage discharge protocols. These batteries, manufactured by A123 (APR18650M1A), have a nominal capacity of 1100 mAh and a nominal voltage of 3.3 V. They were cycled at a temperature of 30 °C with an identical charge protocol but different discharge protocols until failure.

The batteries in the MIT dataset have the same type and chemical composition as the batteries in the HUST dataset. However, unlike the experimental setup at the HUST dataset, the MIT dataset considered multiple fast-charging strategies and one discharging strategy.

(a)   

Fig. 3 | An illustration of extracted features and correlation coefficients.   
a Features of 8 batteries from the XJTU dataset batch 1. The -axis of each subfigure xis SOH, and the -axis is the normalized value of the corresponding feature. The ynumber on the right side of each subfigure represents the feature number.

b Correlation heatmap between extracted features and SOH in four datasets. The numbers 1–16 represent 16 features, and the order of features is consistent with that in (a).

Specifically, they were cycled under a fast-charging experiment with a one-step or two-step fast-charging policy, and discharged at 4 C. The experiment temperature is $3 0 ^ { \circ } \mathrm { C }$ .

# Feature extraction

Robust features can often improve the performance of SOH estimation. However, how to extract general and robust features is a worthy research problem. In existing studies, various feature extraction methods for specific datasets and charge/discharge protocols were proposed, yet the generalization of features has been insufficiently considered. There are few studies on methods for extracting general features for different battery types or charge/discharge protocols. To extract more robust and generalizable features, we propose a method to extract features from a short period of charging voltage curve and current curve through observation and exploration of multiple datasets. It is an undoubted fact that the discharging process of the battery is user-specific. In contrast, the charging process is essential and more fixed and regular, and the probability of the battery being fully charged is relatively high. We found that most datasets contain constant current and constant voltage (CC-CV) charging modes. For the four public datasets we used, no matter what strategy the battery is discharged with or whether it is fully discharged, it will eventually be fully charged when charging.

Therefore, we selected a short period of data before the battery was fully charged to extract features, as shown in Fig. 1b. Define the charge cut-off voltage of a battery as $V _ { \mathrm { e n d } } ,$ and the voltage data whose value is within $[ V _ { \mathrm { e n d - } 0 . 2 } , V _ { \mathrm { e n d } } ]$ V V is selected. For the current data, we V Vchoose the data with the current between 0.5 A and 0.1 A during the constant voltage charging. Regardless of whether the battery is fully discharged, as long as the battery is fully charged, the voltage range and current range always exist.

The mean, standard deviation, kurtosis, skewness, charging time, accumulated charge, curve slope, and curve entropy from the selected current and voltage curves, respectively (these features are numbered

1–16, respectively. See Supplementary Note 2 for more details) are extracted. An illustration of extracted features from XJTU dataset batch 1 is given in Fig. 3a. Further, we extracted features from 387 batteries in 4 datasets respectively, and calculated the Pearson correlation coefficient between features and SOH within each dataset, as shown in Fig. 3b.

Based on experimental phenomena and analysis of Fig. 3b, we give a natural conjecture: the magnitude of the correlation coefficient between each feature and SOH is related to the chemical composition of a battery and is less affected by the charge/discharge protocols. To the best knowledge, we are the first to focus on this phenomenon. It can be seen from Table 1 that both the XJTU dataset and the TJU dataset are LiNiCo-x type batteries. Even though they have completely different nominal capacities and charge/discharge protocols, the features extracted from our selected range are highly similar. For example, there is a very strong negative correlation between features 11–16 and SOH. Features 9 and 10 have a strong positive correlation with SOH. In contrast, the MIT dataset and the HUST dataset are both LiFePO batteries. Features 11–16 show a weak positive correlation with SOH, while features 9 and 10 show a negative correlation with SOH. Besides, features 3–6 and 8 of the latter two datasets show a strong positive correlation with SOH.

# SOH estimation

The extracted 16 features and time (cycle) are used as inputs of the proposed PINN to estimate SOH. To reduce the impact of the difference in feature magnitude on the model and make the model training more stable, the min–max normalization is performed on the features. That is, all features are scaled to the range [−1,1]. The SOH estimation results of the proposed PINN on 4 datasets are given in Fig. 4a (the number of test batteries in each dataset can be found in Table S2.

To demonstrate the advancement of the proposed PINN, Multi-Layer Perceptron (MLP) with the same structure and parameter amounts and Convolutional Neural Network (CNN) with similar

(b)   

Fig. 4 | The illustrations of SOH estimation results. a The SOH estimation results of proposed PINN on four datasets. The predicted and true SOH are distributed near the diagonal, indicating that the model performs well. b Distributions of mean absolute error (MAE), mean absolute percentage error (MAPE), and root mean square error (RMSE) of 3 models (the proposed PINN (Ours), multi-layer perceptron   
(MLP), and convolutional neural network (CNN)) on four datasets. Each error bar contains 10 points (10 experiment) and is marked with mean and standard deviation lines. Compared with the other two methods, our method has smaller prediction errors and is more stable. Source data are provided as a Source Data file.

Table 2 | The results of proposed PINN (Ours), multi-layer perceptron (MLP), and convolutional neural network (CNN) on four datasets 

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Batch</td><td colspan="2">Ours</td><td colspan="2">MLP</td><td colspan="2">CNN</td></tr><tr><td>MAPE</td><td>RMSE</td><td>MAPE</td><td>RMSE</td><td>MAPE</td><td>RMSE</td></tr><tr><td>XJTU</td><td>1</td><td>0.0070</td><td>0.0094</td><td>0.0260</td><td>0.0277</td><td>0.0270</td><td>0.0330</td></tr><tr><td></td><td>2</td><td>0.0113</td><td>0.0122</td><td>0.0275</td><td>0.0304</td><td>0.0298</td><td>0.0352</td></tr><tr><td></td><td>3</td><td>0.0086</td><td>0.0100</td><td>0.0211</td><td>0.0237</td><td>0.0177</td><td>0.0212</td></tr><tr><td></td><td>4</td><td>0.0071</td><td>0.0105</td><td>0.0200</td><td>0.0235</td><td>0.0150</td><td>0.0189</td></tr><tr><td></td><td>5</td><td>0.0105</td><td>0.0135</td><td>0.0183</td><td>0.0217</td><td>0.0350</td><td>0.0453</td></tr><tr><td></td><td>6</td><td>0.0063</td><td>0.0097</td><td>0.0204</td><td>0.0242</td><td>0.0149</td><td>0.0194</td></tr><tr><td>TJU</td><td>1</td><td>0.0164</td><td>0.0158</td><td>0.0206</td><td>0.0197</td><td>0.0198</td><td>0.0208</td></tr><tr><td></td><td>2</td><td>0.0119</td><td>0.0132</td><td>0.0149</td><td>0.0157</td><td>0.0143</td><td>0.0149</td></tr><tr><td></td><td>3</td><td>0.0080</td><td>0.0079</td><td>0.0150</td><td>0.0144</td><td>0.0124</td><td>0.0125</td></tr><tr><td>MIT</td><td></td><td>0.0065</td><td>0.0074</td><td>0.0079</td><td>0.0087</td><td>0.0065</td><td>0.0075</td></tr><tr><td>HUST</td><td></td><td>0.0078</td><td>0.0087</td><td>0.0080</td><td>0.0090</td><td>0.0074</td><td>0.0087</td></tr></table>

MAPE is the mean absolute percentage error, and RMSE is the root mean square error. The best results are shown in bold. All values are averaged from ten experiments.

Table 3 | Results of small sample experiments on the XJTU dataset batch 1 and HUST dataset 

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Train batteries</td><td colspan="2">Ours</td><td colspan="2">MLP</td><td colspan="2">CNN</td></tr><tr><td>MAPE</td><td>RMSE</td><td>MAPE</td><td>RMSE</td><td>MAPE</td><td>RMSE</td></tr><tr><td>XJTU</td><td>1</td><td>0.0141</td><td>0.0184</td><td>0.0343</td><td>0.0390</td><td>0.0929</td><td>0.0949</td></tr><tr><td></td><td>2</td><td>0.0105</td><td>0.0134</td><td>0.0267</td><td>0.0304</td><td>0.0728</td><td>0.0826</td></tr><tr><td></td><td>3</td><td>0.0069</td><td>0.0096</td><td>0.0347</td><td>0.0383</td><td>0.0548</td><td>0.0666</td></tr><tr><td></td><td>4</td><td>0.0056</td><td>0.0076</td><td>0.0292</td><td>0.0327</td><td>0.0560</td><td>0.0647</td></tr><tr><td>HUST</td><td>1</td><td>0.0446</td><td>0.0485</td><td>0.0601</td><td>0.0682</td><td>0.3614</td><td>0.1550</td></tr><tr><td></td><td>2</td><td>0.0178</td><td>0.0202</td><td>0.0391</td><td>0.0461</td><td>0.0826</td><td>0.0925</td></tr><tr><td></td><td>3</td><td>0.0154</td><td>0.0181</td><td>0.0251</td><td>0.0287</td><td>0.0514</td><td>0.0618</td></tr><tr><td></td><td>4</td><td>0.0144</td><td>0.0173</td><td>0.0253</td><td>0.0288</td><td>0.0429</td><td>0.0521</td></tr></table>

MAPE is the mean absolute percentage error, and RMSE is the root mean square error. The best results are shown in bold. All values are averaged from ten experiments. The “Train Batteries” means that we use 1, 2, 3, and 4 batteries to train the model respectively, and then test it on test set.

(a)   

Fig. 5 | An illustration of test root mean square error (RMSE) distributions for three models (the proposed PINN (Ours), multi-layer perceptron (MLP), and convolutional neural network (CNN)) on two datasets. Each error bar contains 10 points (10 experiment) and is marked with mean and standard deviation lines. The “1 battery” in the legend means that we only use the data of 1 battery to train the

parameter amounts are used as comparison methods. The details of MLP and CNN can be found in Supplementary Note 3. For each dataset, we divide the training batteries, validation batteries, and test batteries approximately in a ratio of 6:2:2. The number of test batteries in each dataset can be found in Table S2. To ensure fairness, the numbers of test batteries are evenly distributed throughout the dataset, as shown in Tables S3–S6. The results of the 3 models on the 4 datasets are shown in Table 2 (only the average test errors on each dataset are given, and the test results of each battery can be viewed in Tables S3–S6. It can be seen from the table that our method has the smallest estimation errors in most cases. The average MAPE of the proposed PINN on the 4 datasets is 0.85%, 1.21%, 0.65%, and 0.78%, while that of MLP is 2.60%, 1.72%, 0.83%, and 0.83%. It is worth noting that they have the same number of parameters and model structure during inference.

Further, to reflect the stability of the model, the training and testing process of each model on each dataset is repeated 10 times. The test results are shown in Fig. 4b. From the figure, we can see that our proposed PINN is the most stable on all tasks and all metrics. The sample size in each batch of the XJTU battery dataset is small, causing significant fluctuations in MLP and CNN. In contrast, our method is more stable and yields a smaller test error. For HUST dataset and MIT dataset, they contain a large number of training samples, so the fluctuations of MLP and CNN become smaller, and the test errors become smaller. However, our proposed PINN is still the bestperforming model.

# Experiments with small samples

Our proposed PINN models battery degradation dynamics, taking into account more physical laws and thus can be trained with less data. Compared with pure data-driven methods, our method can show greater superiority when the amount of available training data is small. To verify the above inference, small sample experiments are conducted on the XJTU dataset and HUST dataset.

Specifically, we use 1 battery data to train 3 models, and test on multiple batteries (the test set is the same as in 2.4), and record the test results. In addition, we gradually increase the number of training batteries and observe the performance change of each model on the test set. The results are given in Table 3 and Fig. 5 (only the batch 1 results are given for the XJTU dataset, more results can be found in Table S7 and Fig. S5).

It can be observed that our proposed PINN obtains the best results in all tasks and settings. As the number of training batteries increases, the test errors decrease for all 3 models. This is a generally accepted fact: increasing the number of training samples can improve the model performance when the training data is small. In Fig. 4b, due to the large number of samples in the HUST dataset, the performance of the CNN and MLP is comparable to that of our PINN. This also illustrates the fact that when the structure or number of parameters of the models are the same or similar, given enough training samples, the model performance does not differ much. However, it is evident from Fig. 5 that our method has a significant advantage when the number of training samples is small. In addition, it is worth noting that the performance of our PINN trained with only 1 battery is comparable to that of MLP and CNN trained with 3–4 batteries, which demonstrates the superiority of our PINN in the small sample scenario.

(b)   

model. Others are similar. As the number of batteries increases, the performance of the three models is getting better. However, our method still performs best among them. a The results on the XJTU dataset batch 1. b The results on the HUST dataset. Source data are provided as a Source Data file.

(a) Training

(b) Fine-tuning   
Fig. 6 | An illustration of the proposed physics-informed neural network. a The extracted features x and cycle are used to estimate SOH . The ^ represent the t u uestimated SOH, and the subscripts and x represent the corresponding partial derivatives. Neural networks Fð-Þ and Gð-Þ is used to model the mapping between   
features to SOH and the degradation dynamics of battery, respectively. b When the proposed PINN is applied to transfer learning scenarios, the dynamics Gð-Þ is frozen and only solution is fine-tuned.

Table 4 | The test root mean square error (RMSE) of fine-tuning experiments among four datasets 

<table><tr><td rowspan="2"></td><td colspan="4">Fine-tuning</td><td colspan="4">Source-only</td><td colspan="2">Train with target cell</td></tr><tr><td>XJTU</td><td>TJU</td><td>MIT</td><td>HUST</td><td>XJTU</td><td>TJU</td><td>MIT</td><td>HUST</td><td>1</td><td>2</td></tr><tr><td>XJTU</td><td>-</td><td>0.0100</td><td>0.0145</td><td>0.0104</td><td>-</td><td>0.0967</td><td>0.1329</td><td>0.0733</td><td>0.0184</td><td>0.0134</td></tr><tr><td>TJU</td><td>0.0093</td><td>-</td><td>0.0119</td><td>0.0146</td><td>0.1266</td><td>-</td><td>0.1674</td><td>0.1266</td><td>0.0121</td><td>0.0202</td></tr><tr><td>MIT</td><td>0.0239</td><td>0.0272</td><td>-</td><td>0.0248</td><td>0.0347</td><td>0.1277</td><td>-</td><td>0.0561</td><td>0.0324</td><td>0.0142</td></tr><tr><td>HUST</td><td>0.0333</td><td>0.0343</td><td>0.0307</td><td>-</td><td>0.1131</td><td>0.2008</td><td>0.0801</td><td>-</td><td>0.0485</td><td>0.0202</td></tr></table>

The top 3 results are in bold, italic, and underlined respectively. All values are averaged from ten experiments. View the table in terms of rows. The first row represents that the XJTU dataset is used as the target domain, and other datasets are used as the source domain. “Fine-tuning” means that the PINN was trained on the source domain, then fine-tuned with the data from the 1st battery in the target domain, and tested on the test set of the target domain. “Source-only” means that the PINN was trained on the source domain, and then tested on the test set of target domain directly. “Train with target cell” represents that the PINN was trained with 1 or 2 batteries from the target domain and then tested on the test set of the target domain (the same as the small sample experiments). For convenience, we only select the data in XJTU batch 1 to represent the XJTU dataset. Similarly, batch 3 of the TJU dataset is used to represent the TJU dataset.

# Fine-tuning between different datasets

Fine-tuning is one implementation of transfer learning, which improves learning ability by rapidly tuning the model using a small amount of newly collected data. The advantage is that it can use the massive data collected in other scenarios (source domain) to pre-train a model and learn the essential relation between features and labels. Then a small amount of target domain data is used to quickly fine-tune the model to obtain good performance. Most of the existing studies on transfer learning for SOH estimation are transfers between different charge/discharge protocols, and there are few studies on transfers between different datasets (different chemical compositions). In this paper, we combined 4 datasets in pairs for fine-tuning experiments.

We believe that the degradation dynamics are independent of charge/discharge protocols and datasets, while the solution  is related to them. After learning from massive data, should contain general information that can reflect the nature of battery degradation, which is useful for cross-scenario SOH estimation. Therefore, we only fine-tune the weights of the solution and make the weights of dynamics G frozen, as shown in Fig. 6b.

We carried out fine-tuning experiments and source-only experiments, and also compared them with the small sample experiments. All results are given in Table 4. It can be seen from the figure that the finetuned model is significantly better than the source-only method. What is more, when there is only 1 labeled target domain battery, the models following the “pre-training—fine-tuning” paradigm perform better than models trained directly using 1 target domain battery. This demonstrates the effectiveness of the “pre-training—fine-tuning” paradigm. For the XJTU dataset and TJU dataset, even if the model is trained with 2 target domain batteries, its performance is not as good as that of the model fine-tuned with 1 target domain battery. This also proves that dynamics has learned useful information from a large amount of data in the source domain.

There also seem to be some intuitively correct but less obvious insights if Table 4 is revisited from a fairer perspective, i.e., ignoring the last column of the table. Both the XJTU dataset and the TJU dataset are LiNiCo-x batteries, and the correlation between the features and SOH is more similar (see section “Feature extraction” for correlations), so the fine-tuning effect between them is better. Similarly, both MIT and HUST are LiFePO4 batteries, and the fine-tuning effect between them is also promising. This may be a meaningful finding, and we will continue to study it in the future.

# Discussion

Accurate SOH estimation facilitates health management and maintenance decisions of lithium-ion batteries. Existing SOH estimation methods need to extract different features for different datasets, and the performance of the model fluctuates greatly. In this work, we propose a general PINN for battery SOH estimation. Specifically, we propose a general feature extraction method to extract statistical features from a short period of data before the battery is fully charged, which is included in batteries charged with a constant-current and constant-voltage mode. Then, we modeled the battery degradation dynamics with a PINN, and the SOH was estimated by taking the extracted features as inputs.

To validate our approach, we performed battery aging experiments and developed a dataset with 55 batteries. Finally, we validate our method on 387 batteries with different chemistries and charge/ discharge protocols from 4 large-scale datasets. The results demonstrated the effectiveness and feasibility of our proposed method. Further, we conduct small sample experiments and transfer experiments, proving that considering physical knowledge helps data-driven models to learn faster and better from data. Our study highlights the promise of physics-informed machine learning in battery degradation modeling and SOH estimation. It can facilitate the rapid development of battery management systems for next-generation batteries using existing experimental data and small new data.

Battery degradation modeling and SOH estimation are research hotspots in the field of battery health management. As batteries aging, various interface degradation processes occur, along with the loss of lithium inventory and active materials, leading to increased resistance in ion and electron transfer as well as intercalation reactions, thereby resulting in changes in their charging curves46. Consequently, the charging curve contains rich information on the degradation process. However, using charge and discharge curves to estimate battery SOH may fall into the trap of information leakage. Geslin et al.47 pointed out that inconsistent charging and discharging protocols, usage conditions, etc. may lead to information leakage, which is a serious problem that may be ignored by scholars. They believe that a fixed CC-CV mode can alleviate the problem of information leakage. Hence, it is advisable to avoid incorporating factors related to internal battery quality, manufacturing variability, and usage conditions as much as possible when performing SOH estimation tasks. In our study, the features are extracted from a small segment of data from the CC-CV stage before the battery is fully charged, which is independent of the battery usage conditions. This ensures the usefulness and versatility of the features we extracted, while avoiding the problem of information leakage caused by inconsistent charging protocols or battery usage conditions. During the training and test stage, we train the model with data from battery A and test it on battery B; instead of training the model with early data from battery A and testing it with later data, which avoids information leakage from the training set to the test set.

When building the SOH estimation model, we proposed a PINN for battery SOH estimation. Physics-informed neural network holds promise as an effective avenue for leveraging artificial intelligence to address practical engineering problems. By amalgamating traditional physics models with neural networks models, it can more accurately capture the intricate dynamic behavior of battery systems, thereby facilitating more reliable and precise state estimation. However, this burgeoning field still requires further exploration by scholars. Within the framework proposed by Aykol et al.38, Hybrid methods, which utilize physical equations to constrain neural networks or integrate physical equations into neural networks, will become dominant in the long term. This class of hybrid methods have the potential to blend the causality and extrapolation capabilities of physics-based models with the speed, flexibility, and high-dimensional capabilities of neural networks. However, the limitation of these methods lies in the complexity of the battery’s physical model (e.g., the P2D model), which contains numerous parameters, and the internal parameters of the battery are difficult to collect. There is currently no satisfactory method to seamlessly integrate physical models and neural networks. The PINN proposed in this paper is modeled from the perspective of empirical degradation and state space equations, serving merely as an exploration of such hybrid methods and acting as a catalyst for further research. Additionally, we only consider extracting features from easily accessible current and voltage data. As more data and internal variables become available, more complex electrochemical models can be considered. The optimal integration of battery governing equations and neural networks for health management within the constraints of existing data and computational resources remains ripe for further exploration.

# Methods

# Battery degradation modeling

Battery aging is primarily characterized by a decrease in available capacity and an increase in internal resistance, typically following a declining trajectory. To accurately describe the battery degradation trajectory, scholars have proposed various empirical models to describe the loss of battery capacity as a function of time or cycle numbers, including the linear model48, exponential model49,50, powerlaw model51, and failure forecast model (FFM)52, etc. These models all describe the battery’s degradation trajectory as a univariate function of time.

However, representing the degradation trajectory of batteries solely as a univariate function of time oversimplifies the process. In fact, battery degradation is not only related to time but also related to charging rate, discharging rate, calendar time, temperature, depth of discharge (DOD), etc. For example, Xu et al.53 divided battery aging into calendar aging and cycle aging, which considered factors such as state-of-charge (SOC), DOD, cell temperature, and solid electrolyte interphase (SEI) film growth. They modeled calendar aging and cycle aging as functions of calendar time, SOC, DOD, and temperature.

Therefore, modeling the degradation trajectory of a battery solely as a function of time is inadequate. In this study, we propose to model it as a multivariate function:

$$
u = f (t, \mathbf {x}), \tag {1}
$$

where  represents time and x represents a vector composed of SOC, tDOD, temperature, charge rate, discharge rate, health indicators (HIs), and all other factors. In our work, x represents the HIs extracted from the charging data (see “Feature extraction” section for more details).

Without loss of generality, to describe the degradation dynamics of the battery, its SOH decay rate can be described as:

$$
\frac {\partial u}{\partial t} = g (t, \mathbf {x}, u; \theta). \tag {2}
$$

The above equation is an explicit partial differential equation (PDE) parameterized by θ, and ( ⋅ ) represents the nonlinear function of , x, g tand . The function ( ⋅ ) characterizes the internal degradation u gdynamics of the battery, and by altering this nonlinear function, various forms of degradation can be represented. Models such as linear model, exponential model, power-law model, and FFM can be viewed as particular cases of Eq. (2) when only the time is considered.

# Physics-informed neural network

An unavoidable problem is that the explicit form of $g ( \cdot )$ is unknown gand difficult to obtain. In response to similar problems, Sun et al.36 proposed a sparse regression physics-informed neural network that exploits sparsity to learn the parameters θ of $g ( \cdot )$ from a given cangdidate set. Raissi et al.54 proposed deep hidden physics models to model $g ( \cdot ) .$ . Inspired $\mathsf { b y } ^ { 5 4 , 5 5 }$ , we propose to use a more generalized gfunction approximator 0 with parameters $\theta ^ { \prime }$ to represent the nonglinear dynamics ( ⋅ ). Therefore, Eq. (2) becomes:

$$
u _ {t} \approx g ^ {\prime} (t, \mathbf {x}, u, u _ {t}, u _ {\mathbf {x}}, u _ {\mathbf {x x}}, \dots ; \theta^ {\prime}). \tag {3}
$$

In the equation, $\begin{array} { r } { u _ { t } = \frac { \partial u } { \partial t } , } \end{array}$ , we employ a neural network $\mathcal { F } ( t , \mathbf { x } ; \Phi )$ with ut t tlearnable parameters Φ to model ( , x) and utilize automatic differentiation mechanisms to compute $\begin{array} { r } { \dot { \boldsymbol { u } } _ { t } . \boldsymbol { u } _ { \mathbf { x } } = \left\lceil \frac { \partial u } { \partial x _ { 1 } } , \frac { \partial u } { \partial x _ { 2 } } , \cdot \cdot \cdot \right\rceil } \end{array}$ represents the ut u x1 x2first-order partial derivative of  with respect to x, and $u _ { { \bf x } { \bf x } }$ represents u uthe second-order partial derivative. One advantage of this approach is that we do not need to specify a candidate basis function set for $g ( \cdot )$ , but instead employ a more generalized approximators $g ^ { \prime } ( \cdot )$ g. The gfunction approximator 0 propose a more flexible relationship to , , g t ux, and their arbitrary order partial derivatives. A neural network with learnable parameters Θ is used to model $g ^ { \prime } ( \cdot )$ so that it can learn gthe aging mechanism of the battery from the given x, , and other tpartial derivatives. To balance accuracy and computational complexity, we only consider the influence of first-order partial derivatives, discarding higher-order derivatives.

Building upon the aforementioned analysis, we define a physicsinformed neural network $\mathcal { H } ^ { 3 7 , 5 5 }$ for battery aging:

$$
\mathcal {H} := \frac {\partial \mathcal {F} (t , \mathbf {x} ; \Phi)}{\partial t} - \mathcal {G} (t, \mathbf {x}, u, u _ {t}, u _ {\mathbf {x}}; \Theta), \tag {4}
$$

where $\frac { \partial \mathcal { F } ( t , \mathbf { x } ; \Phi ) } { \partial t }$ represents the partial derivation of solution neural network $\mathcal F ( \cdot )$ with respect to , and  denotes the battery degradattion dynamic equation modeled by the neural network. The structure of the proposed PINN is shown in Fig. 6.

Equation (4) is derived from Eqs. (2) and (3). However, since it is fitted by a neural network, its training process is discrete, so it does not strictly satisfy Eq. (2). For battery SOH, the calculation formula is13:

$$
u ^ {k} = f (k, \mathbf {x}) = \frac {Q ^ {k}}{Q ^ {0}}, \tag {5}
$$

where $Q ^ { k }$ represents the capacity of cycle  and $Q ^ { 0 }$ represents the Q k Qnominal capacity. The SOH value k coincides with the point on the udegradation trajectory ( ⋅ ) when = . We need to make $\mathcal { H } ( t ^ { i } , \pmb { x } ^ { i } ) = 0$ f t k thold at sample point  to approximate Eq. (2). Therefore, the optimiization process of the PINN needs to adhere to the PDE loss specified by Eq. (2), i.e.:

$$
\mathcal {L} _ {\mathrm{PDE}} = \sum_ {i = 1} ^ {N} \left| \mathcal {H} (t ^ {i}, \mathbf {x} ^ {i}) \right| ^ {2}, \tag {6}
$$

where superscript denotes the th sample and denotes the number i i Nof samples. Also, the optimization objective includes data item loss and monotonicity loss:

$$
\mathcal {L} _ {\text { data }} = \sum_ {i = 1} ^ {N} \left| u ^ {i} - \hat {u} ^ {i} \right| ^ {2}, \tag {7}
$$

$$
\mathcal {L} _ {\mathrm{mono}} = \sum_ {j = 1} ^ {M} \sum_ {k = 1} ^ {N _ {j}} \operatorname{ReLU} \left(\hat {u} ^ {k + 1} - \hat {u} ^ {k}\right), \tag {8}
$$

where $\hat { \boldsymbol { u } } ^ { i }$ represents the estimated SOH,  represents the number of ubatteries, $N _ { j }$ Mdenotes the number of cycles of battery , and ReLU( ⋅ ) is NjRectified Linear Unit. The monotonicity loss $\mathcal { L } _ { \mathrm { m o n o } }$ jis based on the physical properties of battery degradation, that is, the SOH of the next cycle should be less than or equal to that of the previous cycle (unless capacity regeneration occurs). The total function is formulated as:

$$
\mathcal {L} = \mathcal {L} _ {\text { data }} + \alpha \mathcal {L} _ {\text { PDE }} + \beta \mathcal {L} _ {\text { mono }}, \tag {9}
$$

where the α and $\beta$ are trade-off parameters. More details about our model can be found in Supplementary Note 3.

# Transfer learning with physics-informed neural network

Our PINN for battery aging consists of two parts: a solution neural network $\mathcal { F } ( \cdot )$ that builds the feature-to-SOH mapping and a neural network that models battery degradation dynamics, as shown in Fig. 6. We believe that the degradation dynamics  are independent of charge/discharge protocols and datasets, while the solution $\mathcal { F } ( \cdot )$ is related to them. Therefore, when our PINN is applied to transfer learning scenarios, dynamics is frozen, and only solution $\mathcal F ( \cdot )$ is fine-tuned, as shown in Fig. 6b.

# Data availability

The XJTU battery dataset generated in this study is publicly available in the Zenodo database under accession code [https://doi.org/10.5281/ zenodo.10963339], as reference56. The TJU dataset is available at: https:// zenodo.org/record/6405084. The HUST dataset is available at: https:// data.mendeley.com/datasets/nsc7hnsg4s/2. The MIT dataset is available at: https://data.matr.io/1/projects/5c48dd2bc625d700019f3204. Source data are provided with this paper.

# Code availability

Our code is available on Github [https://github.com/wang-fujin/ PINN4SOH] or on Zenodo database under accession code [https:// doi.org/10.5281/zenodo.11046967], as reference57.

# References

1. Schmuch, R., Wagner, R., Hörpel, G., Placke, T. & Winter, M. Performance and cost of materials for lithium-based rechargeable automotive batteries. Nat. Energy 3, 267–278 (2018).   
2. Harper, G. et al. Recycling lithium-ion batteries from electric vehicles. Nature 575, 75–86 (2019).   
3. Zubi, G., Adhikari, R. S., Sánchez, N. E. & Acuña-Bravo, W. Lithiumion battery-packs for solar home systems: layout, cost and implementation perspectives. J. Energy Storage 32, 101985 (2020).   
4. Zhao, D. et al. Online estimation of satellite lithium-ion battery capacity based on approximate belief rule base and hidden Markov model. Energy 256, 124632 (2022).   
5. Yun, S.-T. & Kong, S.-H. Data-driven in-orbit current and voltage prediction using Bi-LSTM for LEO satellite lithium-ion battery SOC estimation. IEEE Trans. Aerosp. Electron. Syst. 58, 5292–5306 (2022).   
6. Shen, L., Cheng, Q., Cheng, Y., Wei, L. & Wang, Y. Hierarchical control of DC micro-grid for photovoltaic EV charging station based on flywheel and battery energy storage system. Electr. Power Syst. Res. 179, 106079 (2020).   
7. Deng, J., Bae, C., Denlinger, A. & Miller, T. Electric vehicles batteries: requirements and challenges. Joule 4, 511–515 (2020).   
8. Liang, Y. et al. A review of rechargeable batteries for portable electronic devices. InfoMat 1, 6–32 (2019).   
9. Markets, R. Global and China Li-ion power battery industry report, 2019-2025. Research and Markets. https://www. researchandmarkets.com/reports/5021667/globaland-china-li-ionpower-battery-industry (2020).

10. Miao, Y., Liu, L., Zhang, Y., Tan, Q. & Li, J. An overview of global power lithium-ion batteries and associated critical metal recycling. J. Hazard. Mater. 425, 127900 (2022).   
11. Zhang, Y. & Li, Y.-F. Prognostics and health management of lithiumion battery using deep learning methods: a review. Renew. Sustain. Energy Rev. 161, 112282 (2022).   
12. Berecibar, M. et al. Critical review of state of health estimation methods of Li-ion batteries for real applications. Renew. Sustain. Energy Rev. 56, 572–587 (2016).   
13. Ng, M.-F., Zhao, J., Yan, Q., Conduit, G. J. & Seh, Z. W. Predicting the state of charge and health of batteries using data-driven machine learning. Nat. Mach. Intell. 2, 161–170 (2020).   
14. Che, Y., Hu, X., Lin, X., Guo, J. & Teodorescu, R. Health prognostics for lithium-ion batteries: mechanisms, methods, and prospects. Energy Environ. Sci. 16, 338–371 (2023).   
15. Wang, F. et al. A transferable lithium-ion battery remaining useful life prediction method from cycle-consistency of degradation trend. J. Power Sources 521, 230975 (2022).   
16. Baghdadi, I., Briat, O., Delétage, J.-Y., Gyan, P. & Vinassa, J.-M. Lithium battery aging model based on Dakin’s degradation approach. J. Power Sources 325, 273–285 (2016).   
17. Xu, X. et al. Remaining useful life prediction of lithium-ion batteries based on wiener process under time-varying temperature condition. Reliab. Eng. Syst. Saf. 214, 107675 (2021).   
18. Dong, G. & Wei, J. A physics-based aging model for lithium-ion battery with coupled chemical/mechanical degradation mechanisms. Electrochim. Acta 395, 139133 (2021).   
19. Lui, Y. H. et al. Physics-based prognostics of implantable-grade lithium-ion battery for remaining useful life prediction. J. Power Sources 485, 229327 (2021).   
20. Attia, P. M. et al. Closed-loop optimization of fast-charging protocols for batteries with machine learning. Nature 578, 397–402 (2020).   
21. Rauf, H., Khalid, M. & Arshad, N. Machine learning in state of health and remaining useful life estimation: theoretical and technological development in battery degradation modelling. Renew. Sustain. Energy Rev. 156, 111903 (2022).   
22. Li, W., Zhang, H., van Vlijmen, B., Dechent, P. & Sauer, D. U. Forecasting battery capacity and power degradation with multi-task learning. Energy Storage Mater. 53, 453–466 (2022).   
23. Wang, F. et al. Explainability-driven model improvement for SOH estimation of lithium-ion battery. Reliab. Eng. Syst. Saf. 232, 109046 (2023).   
24. Berecibar, M. Machine-learning techniques used to accurately predict battery life. Nature 568, 325-326 (2019).   
25. Severson, K. A. et al. Data-driven prediction of battery cycle life before capacity degradation. Nat. Energy 4, 383–391 (2019).   
26. Nuhic, A., Terzimehic, T., Soczka-Guth, T., Buchholz, M. & Dietmayer, K. Health diagnosis and remaining useful life prognostics of lithium-ion batteries using data-driven methods. J. Power Sources 239, 680–688 (2013).   
27. Richardson, R. R., Osborne, M. A. & Howey, D. A. Gaussian process regression for forecasting battery state of health. J. Power Sources 357, 209–219 (2017).   
28. Luo, K., Chen, X., Zheng, H. & Shi, Z. A review of deep learning approach to predicting the state of health and state of charge of lithium-ion batteries. J. Energy Chem. 74, 159–173 (2022).   
29. Wang, F. et al. Feature disentanglement and tendency retainment with domain adaptation for lithium-ion battery capacity estimation. Reliab. Eng. Syst. Saf. 230, 108897 (2023).   
30. Xia, F., Wang, K. & Chen, J. State of health and remaining useful life prediction of lithium-ion batteries based on a disturbance-free incremental capacity and differential voltage analysis method. J. Energy Storage 64, 107161 (2023).

31. Wang, J. et al. High-efficient prediction of state of health for lithiumion battery based on AC impedance feature tuned with Gaussian process regression. J. Power Sources 561, 232737 (2023).   
32. Fuller, T. F., Doyle, M. & Newman, J. Simulation and optimization of the dual lithium ion insertion cell. J. Electrochem. Soc. 141, 1 (1994).   
33. Liu, X. et al. A generalizable, data-driven online approach to forecast capacity degradation trajectory of lithium batteries. J. Energy Chem. 68, 548–555 (2022).   
34. Wang, F. et al. Remaining useful life prediction of lithium-ion battery based on cycle-consistency learning. in 2021 International Conference on Sensing, Measurement & Data Analytics in the era of Artificial Intelligence (ICSMD) 1–6 (IEEE, 2021).   
35. Lin, M. et al. A data-driven approach for estimating state-of-health of lithium-ion batteries considering internal resistance. Energy 277, 127675 (2023).   
36. Chen, Z., Liu, Y. & Sun, H. Physics-informed learning of governing equations from scarce data. Nat. Commun. 12, 6136 (2021).   
37. Karniadakis, G. E. et al. Physics-informed machine learning. Nat. Rev. Phys. 3, 422–440 (2021).   
38. Aykol, M. et al. Perspective—Combining physics and machine learning to predict battery lifetime. J. Electrochem. Soc. 168, 030525 (2021).   
39. Nascimento, R. G., Viana, F. A., Corbetta, M. & Kulkarni, C. S. A framework for Li-ion battery prognosis based on hybrid Bayesian physics-informed neural networks. Sci. Rep. 13, 13856 (2023).   
40. Thelen, A. et al. Integrating physics-based modeling and machine learning for degradation diagnostics of lithium-ion batteries. Energy Storage Mater. 50, 668–695 (2022).   
41. Shi, J., Rivera, A. & Wu, D. Battery health management using physics-informed machine learning: online degradation modeling and remaining useful life prediction. Mech. Syst. Signal Process. 179, 109347 (2022).   
42. Wang, F. et al. Inherently interpretable physics-informed neural network for battery modeling and prognosis. IEEE Trans. Neural Netw. Learn. Syst. 1–15 https://doi.org/10.1109/TNNLS.2023. 3329368 (2023).   
43. Hofmann, T. et al. Physics-informed neural networks for state of health estimation in lithium-ion batteries. J. Electrochem. Soc. 170, 090524 (2023).   
44. Zhu, J. et al. Data-driven capacity estimation of commercial lithiumion batteries from voltage relaxation. Nat. Commun. 13, 2261 (2022).   
45. Ma, G. et al. Real-time personalized health status prediction of lithium-ion batteries using deep transfer learning. Energy Environ. Sci. 15, 4083–4094 (2022).   
46. Jiang, B. et al. Bayesian learning for rapid prediction of lithium-ion battery-cycling protocols. Joule 5, 3187–3203 (2021).   
47. Geslin, A. et al. Chueh, selecting the appropriate features in battery lifetime predictions. Joule 7, 1956–1965 (2023).   
48. Spotnitz, R. Simulation of capacity fade in lithium-ion batteries. J. Power Sources 113, 72–80 (2003).   
49. He, W., Williard, N., Osterman, M. & Pecht, M. Prognostics of lithiumion batteries based on Dempster–Shafer theory and the Bayesian Monte Carlo method. J. Power Sources 196, 10314–10321 (2011).   
50. Chen, C. & Pecht, M. Prognostics of lithium-ion batteries using model-based and data-driven methods. in Proceedings of the IEEE 2012 Prognostics and System Health Management Conference (PHM-2012 Beijing), 1–6 (IEEE, 2012).   
51. Ramadesigan, V. et al. Parameter estimation and capacity fade analysis of lithium-ion batteries using reformulated models. J. Electrochem. Soc. 158, A1048 (2011).   
52. Najera-Flores, D. A., Hu, Z., Chadha, M. & Todd, M. D. A physicsconstrained Bayesian neural network for battery remaining useful life prediction. Appl. Math. Model. 122, 42–59 (2023).

53. Xu, B., Oudalov, A., Ulbig, A., Andersson, G. & Kirschen, D. S. Modeling of lithium-ion battery degradation for cell life assessment. IEEE Trans. Smart Grid 9, 1131–1140 (2016).   
54. Raissi, M. Deep hidden physics models: deep learning of nonlinear partial differential equations. J. Mach. Learn. Res. 19, 932–955 (2018).   
55. Raissi, M., Perdikaris, P. & Karniadakis, G. E. Physics-informed neural networks: a deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. J. Comput. Phys. 378, 686–707 (2019).   
56. Wang, F. Project—Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis. https://doi. org/10.5281/zenodo.10963339 (2024).   
57. Wang, F. wang-fujin/PINN4SOH: physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis. https://doi.org/10.5281/zenodo.11046967 (2024).

# Acknowledgements

This work was supported in part by the National Natural Science Foundation of China under Grand 52105116 (Z.Z. (Zhibin Zhao)) and Grand 92060302 (X.C.); the Fundamental Research Funds for the Central Universities (xzy022023060) (F.W.).

# Author contributions

F.W. was responsible for conceptualization, methodology design, conducting experiments, and drafting the original manuscript. Z.Z. (Zhi Zhai) extensively reviewed and edited the manuscript, providing valuable suggestions and revisions. Z.Z. (Zhibin Zhao) contributed to conceptualization and methodology discussions, playing a significant role in the editing process. Y.D. conducted comprehensive reviews and edits, significantly contributing to the refinement of the article. X.C. spearheaded the acquisition of funds necessary for this research, providing crucial support.

# Competing interests

The authors declare no competing interests.

# Additional information

Supplementary information The online version contains supplementary material available at https://doi.org/10.1038/s41467-024-48779-z.

Correspondence and requests for materials should be addressed to Zhibin Zhao or Xuefeng Chen.

Peer review information Nature Communications thanks the anonymous reviewers for their contribution to the peer review of this work. A peer review file is available.

Reprints and permissions information is available at

http://www.nature.com/reprints

Publisher’s note Springer Nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations.

Open Access This article is licensed under a Creative Commons Attribution 4.0 International License, which permits use, sharing, adaptation, distribution and reproduction in any medium or format, as long as you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons licence, and indicate if changes were made. The images or other third party material in this article are included in the article’s Creative Commons licence, unless indicated otherwise in a credit line to the material. If material is not included in the article’s Creative Commons licence and your intended use is not permitted by statutory regulation or exceeds the permitted use, you will need to obtain permission directly from the copyright holder. To view a copy of this licence, visit http://creativecommons.org/ licenses/by/4.0/.

© The Author(s) 2024
