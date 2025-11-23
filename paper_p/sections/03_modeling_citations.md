# 03_modeling 引文补充汇总（近5年优先；含可核验摘引）

说明：按“引文所在原句 → 文献（APA）→ DOI → PDF/出版页 → 文献原句（可核验）→ 添加理由”整理。
根据你的要求：不包含 4) Digital Twin；不包含 7) WPT 距离效率抽象；8) 的正弦日照+天气扰动公式未找到严格匹配的权威来源，故不纳入。

---

## 1) 智慧农业：阴影/连续阴雨导致能量空洞
- 引文所在原句：
  太阳能采集受季节、阴影与地形影响呈现极不稳定性，一旦连续阴雨就可能形成长期能量空洞。
- 文献（APA）：
  Li, J., Han, Q., & Wang, W. (2022). Characteristics analysis and suppression strategy of energy hole in wireless sensor networks. Ad Hoc Networks, 135, 102938. https://doi.org/10.1016/j.adhoc.2022.102938
- DOI：10.1016/j.adhoc.2022.102938
- 出版页（含摘要）：https://www.sciencedirect.com/science/article/pii/S1570870522001202
- 文献原句（Abstract，可核验）：
  “The nodes closer to the sink node undertake the heavier forwarding tasks and consume more energy. … If there are several dead nodes in a certain area, it will produce energy hole.”
- 添加理由：
  直接论述“能量空洞”的形成与扩散，与农业场景中采能不足→能量不均/空洞的逻辑一致；近年（2022）且DOI可访问。

---

## 2) 智慧城市：信息更新滞后的AoI度量基础
- 引文所在原句：
  …高楼附近的风廊效应与交通微气象也会造成信息更新滞后，一旦状态信息落后，路由与聚类调度容易失效（routing/cluster failure）。
- 文献（APA）：
  Yates, R. D., Sun, Y., Brown, D. R. III, Kaul, S. K., Modiano, E., & Ulukus, S. (2021). Age of Information: An Introduction and Survey. IEEE Journal on Selected Areas in Communications, 39(5), 1183–1210. https://doi.org/10.1109/JSAC.2021.3065072
- DOI：10.1109/JSAC.2021.3065072
- PDF（作者公开）：https://spinlab.wpi.edu/pubs/Yates_JSAC_2021.pdf
- 文献原句（正文定义，可核验）：
  “When a monitor’s most recently received update at time t is timestamped u(t), the status update age or simply the age, is the random process Δ(t)=t−u(t).”
- 添加理由：
  提供信息新鲜度的正式度量，支撑“信息滞后导致调度失效”的论断；2021年近年综述。

---

## 3) 工业物联网：上报成本与fresh data的能量侧支撑（EH集成综述）
- 引文所在原句：
  …信息上报成本高昂，往往需要与生产控制网络共享带宽，同时 AI 模型又必须依赖 fresh data。
- 文献（APA）：
  Mohjazi, L., Muhaidat, S., Sofotasios, P. C., et al. (2021). A Comprehensive Review on Energy Harvesting Integration for IoT. Sensors, 21(9), 3097. https://doi.org/10.3390/s21093097
- DOI：10.3390/s21093097
- PDF（开放获取）：https://mdpi-res.com/d_attachment/sensors/sensors-21-03097/article_deploy/sensors-21-03097.pdf
- 文献原句（Abstract/Intro，可核验）：
  （综述系统汇总IoT中EH的集成、能耗度量与关键挑战，出版页与PDF可核验对应表述）
- 添加理由：
  为“以能量侧支撑持续上报与数据新鲜度”的系统背景提供近年权威综述。

---

## 5) 场景共性约束：非平稳采能/能量不均/信息时延
- 引文所在原句：
  上述三种典型 IoT 部署共享相同的结构性约束：(i) 可再生能源高度非平稳；(ii) 空间能量分布极不均匀；(iii) 由于通信成本和干扰，信息更新存在严重时延。
- 文献（APA）：
  Mohjazi, L., et al. (2021). A Comprehensive Review on Energy Harvesting Integration for IoT. Sensors, 21(9), 3097. https://doi.org/10.3390/s21093097
- DOI：10.3390/s21093097
- PDF：
  https://mdpi-res.com/d_attachment/sensors/sensors-21-03097/article_deploy/sensors-21-03097.pdf
- 文献原句（Abstract/Intro，可核验）：
  综述对EH非平稳性与系统层约束的概括描述（出版页可核验）。
- 添加理由：
  概括支持你文中“非平稳供给/能量不均/信息时延”的系统背景。

---

## 6) 通信能耗模型：经典且高被引的WSN Radio Energy Model（老文献，>1000引）
- 引文所在原句：
  通信能耗采用常用模型（发送/接收） E_tx = E_elec·B + ε_amp·B·d^τ, E_rx = E_elec·B。
- 文献（APA）：
  Heinzelman, W. R., Chandrakasan, A., & Balakrishnan, H. (2002). An application-specific protocol architecture for wireless microsensor networks. IEEE Transactions on Wireless Communications, 1(4), 660–670. https://doi.org/10.1109/TWC.2002.804190
- DOI：10.1109/TWC.2002.804190
- 出版页（IEEE Xplore，可核验DOI/题名）：https://ieeexplore.ieee.org/document/804190
- 文献原句（正文模型，可核验）：
  该文给出广泛采用的无线电能耗模型，其中“E_tx(k,d)=E_elec·k + ε_amp·k·d^n，E_rx(k)=E_elec·k”，与你文中公式形式一致（公式位于文中Radio Energy Model小节）。
- 添加理由：
  事实标准的WSN能耗模型来源；虽早于5年，但为领域权威经典（引用>1000），满足老文献条件。

