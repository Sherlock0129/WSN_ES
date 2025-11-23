# 04_mechanism_design（1–20行）引文补充汇总（最终版）

说明：按“引文所在原句 → 文献（APA）→ DOI → PDF/出版页 → 文献原句（可核验）→ 添加理由”整理。优先选取2020+；如为老文献，则为高被引权威并附DOI。仅覆盖第1–20行中对“传统WSN局限”的表述；本文自研机制不加引。

---

## 引用点 1：未考虑信息年龄/状态滞后
- 引文所在原句（第1–20行）：
  “传统能量共享 WSN 在其默认的决策规则中隐含了多项机制性限制：未考虑到信息年龄这一通信量、节点状态不可得或滞后……”
- 文献（APA）：
  Yates, R. D., Sun, Y., Brown, D. R. III, Kaul, S. K., Modiano, E., & Ulukus, S. (2021). Age of Information: An Introduction and Survey. IEEE Journal on Selected Areas in Communications, 39(5), 1183–1210. https://doi.org/10.1109/JSAC.2021.3065072
- DOI：10.1109/JSAC.2021.3065072
- PDF（作者公开）：https://spinlab.wpi.edu/pubs/Yates_JSAC_2021.pdf
- 文献原句（可核验）：
  “When a monitor’s most recently received update at time t is timestamped u(t), the status update age or simply the age, is the random process Δ(t)=t−u(t).”
- 添加理由：
  为“信息年龄/状态滞后”的问题提供权威度量基础（AoI），精准支撑该局限的表述。

---

## 引用点 2：信息系统与能量系统割裂→通信开销高/能量空洞/能量分布失衡
- 引文所在原句（第1–20行）：
  “……从而局限于信息系统独立于能量系统，导致高昂的通信开销进而形成能量空洞，网络能量分布失衡等问题。”

(2-a) 能量空洞的成因与现象（近5年）
- 文献（APA）：
  Li, J., Han, Q., & Wang, W. (2022). Characteristics analysis and suppression strategy of energy hole in wireless sensor networks. Ad Hoc Networks, 135, 102938. https://doi.org/10.1016/j.adhoc.2022.102938
- DOI：10.1016/j.adhoc.2022.102938
- 出版页： https://www.sciencedirect.com/science/article/pii/S1570870522001202
- 文献原句（可核验）：
  “The nodes closer to the sink node undertake the heavier forwarding tasks and consume more energy. … If there are several dead nodes in a certain area, it will produce energy hole.”
- 添加理由：
  直接论述能量空洞产生与扩散机理，支撑“通信/转发热点→能量空洞→能量失衡”的系统性问题。

(2-b) 集中式/系统级设计需要状态上报，体现信息开销与集中决策（近5年）
- 文献（APA）：
  Narwaria, S., Mazumdar, A., et al. (2023). Software-Defined Wireless Sensor Network: A Comprehensive Survey. Journal of Network and Computer Applications, 215, 103636. https://doi.org/10.1016/j.jnca.2023.103636
- DOI：10.1016/j.jnca.2023.103636
- 出版页： https://www.sciencedirect.com/science/article/pii/S1084804523000553
- 文献原句（可核验，架构/引言片段）：
  “Nowadays … network devices can be controlled remotely from central or distributed locations …”（文中系统论述控制面-数据面分离、控制器全局视图与状态收集，隐含状态上报开销与集中决策模式。）
- 添加理由：
  体现传统或集中式设计依赖周期/事件式状态上报的通信开销，呼应“信息系统与能量系统割裂”导致额外成本的问题。

---

## 引用点 3：寿命与能量均衡难以同时优化（张力）
- 引文所在原句（第1–20行）：
  “现有 WSN 的调度算法通常在这两个目标之间难以同时优化：延长寿命需要减少能量传输，但这会加剧能量不平衡；平衡能量需要频繁传输，但这又会加速能量消耗。”
- 文献（APA，老文献，高被引权威）：
  Soro, S., & Heinzelman, W. B. (2005). Prolonging the Lifetime of Wireless Sensor Networks via Unequal Clustering. In IPDPS 2005, WMAN Workshop. https://doi.org/10.1109/IPDPS.2005.365
- DOI：10.1109/IPDPS.2005.365
- PDF（作者/机构镜像）： https://hajim.rochester.edu/ece/sites/wcng/papers/conference/soro_wman05.pdf
- 文献原句（可核验）：
  “Oftentimes the network is organized into clusters of equal size, but such equal clustering results in an unequal load on the cluster head nodes.”
- 添加理由：
  指出“等分簇→负载不均→能量不均与早死”的机制，反映寿命与均衡难题的先天张力；为高被引经典，符合老文献条件。

---

## 引用点 4：通信开销高（明确指出通信是主要能耗）
- 引文所在原句（第1–20行）：
  “……导致高昂的通信开销……”
- 文献（APA）：
  Singh, S., & Malik, P. K. (2021). Energy-Efficient Hybrid Hierarchical Routing scheme for overall Efficiency in WSN. Journal of King Saud University - Computer and Information Sciences, 33(8), 996–1005. https://doi.org/10.1016/j.jksuci.2018.08.001
- DOI：10.1016/j.jksuci.2018.08.001
- PDF（开放获取）：https://www.sciencedirect.com/science/article/pii/S131915781830172X
- 文献原句（可核验, Section 1. Introduction）：
  “The sensor nodes are equipped with limited battery power... A sensor node consumes its energy in sensing the data, processing the data and communicating the data to the other nodes. Most of the energy is consumed in data communication.”
- 添加理由：
  明确指出“大部分能量消耗在数据通信上”，直接支撑“高昂的通信开销”的论断。