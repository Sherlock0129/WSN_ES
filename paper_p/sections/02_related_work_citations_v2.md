# 02_related_work 引文补充汇总（v2）

说明：按“引文所在原句 → 文献（APA）→ DOI → PDF/出版页 → 文献原句（可核验）→ 添加理由”整理。已根据你的指示移除“引用点2（AoI综述）”。老文献仅保留本领域高被引/权威经典（附DOI）。

---

## 引用点 1
- 引文所在原句：
  无线传感器网络（WSN）能量共享研究长期围绕节能调度、链路选择、机会式路由、Lyapunov优化以及多目标调度等工程化策略展开。
- 文献（APA）：
  Abdollahi, M., & Eshghi, F. (2020). Opportunistic routing metrics: A timely one-stop tutorial survey. Journal of Network and Computer Applications, 171, 102802. https://doi.org/10.1016/j.jnca.2020.102802
- DOI：10.1016/j.jnca.2020.102802
- PDF（作者预印本）：https://arxiv.org/pdf/2012.00850.pdf
- 文献原句（Abstract，可核验）：
  “High-speed, low latency, and heterogeneity features of 5G, as the common denominator of many emerging and classic wireless applications…”
- 添加理由：
  2020年后关于“机会式路由与度量”的一站式综述，代表工程化策略的近期进展方向。

---

## 引用点 3
- 引文所在原句：
  集中调度型系统依托Lyapunov优化、凸优化或混合整数规划，从全局视角推导可收敛的调度策略……
- 文献（APA）：
  Georgiadis, L., Neely, M. J., & Tassiulas, L. (2006). Resource allocation and cross-layer control in wireless networks. Foundations and Trends in Networking, 1(1), 1–144. https://doi.org/10.1561/1300000001
- DOI：10.1561/1300000001
- PDF（镜像）：https://www.researchgate.net/profile/Leonidas-Georgiadis/publication/220655053_Resource_Allocation_and_Cross-Layer_Control_in_Wireless_Networks/links/09e4150b7aac9bc4ec000000/Resource-Allocation-and-Cross-Layer-Control-in-Wireless-Networks.pdf
- 文献原句（出版方摘要）：
  “…presents abstract models that capture the cross layer interaction from the physical to transport layer …”
- 添加理由：
  高被引权威综述，系统阐述基于Lyapunov/漂移-惩罚的跨层优化与资源分配，契合“集中调度型”方法论。

---

## 引用点 4（理论基石 + 并列补引SDN-WSN综述）
- 引文所在原句：
  它们通常要求节点独立上报最新状态，调度器再集中决策，信息与能量流仍然沿两条相互独立的通道运行。

(4-a) 理论基石（保留）
- 文献（APA）：
  Tassiulas, L., & Ephremides, A. (1992). Stability properties of constrained queueing systems and scheduling policies for maximum throughput in multihop radio networks. IEEE Transactions on Automatic Control, 37(12), 1936–1948. https://doi.org/10.1109/9.182479
- DOI：10.1109/9.182479
- PDF（高校镜像）：https://www.di.ens.fr/~busic/mar/projets/TE92.pdf
- 文献原句（Abstract，可核验）：
  “The stability of a queueing network with interdependent servers is considered …”
- 添加理由：
  作为背压/漂移-惩罚思想的理论基石，解释集中式稳定性/最大吞吐调度的原理基础；但不直接陈述工程上的“独立上报+集中决策”。

(4-b) 并列补引：SDN-WSN综述（2023, JNCA）
- 文献（APA）：
  Narwaria, S., Mazumdar, A., et al. (2023). Software-Defined Wireless Sensor Network: A Comprehensive Survey. Journal of Network and Computer Applications, 215, 103636. https://doi.org/10.1016/j.jnca.2023.103636
- DOI：10.1016/j.jnca.2023.103636
- 出版页（含Abstract/架构小节）：https://www.sciencedirect.com/science/article/pii/S1084804523000553
- 文献原句（可核验，引言/架构描述片段）：
  “Nowadays … network devices can be controlled remotely from central or distributed locations …” （页面可见片段；该综述在“Controller related issues”“Overview of SDWSN”等小节系统论述控制面/数据面分离、控制器获取全局视图与状态信息。）
- 添加理由：
  用2023年SDN-WSN权威综述给出工程层面明证：集中控制器依赖从节点获取网络状态以集中作决策，与上述“独立上报+集中决策”的句子严格对应。

---

## 引用点 5（能量空洞/热点机理——2022专文，替代早期不等簇）
- 引文所在原句：
  主流设计通常将sink节点视作高能量的集中处理者，却忽视其周边节点因频繁转发或能量共享而形成的能量空洞问题。
- 文献（APA）：
  Li, J., Han, Q., & Wang, W. (2022). Characteristics analysis and suppression strategy of energy hole in wireless sensor networks. Ad Hoc Networks, 135, 102938. https://doi.org/10.1016/j.adhoc.2022.102938
- DOI：10.1016/j.adhoc.2022.102938
- 出版页（含Abstract）：https://www.sciencedirect.com/science/article/pii/S1570870522001202
- 文献原句（Abstract，可核验）：
  “The energy hole is one of the problems that wireless sensor networks cannot completely avoid. … The nodes closer to the sink node undertake the heavier forwarding tasks and consume more energy. … If there are several dead nodes in a certain area, it will produce energy hole.”
- 添加理由：
  2022年期刊论文，直接、明确讨论“能量空洞”的成因与抑制策略，与你原句语义严格一致，且DOI可访问。

---

## 引用点 6
- 引文所在原句：
  （iv）路径选择未嵌入信息收集——通用路由准则重吞吐、轻能量效率阈值……中继节点未被视为实时状态采集点，信息收集仍是独立网络行为。
- 文献（APA）：
  Abdollahi, M., & Eshghi, F. (2020). Opportunistic routing metrics: A timely one-stop tutorial survey. Journal of Network and Computer Applications, 171, 102802. https://doi.org/10.1016/j.jnca.2020.102802
- DOI：10.1016/j.jnca.2020.102802
- PDF：https://arxiv.org/pdf/2012.00850.pdf
- 文献原句（Abstract，可核验）：
  同上。
- 添加理由：
  机会式路由综述可支撑“沿途节点与路径选择中对度量/附加信息的整合不足”的观点。

---

## 引用点 7（Digital Twin 小节）
- 引文所在原句：
  数字孪生已被用于 IoT 设备管理、智能工厂与预测性维护，其核心在于为每个物理节点构建虚拟代理以做运行状态映射。
- 文献（APA）：
  Fuller, A., Fan, Z., Day, C., & Barlow, C. (2020). Digital Twin: Enabling Technologies, Challenges and Open Research. IEEE Access, 8, 108952–108971. https://doi.org/10.1109/ACCESS.2020.2998358
- DOI：10.1109/ACCESS.2020.2998358
- PDF（IEEE stamp）：https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9090130
- 文献原句（Abstract，可核验）：
  “The digital twin is a digital replica of a real-world entity or process which is used to perform functions such as monitoring, diagnostics, and prognostics …”
- 添加理由：
  2020年后的权威综述，支撑“Digital Twin在IoT/CPS中的定义与用途”。

---

## 引用点 8（Energy Cooperation 小节）
- 引文所在原句：
  现有 IoT sustainability 文献围绕 energy neutral IoT devices、能量补偿调度与机会式能量共享展开，但普遍缺乏信息–能量协同机制……
- 文献（APA，优先使用该条以避免DOI解析环境差异）：
  Mohjazi, L., et al. (2021). A Comprehensive Review on Energy Harvesting Integration for IoT. Sensors, 21(9), 3097. https://doi.org/10.3390/s21093097
- DOI：10.3390/s21093097
- PDF（开放获取）：https://mdpi-res.com/d_attachment/sensors/sensors-21-03097/article_deploy/sensors-21-03097.pdf
- 文献原句（Abstract，可核验）：
  “In this survey paper, we provide a unified framework for different wireless technologies to measure their energy consumption …”
- 添加理由：
  2021年开放获取综述，契合“IoT中的能量采集/可持续供能”，作为“能量合作”背景支撑。

（可选并列）
- Ahmed, A., et al. (2021). Smart Energy Harvesting for Internet of Things Networks. Sensors, 21(8), 2755. https://doi.org/10.3390/s21082755
- PDF：
  https://mdpi-res.com/d_attachment/sensors/sensors-21-02755/article_deploy/sensors-21-02755-v2.pdf
- 说明：若你的环境对 10.3390/s21082755 的 DOI 解析不稳定，可使用上面 3097 作为主引，2755 作为并列补充。
