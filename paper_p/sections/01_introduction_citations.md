# 01_introduction 引文补充汇总（最新版本）

说明：本表按“引文所在原句 → 文献条目（APA）→ DOI → PDF下载 → 文献原句（可核验）→ 引用理由”的顺序整理。本清单仅覆盖 01_introduction.tex 中已实际插入的引用点。

---

## 引用点 1
- 引文所在原句：
  大规模边缘 IoT 系统（Edge IoT Systems）支撑智慧农业、智慧城市、工业互联网等关键基础设施，其核心挑战在于长期无人值守条件下保持网络寿命、能量平衡与信息新鲜度\cite{alfuqaha2015iot}。
- 文献（APA）：
  Al-Fuqaha, A., Guizani, M., Mohammadi, M., Aledhari, M., & Ayyash, M. (2015). Internet of Things: A Survey on Enabling Technologies, Protocols, and Applications. IEEE Communications Surveys & Tutorials, 17(4), 2347–2376. https://doi.org/10.1109/COMST.2015.2444095
- DOI：10.1109/COMST.2015.2444095
- PDF：https://cs.wmich.edu/alfuqaha/spring15/cs6570/lectures/IoT-survey.pdf
- 文献原句（可核验，Abstract）：
  “This paper provides an overview of the Internet of Things (IoT) with emphasis on enabling technologies, protocols, and application issues.”
- 引用理由：
  作为高被引IoT综述，为“关键基础设施中的IoT应用与系统级挑战”提供权威背景支撑。

---

## 引用点 2
- 引文所在原句：
  无线传感器网络（Wireless Sensor Networks, WSN）只是其中一种典型实例：能量受限的 WSN 设备已经嵌入智慧农业、智慧城市环境监测、分布式结构健康监测以及工业 IoT 系统等场景，必须在缺乏维护的情况下稳定运行数月乃至数年\cite{yick2008survey}。
- 文献（APA）：
  Yick, J., Mukherjee, B., & Ghosal, D. (2008). Wireless sensor network survey. Computer Networks, 52(12), 2292–2330. https://doi.org/10.1016/j.comnet.2008.04.002
- DOI：10.1016/j.comnet.2008.04.002
- PDF：https://www.science.smith.edu/~jcardell/Courses/EGR328/Readings/WSNSurvey2.pdf
- 文献原句（可核验，Abstract）：
  “Wireless sensor networks (WSNs) have gained worldwide attention in recent years, particularly with the proliferation in Micro-Electro-Mechanical Systems (MEMS) technology which has facilitated the development of smart sensors.”
- 引用理由：
  经典WSN综述，支撑“WSN广泛应用、长期运行与能量受限”的基础论断。

---

## 引用点 3
- 引文所在原句：
  由此，IoT sustainability、edge intelligence 与 self-optimizing IoT 的系统级需求日益凸显，而信息与能量过程彼此分裂导致的结构性瓶颈愈发严重\cite{alfuqaha2015iot}。
- 文献（APA）：
  Al-Fuqaha, A., Guizani, M., Mohammadi, M., Aledhari, M., & Ayyash, M. (2015). Internet of Things: A Survey on Enabling Technologies, Protocols, and Applications. IEEE Communications Surveys & Tutorials, 17(4), 2347–2376. https://doi.org/10.1109/COMST.2015.2444095
- DOI：10.1109/COMST.2015.2444095
- PDF：https://cs.wmich.edu/alfuqaha/spring15/cs6570/lectures/IoT-survey.pdf
- 文献原句（可核验，Abstract）：
  “This paper provides an overview of the Internet of Things (IoT) with emphasis on enabling technologies, protocols, and application issues.”
- 引用理由：
  用权威综述支撑“系统级需求与挑战”的宏观趋势判断。

---

## 引用点 4（同一句包含两条引用）
- 引文所在原句：
  现有研究多采用工程优化思路，包括节能调度、链路选择、机会式路由、Lyapunov 优化以及多目标调度等方法\cite{biswas2005exor,georgiadis2006crosslayer}。

(4-a) 机会式路由（ExOR）
- 文献（APA）：
  Biswas, S., & Morris, R. (2005). ExOR: Opportunistic Multi-Hop Routing for Wireless Networks. In Proceedings of SIGCOMM ’05 (pp. 133–144). ACM. https://doi.org/10.1145/1080091.1080108
- DOI：10.1145/1080091.1080108
- PDF：https://pdos.csail.mit.edu/papers/exor-sigcomm05.pdf
- 文献原句（可核验，Abstract）：
  “We present Extremely Opportunistic Routing (ExOR), a routing technique that integrates routing and MAC to reduce the cost of forwarding in multi-hop wireless networks.”
- 引用理由：
  代表“机会式路由”的里程碑工作，支撑方法学分类中的“机会式路由”。

(4-b) 跨层控制/资源分配（Lyapunov相关综述）
- 文献（APA）：
  Georgiadis, L., Neely, M. J., & Tassiulas, L. (2006). Resource allocation and cross-layer control in wireless networks. Foundations and Trends in Networking, 1(1), 1–144. https://doi.org/10.1561/1300000001
- DOI：10.1561/1300000001
- PDF（可核验全文镜像）：https://www.researchgate.net/profile/Leonidas-Georgiadis/publication/220655053_Resource_Allocation_and_Cross-Layer_Control_in_Wireless_Networks/links/09e4150b7aac9bc4ec000000/Resource-Allocation-and-Cross-Layer-Control-in-Wireless-Networks.pdf
- 文献原句（可核验，出版方摘要摘引）：
  “... presents abstract models that capture the cross layer interaction from the physical to transport layer ...”（摘自 now publishers 条目摘要）
- 引用理由：
  权威综述系统梳理基于Lyapunov/漂移-惩罚思想的跨层资源分配与控制框架，支撑“工程优化方法（Lyapunov优化等）”的范式归纳。

---

## 引用点 5（能量空洞/热点机理）
- 引文所在原句：
  …sink节点周围的传感器节点容易因频繁参与中继转发或能量共享而被耗尽能量，导致过早死亡并形成能量空洞，如何处理能量空洞也是当前WSN系统中的一个重要问题\cite{soro2005ucs}。
- 文献（APA）：
  Soro, S., & Heinzelman, W. B. (2005). Prolonging the Lifetime of Wireless Sensor Networks via Unequal Clustering. In Proceedings of the 19th IEEE International Parallel and Distributed Processing Symposium (IPDPS 2005), Workshop on Mobile Ad Hoc and Sensor Systems (WMAN). IEEE. https://doi.org/10.1109/IPDPS.2005.365
- DOI：10.1109/IPDPS.2005.365
- PDF：https://hajim.rochester.edu/ece/sites/wcng/papers/conference/soro_wman05.pdf
- 文献原句（可核验，IEEE Xplore Abstract）：
  “Oftentimes the network is organized into clusters of equal size, but such equal clustering results in an unequal load on the cluster head nodes.”
- 引用理由：
  指出“均等簇会导致簇头能量负载不均”的根因，与sink邻近热点/能量空洞机理一致，从而支撑本文关于能量空洞问题的表述。

---

## 引用点 6（ALDP 中的 Lyapunov 依据，同一句包含两条引用）
- 引文所在原句（片段）：
  …采用**自适应时长规划技术**（Adaptive Lyapunov Duration Planning, ALDP），通过自适应参数的Lyapunov优化\cite{georgiadis2006crosslayer,tassiulas1992stability}进行前瞻性传输能量时长规划…

(6-a) 跨层控制/资源分配综述（与Lyapunov相关）
- 文献（APA）：
  Georgiadis, L., Neely, M. J., & Tassiulas, L. (2006). Resource allocation and cross-layer control in wireless networks. Foundations and Trends in Networking, 1(1), 1–144. https://doi.org/10.1561/1300000001
- DOI：10.1561/1300000001
- PDF（可核验全文镜像）：https://www.researchgate.net/profile/Leonidas-Georgiadis/publication/220655053_Resource_Allocation_and_Cross-Layer_Control_in_Wireless_Networks/links/09e4150b7aac9bc4ec000000/Resource-Allocation-and-Cross-Layer-Control-in-Wireless-Networks.pdf
- 文献原句（可核验，出版方摘要摘引）：
  “... presents abstract models that capture the cross layer interaction from the physical to transport layer ...”
- 引用理由：
  提供基于Lyapunov/漂移-惩罚思想的跨层优化与控制框架背景，作为ALDP方法论依据之一。

(6-b) 稳定性/最大吞吐原理（背压/漂移-惩罚理论基石）
- 文献（APA）：
  Tassiulas, L., & Ephremides, A. (1992). Stability properties of constrained queueing systems and scheduling policies for maximum throughput in multihop radio networks. IEEE Transactions on Automatic Control, 37(12), 1936–1948. https://doi.org/10.1109/9.182479
- DOI：10.1109/9.182479
- PDF（高校镜像）：https://www.di.ens.fr/~busic/mar/projets/TE92.pdf
- 文献原句（可核验，Abstract）：
  “The stability of a queueing network with interdependent servers is considered. The dependency among the servers is described by the definition of their subsets that can be activated simultaneously.”
- 引用理由：
  该文奠定了多跳无线网络的稳定性与最大吞吐调度（背压）理论基础，为基于Lyapunov的在线规划与稳定性保证提供原理支撑。

---

备注：若你希望第5处“能量空洞”再并列添加一篇直接使用“energy hole”术语的专文，我可以继续检索并补充APA、PDF与逐字摘引。
