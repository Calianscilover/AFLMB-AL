# AFLMB-AL：五组分电解液 DoE

这是一个可复现的首轮配方空间撒点方案，固定组分为 LiFSI、LiDFOB、DME、NDFA、TTE。每份目标质量为 5 g，按**估算总锂盐浓度 0.5–2.5 mol/L**筛选，再以空间填充法挑选 16 个配方（含 BAFF、LHCE 两个 baseline）。两个 baseline 各增加一次独立复配，因此共安排 18 次称量。CSV 的列名、状态值和单位标识均使用英文。

## 文件

| 文件 | 用途 |
| --- | --- |
| `config.py` | 输入：组分身份、密度及来源、设计边界、随机种子、baseline |
| `five_component_molarity_design.py` | 候选生成、约束过滤、选点和区组随机化 |
| `design_utils.py` | RDKit 校验、Sobol 抽样、距离与最远点算法 |
| `plot_formulation_distances.py` | 配方距离的二维投影和距离矩阵 |
| `outputs/formulation_test_set.csv` | 16 个不同配方的 5 g 称量值和质量分数 |
| `outputs/experimental_run_order.csv` | 18 次配制的区组及随机顺序 |
| `outputs/components.csv` | 五种物料的摩尔质量、密度及来源 |
| `outputs/composition_distance_matrix.csv` | 16 个配方的完整组成距离 |
| `outputs/composition_pca_coordinates.csv` | 距离图二维坐标 |
| `outputs/formulation_distances.png` | 配方距离图 |

## 运行

需要 Python 3.10+。在仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python five_component_molarity_design.py
python plot_formulation_distances.py
python -m unittest discover -s tests -v
```

修改 `config.py` 中的 `CONFIG` 后重新运行即可生成新一轮结果。提交的 CSV 对应配置里的随机种子 `20260920`。候选池 `outputs/feasible_candidates.csv` 会自动生成，但不纳入版本管理，以免仓库过大；绘图前需先运行设计脚本。

## 技术路线

1. 使用 RDKit 校验五个 SMILES、分子式与摩尔质量。
2. 在五组分质量单纯形上生成 scrambled Sobol 候选。
3. 换算成每份 5 g 称量单，并按 0.0001 g 步长取整；检查最低非零称量、最高质量分数及总盐质量分数。
4. 以 `n_salt = m_LiFSI/MW_LiFSI + m_LiDFOB/MW_LiDFOB` 计算总盐摩尔数；用 `V_est = Σ(m_i/ρ_i)` 估计体积（mL）；以 `C_est = 1000 × n_salt/V_est` 筛选 0.5–2.5 mol/L。
5. 保留两组 baseline，采用最远点空间填充选择 14 个新配方，并限制完整距离及二维投影距离。
6. 给 baseline 添加独立复配，在两个区组中随机排序，然后用 `pandas.to_csv` 输出。

质量列单位为 g，密度为 g/mL，浓度为 mol/L。`*_mass_fraction` 为各组分占整份 5 g 的质量比例，五列之和为 1。

## 浓度的解释与来源

`V_est` 使用纯组分体积可加假设。固体晶体密度不是盐溶解后的偏摩尔体积，混合也可能改变体积，因此新配方的 `estimated_total_salt_molarity_mol_L` **只用于初筛，不是实测浓度**。60 mL 配液瓶的容量不是溶液最终体积。若得到某配方的实测混合液密度 `rho_solution`，可按 `C_actual = 1000 × n_salt × rho_solution / 5` 回算。接近 0.5 M 或 2.5 M 边界的配方尤其需要复核。

纯组分密度及原始来源列在 [`config.py`](config.py) 和 `outputs/components.csv`。LiFSI 采用 [Kishida 的约 2.32 g/mL](https://www.kishida.co.jp/product/catalog/detail/id/19757)；[另一份供应商 SDS 则报告 1.052 g/mL](https://www.carlroth.com/medias/SDB-20XN-GB-EN.pdf?context=bWFzdGVyfHNlY3VyaXR5RGF0YXNoZWV0c3wyNjIxMjV8YXBwbGljYXRpb24vcGRmfGFEQTVMMmhpWkM4NU1UY3lOVEE0TWpFek1qYzRMMU5FUWw4eU1GaE9YMGRDWDBWT0xuQmtaZ3xkZmJkNmM1ZDEzZDcxYzEzMjA0ZjBhMTkxZWJjM2JjODA0YTY2M2ZhODY5ZGUxZWFmYzc5NGQ3ZmFjZGI2ZDc4)，故模型不宜当作精密体积测量。

仅两个 baseline 有可比的文献混合液密度：BAFF（LiDFOB:NDFA = 1:5）约 [1.36 g/mL](https://assets-eu.researchsquare.com/files/rs-5047161/v1/256b3495-d635-4d2f-bcb8-579d0eea3251.pdf)，LHCE（LiFSI:DME:TTE = 1:1.2:3）约 [1.48 g/mL](https://pmc.ncbi.nlm.nih.gov/articles/PMC7936379/)。CSV 只为这两组填写文献密度及回算浓度；其余 14 组保持空值，不以相近配方的密度冒充实测。所有配方仍须验证溶解、分层、黏度及电化学性能。

本仓库是计算设计和称量计划，不是已验证的实验配方数据库。实验前应核对实际试剂瓶签、纯度、含水量与操作 SOP。
