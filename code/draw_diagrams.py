"""生成方法框架图和本体结构图"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

# ============================================================
# 图1: 方法总体框架
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

def box(x, y, w, h, text, fc="#E8F0FE", ec="#1F4E79", fontsize=10, fontweight='normal'):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                          linewidth=1.6, edgecolor=ec, facecolor=fc)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=fontweight, wrap=True)

def arrow(x1, y1, x2, y2, color="#444"):
    arr = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle='->', mutation_scale=18,
                          linewidth=1.5, color=color)
    ax.add_patch(arr)

# 五个模块
box(0.3, 5.5, 2.3, 1.6, "Autonomy capability\nrequirement", fc="#FFF2CC", ec="#7F6000", fontsize=10, fontweight='bold')
box(3.2, 5.5, 2.4, 1.6, "LSS-CO\nscenario ontology", fc="#E2F0D9", ec="#385723", fontsize=10, fontweight='bold')
box(6.3, 5.5, 2.4, 1.6, "Functional\nscenarios F1..Fn", fc="#DEEBF7", ec="#1F4E79", fontsize=10, fontweight='bold')
box(9.4, 5.5, 2.4, 1.6, "Logical scenarios\n(parameter ranges)", fc="#DEEBF7", ec="#1F4E79", fontsize=10, fontweight='bold')

box(3.6, 2.5, 6.8, 1.8, "Multi-objective NSGA-II evolution\n[ Test pressure | Phys-feasibility | Diversity ]\nwith ontology repair operator",
    fc="#FBE5D6", ec="#843C0C", fontsize=11, fontweight='bold')

box(0.3, 0.2, 3.2, 1.5, "Concrete scenario set\n(Pareto front)", fc="#FFF2CC", ec="#7F6000", fontsize=10)
box(4.0, 0.2, 3.0, 1.5, "Simulator\n(AirSim + PX4 SITL)", fc="#E8F0FE", ec="#1F4E79", fontsize=10)
box(7.5, 0.2, 3.0, 1.5, "Failure surrogate\nXGBoost regressor", fc="#FFE6E6", ec="#9C0006", fontsize=10)
box(11.0, 0.2, 2.8, 1.5, "Capability\nweak-spot report", fc="#FFF2CC", ec="#7F6000", fontsize=10, fontweight='bold')

# 箭头
arrow(2.6, 6.3, 3.2, 6.3)
arrow(5.6, 6.3, 6.3, 6.3)
arrow(8.7, 6.3, 9.4, 6.3)
arrow(10.6, 5.5, 7.5, 4.3)
arrow(7.0, 2.5, 2.0, 1.7)
arrow(3.5, 0.95, 4.0, 0.95)
arrow(7.0, 0.95, 7.5, 0.95)
arrow(10.5, 0.95, 11.0, 0.95)
# 反馈
arrow(11.0, 1.5, 5.0, 2.5, color="#888")
ax.text(8.0, 2.0, "feedback (update bounds)", fontsize=8.5, color="#666", style='italic')

ax.text(7, 7.7, "Figure 1.  Overall framework of LSS-CO + NSGA-II scenario generation",
        ha='center', fontsize=11, fontweight='bold')

plt.savefig("/home/claude/experiment/figures/fig_framework.png", dpi=180, bbox_inches='tight')
plt.close()
print("已保存 fig_framework.png")

# ============================================================
# 图2: 本体层次结构
# ============================================================
fig, ax = plt.subplots(figsize=(11.5, 6.5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis('off')

# 根节点
box(5.5, 7.5, 3, 1.0, "Scenario (root)", fc="#1F4E79", ec="#0D2240", fontsize=12, fontweight='bold')
ax.texts[-1].set_color('white')

# 五个子域
subs = [
    (0.3, 5.0, 2.4, "Environment\n(E)", "#E2EFDA", "#385723"),
    (3.1, 5.0, 2.4, "Target\n(T)", "#FFF2CC", "#7F6000"),
    (5.8, 5.0, 2.4, "Mission\n(M)", "#DEEBF7", "#1F4E79"),
    (8.5, 5.0, 2.4, "Disturbance\n(D)", "#FBE5D6", "#843C0C"),
    (11.2, 5.0, 2.4, "Capability\n(C)", "#F2D7E0", "#7B2D5B"),
]
for x, y, w, text, fc, ec in subs:
    box(x, y, w, 1.2, text, fc=fc, ec=ec, fontsize=11, fontweight='bold')

# 子域→子类
def sub_children(parent_x, parent_y, children, fc, ec):
    n = len(children)
    spacing = 1.2 / max(n, 1)
    for i, ch in enumerate(children):
        cx = parent_x - 0.4 + i * 1.05
        cy = parent_y - 1.5
        rect = FancyBboxPatch((cx, cy), 0.9, 0.6, boxstyle="round,pad=0.02",
                              linewidth=1.0, edgecolor=ec, facecolor=fc, alpha=0.7)
        ax.add_patch(rect)
        ax.text(cx + 0.45, cy + 0.3, ch, ha='center', va='center',
                fontsize=8.5)
        # 连线
        arr = FancyArrowPatch((parent_x + 1.2, parent_y),
                              (cx + 0.45, cy + 0.6),
                              arrowstyle='-', linewidth=0.8, color="#999")
        ax.add_patch(arr)

sub_children(0.3, 5.0, ["wind", "vis", "illum"], "#E2EFDA", "#385723")
sub_children(3.1, 5.0, ["plat", "rcs", "obs"], "#FFF2CC", "#7F6000")
sub_children(5.8, 5.0, ["type", "alt", "vel"], "#DEEBF7", "#1F4E79")
sub_children(8.5, 5.0, ["gps", "comm", "bat"], "#FBE5D6", "#843C0C")
sub_children(11.2, 5.0, ["perc", "plan", "ctrl"], "#F2D7E0", "#7B2D5B")

# 主从→子域连线
for x, _, _, _, _, _ in subs:
    arr = FancyArrowPatch((7, 7.5), (x + 1.2, 6.2),
                          arrowstyle='->', linewidth=1.2, color="#444")
    ax.add_patch(arr)

# 横向约束关系
ax.annotate("", xy=(11.5, 5.6), xytext=(2.0, 5.6),
            arrowprops=dict(arrowstyle="<->", linestyle='dashed',
                           color="#C0392B", linewidth=1.2))
ax.text(7, 5.9, "ObjectProperty: env constrains cap, mis requires cap, ...",
        ha='center', fontsize=9, color="#C0392B", style='italic')

# 约束公理示例
ax.text(7, 1.8, "Example axioms:\n"
                "  (illum = night) ∧ (cap = visual-avoidance) ⇒ degraded\n"
                "  (plat = fixed-wing) ⊓ ¬(mission = hover)\n"
                "  (em_interference > 0.8) ⇒ (gps_loss > 0.3)",
        ha='center', fontsize=9.5, family='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor="#F5F5F5", edgecolor="#999"))

ax.text(7, 8.6, "Figure 2.  Hierarchical structure of LSS-CO (excerpt; full ontology has 94 classes)",
        ha='center', fontsize=11, fontweight='bold')

plt.savefig("/home/claude/experiment/figures/fig_ontology.png", dpi=180, bbox_inches='tight')
plt.close()
print("已保存 fig_ontology.png")
