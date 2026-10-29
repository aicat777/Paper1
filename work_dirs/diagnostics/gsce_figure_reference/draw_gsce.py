"""Code-faithful GSCE module diagram, with the box/arrow style of LC Fig. 2."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle, Rectangle

OUT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'svg.fonttype': 'none', 'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 3, figsize=(17, 8.8), gridspec_kw={'width_ratios': [1, 1.15, 1.35]})
orange = '#fff0d9'
orange_edge = '#d88b16'
blue = '#e2edf6'
ink = '#252b32'


def text(ax, x, y, value, size=10, **kw):
    ax.text(x, y, value, ha='center', va='center', fontsize=size, color=ink, **kw)


def box(ax, x, y, w, h, label, changed=False, baseline=False, dashed=False, size=10):
    patch = FancyBboxPatch((x-w/2, y-h/2), w, h,
        boxstyle='round,pad=0.012,rounding_size=0.035',
        linewidth=1.35, edgecolor=orange_edge if changed else '#56616c',
        facecolor=orange if changed else blue if baseline else 'white',
        linestyle='--' if dashed else '-', zorder=3)
    ax.add_patch(patch)
    text(ax,x,y,label,size,zorder=4)


def arrow(ax, start, end, dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=12,
                 linewidth=1.25, color=ink, linestyle='--' if dashed else '-',
                 shrinkA=0, shrinkB=0, zorder=2))


def path(ax, points, dashed=False, tip=True):
    xs,ys=zip(*points)
    ax.plot(xs,ys,color=ink,linewidth=1.25,linestyle='--' if dashed else '-',zorder=2)
    if tip:
        arrow(ax,points[-2],points[-1],dashed)


for ax in axes:
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 10)
    ax.set_aspect('auto')
    ax.axis('off')
    ax.add_patch(Rectangle((.03,.18),4.94,9.48,fill=False,edgecolor='#a0a7af',linewidth=1.1))

# (a) The stride-2 operation. Both convolutions replace the original dense 3x3.
ax=axes[0]
text(ax,2.5,9.26,'(a) Downsampling unit',12,weight='bold')
text(ax,2.5,8.88,'4 selected transitions',10)
text(ax,2.5,8.14,r'Input: $C_{in}\times H\times W$',11)
arrow(ax,(2.5,7.84),(2.5,7.23))
box(ax,2.5,6.75,3.15,.96,'1×1 PWConv\nBN + SiLU',changed=True,size=12)
text(ax,3.47,5.66,'$C_{in}\\rightarrow C_{out}$\nstride = 1',9.5)
arrow(ax,(2.5,6.25),(2.5,4.92))
box(ax,2.5,4.38,3.15,1.06,'3×3 DWConv\nstride = 2; BN',changed=True,size=12)
text(ax,3.55,3.45,r'groups = $C_{out}$',9.5)
arrow(ax,(2.5,3.84),(2.5,2.73))
text(ax,2.5,2.38,r'Output: $C_{out}\times\frac{H}{2}\times\frac{W}{2}$',12)
text(ax,2.5,1.38,'Channel transformation first;\nspatial reduction second.',10,linespacing=1.5)
text(ax,2.5,.65,'No activation after the final BN.',9)

# (b) One CSPNeXt bottleneck; the shortcut is present only at P3/P4 in backbone.
ax=axes[1]
text(ax,2.5,9.26,'(b) Grouped bottleneck',12,weight='bold')
text(ax,2.5,8.88,'Inside the 7 selected CSP units',10)
text(ax,2.35,8.12,r'Input: $C_h\times H\times W$',11)
ax.plot(2.35,7.66,'o',color=ink,markersize=3,zorder=4)
arrow(ax,(2.35,7.84),(2.35,7.18))
box(ax,2.35,6.72,2.85,.90,'3×3 GConv, G = 4\nBN + SiLU',changed=True,size=11)
arrow(ax,(2.35,6.26),(2.35,5.68))
box(ax,2.35,5.23,2.85,.86,'5×5 DWConv\nBN + SiLU',size=11)
arrow(ax,(2.35,4.79),(2.35,4.19))
box(ax,2.35,3.74,2.85,.86,'1×1 PWConv\nBN + SiLU',size=11)
arrow(ax,(2.35,3.30),(2.35,2.57))
ax.add_patch(Circle((2.35,2.35),.21,edgecolor=ink,facecolor='white',lw=1.2,zorder=3))
text(ax,2.35,2.35,'+',15,zorder=4)
path(ax,[(2.35,7.66),(.43,7.66),(.43,2.35),(2.13,2.35)],dashed=True)
text(ax,.66,4.9,'Identity shortcut',9,rotation=90)
arrow(ax,(2.35,2.13),(2.35,1.70))
text(ax,2.35,1.39,r'Output: $C_h\times H\times W$',11)
text(ax,2.5,.67,'Dashed shortcut and +: backbone P3/P4 only.\nP5 and neck: direct output of the last PWConv.',8.7,linespacing=1.55)

# (c) The enclosing CSPLayer. Both branches project the full input independently.
ax=axes[2]
text(ax,2.5,9.26,'(c) Enclosing CSP unit',12,weight='bold')
text(ax,2.5,8.88,'Original outer structure is retained',10)
text(ax,2.5,8.24,r'Input: $C_{in}\times H\times W$',11)
arrow(ax,(2.5,7.98),(2.5,7.67))
ax.plot(2.5,7.67,'o',color=ink,markersize=3,zorder=4)
path(ax,[(2.5,7.67),(1.35,7.67),(1.35,7.30)])
path(ax,[(2.5,7.67),(3.78,7.67),(3.78,7.30)])
box(ax,1.35,6.9,1.98,.78,'1×1 Conv\nBN + SiLU',baseline=True)
box(ax,3.78,6.9,1.85,.78,'1×1 Conv\nBN + SiLU',baseline=True)
text(ax,1.35,6.19,r'$C_{in}\rightarrow C_h$',9.5)
text(ax,3.78,6.19,r'$C_{in}\rightarrow C_h$',9.5)
arrow(ax,(1.35,6.50),(1.35,5.73))
box(ax,1.35,5.22,1.98,1.0,'Grouped\nbottleneck (b)',baseline=True,size=11)
arrow(ax,(1.35,4.71),(1.35,4.12))
path(ax,[(3.78,6.50),(3.78,4.34),(3.05,4.34),(3.05,4.12)])
box(ax,2.20,3.72,3.30,.78,'Concat: 2Cₕ channels',size=11)
arrow(ax,(2.20,3.32),(2.20,2.93))
box(ax,2.20,2.54,3.30,.76,'Existing channel attention*',baseline=True,dashed=True,size=10)
arrow(ax,(2.20,2.15),(2.20,1.85))
box(ax,2.20,1.49,3.30,.70,'1×1 Conv + BN + SiLU',baseline=True,size=10)
arrow(ax,(2.20,1.13),(2.20,.88))
text(ax,2.20,.63,r'Output: $C_{out}\times H\times W$',10)

fig.subplots_adjust(left=.025,right=.975,top=.94,bottom=.12,wspace=.075)
fig.text(.5,.975,'GSCE: Grouped Spatial–Channel Encoding',ha='center',va='center',fontsize=16,weight='bold')
fig.text(.5,.075,'Orange = replaced operators. Blue = retained outer organization; white = retained operators. PW = pointwise; DW = depthwise; GConv = grouped convolution.',
         ha='center',fontsize=10,color=ink)
fig.text(.5,.045,'* Channel attention in (c): backbone only; omitted in neck.  Cₕ = C_out / 2.  The two CSP branches are learned projections, not a raw channel split.',
         ha='center',fontsize=9.5,color=ink)
fig.text(.5,.017,'Panels (a) and (c) are placed at different network locations; they are not a universal serial chain. All arrows follow the current implementation.',
         ha='center',fontsize=9.5,color=ink)
for extension in ('svg','pdf','png'):
    target=OUT/f'gsce_structure_draft.{extension}'
    fig.savefig(target,dpi=150,facecolor='white')
    print(target)
plt.close(fig)
