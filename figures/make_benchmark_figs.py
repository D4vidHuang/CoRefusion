import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

BLUE='#0076C2'; BLUE_SH='#00558C'; BLUE_TI='#D1E6F4'
ORANGE='#FF8000'; ORANGE_SH='#B85C00'; ORANGE_TI='#FFE8D1'
INK='#14202B'; MUTED='#5C6B78'; PANEL='#F2F6FA'; LINE='#D9E2EA'
plt.rcParams.update({'font.family':'DejaVu Sans','text.color':INK,'axes.edgecolor':LINE})

groups=[
 ('Diffusion LLM (dLLM)',[
   ('DreamCoder-7B',33.2,66.2,7),('DiffuCoder-7B',31.1,64.1,7),('DreamOn-7B',16.0,55.9,7)]),
 ('Decoder-only',[
   ('CodeLlama-13B',20.2,32.1,13),('CodeLlama-7B',20.2,33.5,7),('StarCoder2-15B',18.6,33.5,15),
   ('Qwen2.5-Coder-7B',18.1,29.3,7),('DeepSeek-Coder-1.3B',17.7,29.2,1.3),('DeepSeek-Coder-6.7B',15.6,24.2,6.7),
   ('Qwen2.5-Coder-3B',14.4,24.9,3),('Qwen2.5-Coder-14B',13.7,22.3,14),('Qwen2.5-Coder-1.5B',13.2,25.0,1.5),
   ('StarCoder2-3B',12.4,30.6,3),('StarCoder2-7B',11.7,29.7,7),('CodeGemma-7B',9.8,18.2,7),('CodeGemma-2B',7.1,13.9,2)]),
 (u'Encoder–decoder',[
   ('CodeT5+ 16B',23.5,26.6,16),('CodeT5+ 6B',21.1,24.0,6),('CodeT5-large',20.6,22.5,0.77),
   ('CodeT5+ 2B',20.5,22.9,2),('CodeT5-base',17.7,19.0,0.22),('CodeT5-small',13.2,15.1,0.06)]),
]

# ================= FIG 1: grouped horizontal bars =================
fig,ax=plt.subplots(figsize=(9.4,10.6))
y=0.0; bh=0.40; off=0.215; ROW=1.0
yt=[]; ytl=[]; headers=[]; bands=[]
for gi,(gname,models) in enumerate(groups):
    y-=0.4; htop=y+0.7; headers.append((y,gname)); y-=1.0
    for (name,em,lj,p) in models:
        ax.barh(y+off,em,height=bh,color=BLUE,zorder=3)
        ax.barh(y-off,lj,height=bh,color=ORANGE,zorder=3)
        ax.text(em+0.7,y+off,f'{em:.1f}',va='center',ha='left',fontsize=8.4,color=BLUE_SH,zorder=4)
        ax.text(lj+0.7,y-off,f'{lj:.1f}',va='center',ha='left',fontsize=8.4,color=ORANGE_SH,zorder=4)
        yt.append(y); ytl.append(name); y-=ROW
    bands.append((htop,y+0.6,gi)); y-=0.3
ymin=y+0.5; ymax=0.6
for (htop,hbot,gi) in bands:
    col=BLUE_TI if gi==0 else (PANEL if gi==2 else 'white')
    ax.axhspan(hbot,htop,color=col,alpha=0.45 if gi==0 else 1.0,zorder=0)
ax.set_yticks(yt); ax.set_yticklabels(ytl,fontsize=9); ax.set_ylim(ymin,ymax)
ax.set_xlim(0,72); ax.set_xticks(range(0,71,10)); ax.tick_params(axis='x',labelsize=9,color=LINE)
ax.set_xlabel('Accuracy (%)',fontsize=11,color=INK)
ax.xaxis.grid(True,color=LINE,lw=0.8,zorder=0); ax.set_axisbelow(True)
for s in ['top','right','left']: ax.spines[s].set_visible(False)
ax.spines['bottom'].set_color(LINE)
for (hy,gname) in headers:
    ax.text(0.2,hy,gname,ha='left',va='center',fontsize=11.5,fontweight='bold',color=INK,zorder=5)
# callout placed in empty right area of decoder group, pointing up to dLLM
ax.annotate('diffusion LLMs lead by a\nwide margin on LJ',
    xy=(64.1,yt[1]-off),xytext=(54,yt[5]),fontsize=9.3,color=BLUE_SH,ha='center',va='center',
    arrowprops=dict(arrowstyle='-|>',color=BLUE_SH,lw=1.4,connectionstyle='arc3,rad=0.28'))
ax.legend(handles=[Patch(fc=BLUE,label='Exact Match (EM)'),Patch(fc=ORANGE,label='LLM-judge (LJ)')],
    loc='lower right',frameon=True,fontsize=10,facecolor='white',edgecolor=LINE)
ax.set_title('Benchmark under the masked condition (RQ1)\nExact Match vs. LLM-judge accuracy, by architecture',
    fontsize=13.5,fontweight='bold',color=INK,pad=12)
fig.tight_layout()
fig.savefig('benchmark_bars.png',dpi=200,facecolor='white',bbox_inches='tight')
fig.savefig('benchmark_bars.pdf',facecolor='white',bbox_inches='tight'); plt.close(fig)

# ================= FIG 2: EM vs LJ scatter =================
fig2,ax2=plt.subplots(figsize=(7.8,6.8))
xlo,xhi,ylo,yhi=5,38,10,70
ax2.fill_between([xlo,xhi],[xlo,xhi],yhi,color=PANEL,zorder=0)        # region LJ>EM
ax2.plot([ylo,xhi],[ylo,xhi],ls='--',color=MUTED,lw=1.2,zorder=1)    # y=x within frame
ax2.text(33.5,35.6,'LJ = EM',color=MUTED,fontsize=9.5,rotation=40,ha='center',va='center',zorder=2)
ax2.text(6.2,67.5,'above line: semantically correct\nbut not an exact match',
    color=MUTED,fontsize=9,ha='left',va='top',style='italic',zorder=2)
sty={'Diffusion LLM (dLLM)':(BLUE,'o'),'Decoder-only':(MUTED,'s'),u'Encoder–decoder':(ORANGE,'^')}
sz=lambda p:40+70*np.sqrt(p)
for gname,models in groups:
    col,mk=sty[gname]
    ax2.scatter([m[1] for m in models],[m[2] for m in models],s=[sz(m[3]) for m in models],
        c=col,marker=mk,alpha=0.80,edgecolors='white',linewidths=1.0,zorder=4,label=gname)

# 'better' direction guide: up-right = higher EM AND higher LJ
ax2.annotate('',xy=(34.0,53.0),xytext=(24.2,40.0),zorder=5,
    arrowprops=dict(arrowstyle='-|>',color=INK,lw=2.6,mutation_scale=24,
        shrinkA=0,shrinkB=0))
ax2.text(34.6,55.0,'better',fontsize=12.5,fontweight='bold',color=INK,ha='left',va='center',zorder=6)
ax2.text(34.6,52.6,'higher EM & LJ',fontsize=8.6,color=MUTED,style='italic',ha='left',va='center',zorder=6)

def lab(name,em,lj,tx,ty,ha):
    ax2.annotate(name,(em,lj),xytext=(tx,ty),fontsize=8.6,color=INK,ha=ha,va='center',zorder=6,
        arrowprops=dict(arrowstyle='-',color=MUTED,lw=0.8,shrinkA=0,shrinkB=4))
lab('DreamCoder-7B',33.2,66.2,29.5,68.5,'right')
lab('DiffuCoder-7B',31.1,64.1,24.0,61.5,'right')
lab('DreamOn-7B',16.0,55.9,19.5,57.8,'left')
lab('StarCoder2-15B',18.6,33.5,13.8,37.0,'right')
lab('CodeLlama-7B/13B',20.2,32.5,24.2,31.0,'left')
lab('CodeT5+ 16B',23.5,26.6,27.0,25.0,'left')
ax2.set_xlabel('Exact Match  EM (%)',fontsize=11.5); ax2.set_ylabel('LLM-judge  LJ (%)',fontsize=11.5)
ax2.set_xlim(xlo,xhi); ax2.set_ylim(ylo,yhi)
ax2.grid(True,color=LINE,lw=0.7); ax2.set_axisbelow(True)
for s in ['top','right']: ax2.spines[s].set_visible(False)
for s in ['left','bottom']: ax2.spines[s].set_color(LINE)
leg=ax2.legend(loc='upper left',frameon=True,fontsize=9.5,facecolor='white',edgecolor=LINE,
    title='Architecture',bbox_to_anchor=(0.012,0.86)); leg.get_title().set_fontsize(9.5)
ax2.text(0.985,0.035,'marker size ∝ parameters',transform=ax2.transAxes,ha='right',va='bottom',
    fontsize=9,color=MUTED,style='italic')
ax2.set_title('Benchmark landscape: LLM-judge vs. Exact Match (RQ1)',fontsize=13.5,fontweight='bold',color=INK,pad=10)
fig2.tight_layout()
fig2.savefig('benchmark_scatter.png',dpi=200,facecolor='white',bbox_inches='tight')
fig2.savefig('benchmark_scatter.pdf',facecolor='white',bbox_inches='tight')
print('done')
