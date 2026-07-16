

## Drugbank 

## Hetionet

Entity 缩写：

缩写	含义
C	Compound (drug)
G	Gene (protein)
D	Disease
A	Anatomy
BP	Biological Process
MF	Molecular Function
CC	Cellular Component
PW	Pathway
S	Symptom
SE	Side Effect
PC	Pharmacologic Class


Verb 缩写：
a=associates 
b=binds 
c=causes 
d=down-regulates 
e=expresses 
i=interacts/includes 
l=localizes 
p=participates/presents/palliates 
r=regulates/resembles 
t=treats 
u=up-regulates

i.e. top 20 relation:

GpBP	Gene participates in Biological Process	基因学
AeG	Anatomy expresses Gene（某组织表达某基因）	解剖学
Gr>G	Gene regulates Gene（有向，转录调控）	基因学
GiG	Gene interacts with Gene（PPI 蛋白互作）	基因学
> CcSE	Compound causes Side Effect（drug→副作用）	⭐ PD 候选
AdG	Anatomy down-regulates Gene	解剖学
AuG	Anatomy up-regulates Gene	解剖学
GpMF	Gene participates in Molecular Function	基因学
GpPW	Gene participates in Pathway	基因学
GpCC	Gene participates in Cellular Component	基因学
GcG	Gene covaries with Gene（共表达）	基因学
> CdG	Compound down-regulates Gene（药下调基因表达）	PK-ish
> CuG	Compound up-regulates Gene（药上调）	PK-ish
DaG	Disease associates Gene（疾病-基因关联）	疾病学
> CbG	Compound binds Gene（药物结合的靶蛋白）	⭐ PK 直接靶
DuG	Disease up-regulates Gene	疾病学
DdG	Disease down-regulates Gene	疾病学
> CrC	Compound resembles Compound（结构相似）	化学相似性
DlA	Disease localizes in Anatomy（疾病发生部位）	疾病-解剖
DpS	Disease presents Symptom（疾病-症状）	疾病-症状
PCiC	Pharmacologic Class includes Compound（药理分类）	分类
> CtD	Compound treats Disease（治疗用）	PD 治疗
DrD	Disease resembles Disease	疾病相似
> CpD	Compound palliates Disease（缓解症状）	PD 治疗


## PrimeKG


## Finding1
观察DDInter的描述，似乎“属于”这个部分可以被丢掉？重要的只有“机制”和“表现为”？


### Path type
#### One hop Path-type       
                                                               PK-B%   PD-B%    NEG%
─────────────────────────────────────────────────────────────────────────────────────────────────────
 1  Drug →[het:CcSE]— Side Effect —[het:CcSE]→ Drug                                51.1%   52.3%   52.8%
 2  Drug —[prime:drug_effect]— effect/phenotype —[...]— Drug                       26.0%   30.3%   33.0%
 3  Drug —[prime:contraindication]— disease —[...]— Drug                            3.9%    6.1%    6.6%
 4  Drug —[prime:drug_protein]— gene/protein —[...]— Drug                           5.9%    2.2%    2.1%   ← PK-B+
 5  Drug →[db:enzyme]— Protein —[db:enzyme]→ Drug                                   5.3%    1.3%    2.0%   ← PK-B+
 6  Drug →[het:CbG]— Gene —[het:CbG]→ Drug                                          2.8%    1.4%    0.8%   ← PK-B+
 7  Drug —[prime:indication]— disease —[prime:indication]— Drug                     0.1%    1.9%    0.0%   ← PD-B+
 8  Drug →[db:transporter]— Protein —[db:transporter]→ Drug                         1.4%    0.3%    0.6%   ← PK-B+
 9  Drug →[het:CdG]— Gene —[het:CdG]→ Drug                                          1.2%    0.6%    0.4%
10  Drug →[het:CuG]— Gene —[het:CuG]→ Drug                                          0.8%    0.5%    0.2%
11  Drug —[prime:contraindication]— disease —[prime:indication]— Drug               0.4%    0.5%    0.6%
12  Drug →[db:carrier]— Protein —[db:carrier]→ Drug                                 0.4%    0.3%    0.4%
13  Drug —[het:CrC]— Drug —[het:CrC]— Drug                                          0.1%    0.6%    0.1%   ← PD-B+
14  Drug →[db:target]— Protein —[db:target]→ Drug                                   0.1%    0.4%    0.1%
15  Drug →[het:CdG]— Gene —[het:CuG]→ Drug                                          0.3%    0.2%    0.1%   ← typed mixed
16  Drug —[prime:indication]— disease —[prime:off-label]— Drug                      0.0%    0.3%    0.2%
17  Drug —[prime:off-label]— disease —[prime:off-label]— Drug                       0.0%    0.4%    0.0%
18  Drug —[prime:contraindication]— disease —[prime:off-label]— Drug                0.0%    0.1%    0.1%
19  Drug →[het:CtD]— Disease —[het:CtD]→ Drug                                       0.0%    0.2%    0.0%
20  Drug →[het:CbG]— Gene —[het:CuG]→ Drug                                          0.0%    0.0%    0.0%
21  Drug →[het:CpD]— Disease —[het:CtD]→ Drug                                       0.0%    0.1%    0.0%
22  Drug —[het:CrC]— Drug_outside —[het:CrC]— Drug                                  0.0%    0.1%    0.0%
23  Drug ←[het:PCiC]— Pharmacologic Class —[het:PCiC]← Drug                         0.0%    0.0%    0.0%
24  Drug →[het:CbG]— Gene —[het:CdG]→ Drug                                          0.0%    0.0%    0.0%
25  Drug →[het:CpD]— Disease —[het:CpD]→ Drug                                       0.0%    0.1%    0.0%
26  Drug →[db:pathway]— Pathway —[db:pathway]→ Drug                                 0.0%    0.0%    0.0%

Total paths (denominator):  PK-B=16,710   PD-B=17,097   NEG=11,946
KG coverage:                PK-B 500/500  PD-B 466/500  NEG 926/1000


#### Multiple hop Path-type   