"""清点输注剂量及定量和低于定量限的观测。"""
from .io import unique,positive,finite

def summarize(v):
    if v['units']!={'time':'hour','dose':'mg','concentration':'mg/L'}:raise ValueError('单位未声明或不匹配')
    unique(v['infusions'],'id');unique(v['samples'],'id');dose=0.0;quant=0;censored=0
    for r in v['infusions']:
        start=finite(r['start']);end=finite(r['end'])
        if start<0 or end<=start:raise ValueError('输注时段错误')
        dose+=positive(r['dose'])
    for r in v['samples']:
        if finite(r['time'])<0:raise ValueError('采样时刻错误')
        positive(r['sigma'])
        if r['kind']=='quantified':
            if finite(r['value'])<0:raise ValueError('浓度不能负')
            quant+=1
        elif r['kind']=='below_limit':positive(r['limit']);censored+=1
        else:raise ValueError('未知观测类型')
    return {'total_dose':dose,'quantified':quant,'below_limit':censored,'infusions':len(v['infusions'])}
