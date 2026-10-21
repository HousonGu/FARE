"""Score only complete same-key predictions; produce paper-ready raw summaries."""
import numpy as np
import pandas as pd
from data_provider.retail import CONFIG, KEYS, checked_merge, check_prediction, metric, json_write

MAIN = {'backbone','soft_history','soft_no_history','soft_history_no_factor',
        'soft_history_no_disagreement','shape_half','factor_half','global_static','history_hard','flat_softmax'}


def families(backbone='all'):
    all_families = ['chronos','lightgbm']+[f'tft_seed_{s}' for s in CONFIG['tft_seeds']]
    return all_families if backbone == 'all' else [f for f in all_families if f == backbone or (backbone == 'tft' and f.startswith('tft_seed_'))]


def score(data,schedule,folder,backbone='all'):
    predictions = {}
    for family in families(backbone):
        frame = pd.read_parquet(folder/f'gates_{family}'/'predictions.parquet')
        expected = MAIN if family=='chronos' else {'backbone','soft_history'}
        if set(frame)-set(KEYS) != expected: raise ValueError(f'Missing/extra methods: {family}')
        predictions[family] = frame
    for family,frame in predictions.items():
        check_prediction(frame)
        if sorted(frame.origin.unique()) != schedule['test']: raise ValueError(f'Wrong schedule: {family}')
    truth = data.truth(schedule['test'])
    # Same subgroup used in the Favorita paper, independent of model predictions.
    subgroup = None
    if data.dataset=='favorita':
        refs = pd.concat([pd.read_parquet(folder/'references'/f'{o}.parquet')[KEYS+['onpromotion','holiday_day']]
                          for o in schedule['test']],ignore_index=True)
        subgroup = checked_merge(truth[KEYS],refs)
        subgroup['selected'] = (subgroup.onpromotion>=6)&(subgroup.holiday_day==1)
        truth = checked_merge(truth,subgroup[KEYS+['selected']])
    overall,weekly,groups = [],[],[]
    for family,frame in predictions.items():
        merged = checked_merge(frame,truth)
        for method in [c for c in frame if c not in KEYS]:
            name = family+'/'+method
            overall.append(dict(method=name,**metric(merged.actual,merged[method])))
            for origin,part in merged.groupby('origin',sort=True):
                weekly.append(dict(origin=origin,method=name,**metric(part.actual,part[method])))
            if subgroup is not None:
                part = merged[merged.selected]
                groups.append(dict(method=name,group='promotion>=6 and planned holiday',**metric(part.actual,part[method])))
    comparisons = []
    for family in families(backbone):
        ours = {r['origin']:r['wape'] for r in weekly if r['method']==family+'/soft_history'}
        controls = [r['method'] for r in overall if r['method'].startswith(family+'/') and r['method']!=family+'/soft_history']
        for control in controls:
            other = {r['origin']:r['wape'] for r in weekly if r['method']==control}
            delta = np.array([ours[o]-other[o] for o in schedule['test']])
            comparisons.append(dict(method=family+'/soft_history',control=control,weeks=len(delta),
                wins=int((delta<0).sum()),ties=int((delta==0).sum()),losses=int((delta>0).sum()),
                mean_weekly_wape_difference=float(delta.mean())))
    seeds = {}
    for method in ['backbone','soft_history']:
        values = [r['wape'] for r in overall if r['method'].startswith('tft_seed_') and r['method'].endswith('/'+method)]
        if values:
            seeds[method] = dict(mean=float(np.mean(values)),std=float(np.std(values,ddof=1)))
    shared = None
    result = dict(overall=overall,weekly=weekly,subgroup=groups,paired_weeks=comparisons,tft_seeds=seeds)
    if shared is not None: result['v3_shared_origin_check'] = shared
    json_write(folder/'metrics.json',result)
    lines = ['# 补充实验：'+folder.name,'','状态：本轮真实预测机器评分；不自动写入论文。','',
        f"数据：{data.dataset}；{schedule['kind']}。",
        f"fit：{schedule['fit'][0]}—{schedule['fit'][-1]}；dev：{schedule['dev'][0]}—{schedule['dev'][-1]}；test：{schedule['test'][0]}—{schedule['test'][-1]}（{len(schedule['test'])} 周，日期均为预测起点）。",'',
        '所有方法严格同键评分；整体 WAPE 按总绝对误差/总销量计算。TFT 均值为种子指标均值，非预测集成。',
        'M5 使用同一拟合／开发协议覆盖完整 52 周；整体指标由全部逐行误差汇总，不平均分段 WAPE。软门控和 softmax 均未按测试结果调参。','']
    for title,rows,leading in [('整体',overall,[]),('Favorita 受控子组',groups,['group']),('逐周',weekly,['origin'])]:
        if not rows: continue
        lines += ['## '+title,'','| '+' | '.join(leading+['方法','行数','WAPE','MAE'])+' |',
                  '| '+' | '.join(['---']*(len(leading)+4))+' |']
        for r in rows:
            lines.append('| '+' | '.join([str(r[k]) for k in leading]+[r['method'],str(r['rows']),f"{r['wape']*100:.6f}%",f"{r['mae']:.6f}"])+' |')
        lines.append('')
    if shared is not None:
        lines += ['## 与 V3 共享起点的逐行对照','','V3 已有的起点上，本轮预测与 V3 预测逐行相减的绝对差。','',
                  '| 方法 | 行数 | 最大绝对差 | 绝对差合计 |','| --- | --- | --- | --- |']
        lines += [f"| {r['method']} | {r['rows']} | {r['max_abs_difference']:.3e} | {r['absolute_difference_sum']:.3e} |" for r in shared]
        lines.append('')
    lines += ['## 逐周配对与种子统计','','详见 metrics.json 的 paired_weeks 与 tft_seeds；负的 WAPE 差表示完整模块更低。','',
        '## 核验与备注','','来源：manifest.json、audit.json、各阶段完成标记及模型拟合记录。',
        '正负结果均保留；不能混入旧表的展示值、调整值或平移估计。逐行预测保留至复核及论文图表完成。','', '作者备注：','']
    (folder/'实验效果.md').write_text('\n'.join(lines),encoding='utf-8')
