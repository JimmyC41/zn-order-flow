#!/usr/bin/env python3
"""Fixed-clock validation of frozen mean forecasts; no fitting or quoting thresholds.

Input CSV(.gz): day (YYYY-MM-DD), t (seconds since session open), one reconstructed
arrival per row, sorted by day then t. Extra columns are ignored. May-August 2026
only according to frozen config. Output is conditional count MSE/MAE, not PnL.
"""
import argparse
import bisect
import csv
import datetime as dt
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
from export_validation_diagnostic import new_output_directory
from conditional_forecast import from_saved_model, baseline_integral, direct_history_count

ROOT = Path(__file__).resolve().parents[2]
LABELS=('stationary_profile','direct_history_integral','conditional_mean')


def file_sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def days_from_csv(path, first, last, session_end):
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',newline='') as f:
        reader=csv.DictReader(f)
        if not {'day','t'}<=set(reader.fieldnames or []):
            raise ValueError('CSV requires day,t columns')
        previous_day=None
        for day,group in itertools.groupby(reader,key=lambda row:row['day']):
            date=dt.date.fromisoformat(day)
            if day!=date.isoformat():
                raise ValueError('day must use exact YYYY-MM-DD format for calendar matching')
            if not first<=date<=last:
                raise ValueError(f'{day} is outside validation; frozen test/fit data are rejected')
            if previous_day is not None and day<=previous_day:
                raise ValueError('Input days must be strictly ordered and contiguous by group')
            times=[float(row['t']) for row in group]
            if any(not math.isfinite(t) or not 0<=t<session_end for t in times):
                raise ValueError(f'Invalid timestamp on {day}')
            if any(a>b for a,b in zip(times,times[1:])):
                raise ValueError(f'Unsorted events on {day}')
            previous_day=day
            yield day,times


def validate_grid(horizons, delays):
    if not horizons or any(not isinstance(h, int) or h <= 0 for h in horizons):
        raise ValueError('Horizons must be positive whole seconds')
    if not delays or any(not math.isfinite(d) or d <= 0 or d > min(horizons) for d in delays):
        raise ValueError('Delays must be positive and no greater than the shortest horizon')
    if len(set(horizons)) != len(horizons) or len(set(delays)) != len(delays):
        raise ValueError('Duplicate horizons or delays')


def evaluate_day(day,times,model,schedule,horizons,delays):
    validate_grid(horizons, delays)
    rows = []
    end = schedule[-1].end
    if end - min(delays) == end:
        raise ValueError('Delay is too small to separate the observation cutoff from the target')
    for horizon in horizons:
        # Nonoverlapping [h,2h),[2h,3h),...; all delays see the same anchors.
        # Starting at h guarantees a nonnegative cutoff for delay<=h.
        anchors=[float(k*horizon) for k in range(1,int(end//horizon))]
        if not anchors:
            raise ValueError('Horizon leaves no forecast intervals')
        actual=[bisect.bisect_left(times,t+horizon)-bisect.bisect_left(times,t) for t in anchors]
        base=[baseline_integral(schedule,t,t+horizon)/(1-model.n) for t in anchors]
        for delay in delays:
            sums={label:[0.0,0.0,0.0] for label in LABELS}
            states=model.history_states(times,[t-delay for t in anchors])
            for anchor,truth,profile,state in zip(anchors,actual,base,states):
                cut=anchor-delay
                predictions=(profile,
                    direct_history_count(model,state,cut,anchor,anchor+horizon,schedule),
                    model.forecast(state,cut,anchor,anchor+horizon,schedule)['expected_events_in_interval'])
                for label,pred in zip(LABELS,predictions):
                    err=pred-truth
                    sums[label][0] += err * err
                    sums[label][1] += abs(err)
                    sums[label][2] += pred
            for label,(sq,ab,pred) in sums.items():
                rows.append({'day':day,'horizon_s':horizon,'delay_ms':delay*1000,'forecast':label,
                             'n_windows':len(anchors),'mse':sq/len(anchors),'mae':ab/len(anchors),
                             'mean_forecast_count':pred/len(anchors),'mean_observed_count':sum(actual)/len(anchors)})
    return rows


def summarize(rows):
    grouped={}
    for row in rows:
        key=(row['horizon_s'],row['delay_ms'],row['forecast'])
        grouped.setdefault(key,[]).append(row)
    lookup={(r['day'],r['horizon_s'],r['delay_ms']):r for r in rows if r['forecast']=='conditional_mean'}
    result=[]
    for (horizon,delay,label),rr in sorted(grouped.items()):
        gains=[r['mse']-lookup[(r['day'],horizon,delay)]['mse'] for r in rr]
        result.append({'horizon_s':horizon,'delay_ms':delay,'forecast':label,'days':len(rr),
                       'equal_day_mean_mse':statistics.mean(r['mse'] for r in rr),
                       'equal_day_mean_mae':statistics.mean(r['mae'] for r in rr),
                       'paired_daily_mse_gain_of_conditional_mean':statistics.mean(gains),
                       'paired_daily_mse_gain_standard_error':statistics.stdev(gains)/math.sqrt(len(gains)) if len(gains)>1 else None})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--orders',type=Path,required=True)
    p.add_argument('--model',type=Path,default=ROOT/'output/model.json')
    p.add_argument('--config',type=Path,default=ROOT/'analysis/config.json')
    p.add_argument('--news',type=Path,default=ROOT/'data/news_releases.csv')
    p.add_argument('--horizons',default='1,10,600',help='comma-separated integer seconds; default fixed design 1,10,600')
    p.add_argument('--delays-ms',default='1,100,1000',help='comma-separated milliseconds')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--input-label',required=True,choices=['validation_export','synthetic_smoke'],help='Input source recorded in the score provenance')
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Output directory already exists: {args.output}')
    inputs = {'orders': args.orders, 'model': args.model, 'config': args.config, 'news': args.news}
    input_hashes = {key: file_sha256(path) for key, path in inputs.items()}
    config=json.loads(args.config.read_text());saved=json.loads(args.model.read_text())
    horizons=[int(h) for h in args.horizons.split(',')]
    delays=[float(h)/1000 for h in args.delays_ms.split(',')]
    validate_grid(horizons, delays)
    if config.get('fit_end')!='2026-04-30' or config.get('validate_end')!='2026-08-31':
        raise ValueError('This runner is locked to May-August 2026 validation; altered split dates are rejected')
    models={kind:from_saved_model(saved,config,kind) for kind in ['none','data release','fed decision']}
    with args.news.open(newline='') as f:
        kinds={r['date']:('fed decision' if r['release']=='FOMC' else 'data release') for r in csv.DictReader(f) if r['date']}
    first=dt.date.fromisoformat(config['fit_end'])+dt.timedelta(days=1)
    last=dt.date.fromisoformat(config['validate_end'])
    daily=[]
    order_count = 0
    for day,times in days_from_csv(args.orders,first,last,models['none'][1][-1].end):
        order_count += len(times)
        model,schedule=models[kinds.get(day,'none')]
        daily.extend(evaluate_day(day,times,model,schedule,horizons,delays))
        print(f'Completed {day}: {len(times)} supplied reconstructed arrivals',flush=True)
    if not daily:raise ValueError('Empty validation input')
    summary=summarize(daily)
    if any(file_sha256(path) != input_hashes[key] for key, path in inputs.items()):
        raise ValueError('A forecast input changed while scores were being calculated')
    provenance={
        'input_label':args.input_label,'method':'Fixed-clock nonoverlapping windows within each horizon; no event-time conditioning, no model fit, no trading thresholds.',
        'validation_range':[str(first),str(last)],'horizons_s':horizons,'delays_s':delays,
        'stationary_profile_definition':'Integral of frozen Hawkes immigrant baseline divided by 1-sum(alpha); unconditional local-equilibrium profile, not a separate refitted model.',
        'direct_history_integral_definition':'The integral of the immigrant rate and the exponentially decaying excitation from observed orders, excluding excitation from unobserved arrivals.',
        'uncertainty':'Equal-weight paired day differences; conventional day SE assumes independent days and is descriptive if serial dependence is present. Different horizon grids overlap each other; no independent-test claim.',
        'data_boundary':'Arrival times are reconstructed from exchange events. The delay is simulated. The export manifest lists included validation sessions.',
        'n_orders': order_count, 'input_sha256': input_hashes}
    with new_output_directory(args.output) as staged:
        for filename, rows in [('daily_scores.csv', daily), ('summary.csv', summary)]:
            with (staged / filename).open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        provenance['daily_scores_sha256'] = file_sha256(staged / 'daily_scores.csv')
        (staged / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'Wrote {len(daily)} daily score rows to {args.output}')

if __name__=='__main__':main()
