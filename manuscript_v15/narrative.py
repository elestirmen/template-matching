# -*- coding: utf-8 -*-
"""Results-dependent paragraphs of v15, built from the numbers of fill_extra.py (strings, one decimal).
Comparative wording is derived from the numbers, so the text cannot contradict the tables. Keyboard hyphens only."""
import math

CAES = ('m4', 'f48', 'c1550')
NAME = {'m4': 'Model 4 lineage', 'f48': 'f48-k4-s1 checkpoint 1850', 'c1550': 'f48-k4-s1 checkpoint 1550'}


def _f(N, k):
    try:
        return float(str(N.get(k + '_x', N[k])).replace(',', ''))
    except (KeyError, ValueError):
        return float('nan')


def _signed(x):
    return f'{x:+.1f}'.replace('+', '+').replace('-', '-')


def build(N):
    f = lambda k: _f(N, k)
    g = lambda k: N.get(k, '?')
    E = {}
    ok = lambda v: not math.isnan(v)

    def rng(mode, suffix=''):
        v = [f(f'{mode}_{c}{suffix}') for c in CAES]
        v = [x for x in v if ok(x)]
        return (min(v), max(v)) if v else (float('nan'), float('nan'))

    or_lo, or_hi = rng('or'); ca_lo, ca_hi = rng('ca'); cap_lo, cap_hi = rng('ca', '_prec')
    best_ca = max(CAES, key=lambda c: f(f'ca_{c}') if ok(f(f'ca_{c}')) else -1)
    worst_ca = min(CAES, key=lambda c: f(f'ca_{c}') if ok(f(f'ca_{c}')) else 1e9)
    N['ca_range'] = f'{ca_lo:.1f}-{ca_hi:.1f}'; N['or_range'] = f'{or_lo:.1f}-{or_hi:.1f}'
    N['ca_prec_range'] = f'{cap_lo:.1f}-{cap_hi:.1f}'
    above_gray_ca = [c for c in CAES if f(f'ca_{c}') > f('ca_gray') + 1.5]
    above_gray_or = [c for c in CAES if f(f'or_{c}') > f('or_gray') + 1.5]
    E['_summary'] = dict(or_range=N['or_range'], ca_range=N['ca_range'], best_ca=best_ca, worst_ca=worst_ca,
                         above_gray_ca=above_gray_ca, above_gray_or=above_gray_or)

    # ---------------------------------------------------------------- 4.2 preprocessing sensitivity
    d_m4_ca = f('eq1_delta_ca_m4'); d_f48_ca = f('eq1_delta_ca_f48')
    d_m4_or = f('eq1_delta_or_m4'); d_f48_or = f('eq1_delta_or_f48')
    E['preproc_par'] = (
        "Input preprocessing. The training scripts of both cross-autoencoder lineages equalized the histogram of every "
        "input tile, whereas the flight software of the doctoral study and the benchmark of Section 3.8 feed the "
        "networks without equalization; the replay protocol follows the latter and was fixed before the replay. With "
        "PIL-style equalization applied to map and frame tiles alike, the oracle success on the doctoral routes " + (
            "" if all(N.get(f'eq1_routes_{k}', '11') == '11' for k in ('or_m4', 'or_f48', 'ca_m4', 'ca_f48'))
            else "(compared on the routes replayed in both settings) ") + "changes "
        f"by {_signed(d_m4_or)} points for the Model 4 lineage and by {_signed(d_f48_or)} points for checkpoint 1850, and "
        f"the causal success by {_signed(d_m4_ca)} and {_signed(d_f48_ca)} points. The preprocessing is therefore part of "
        "the method: it must be fixed together with the checkpoint, and the setting used to select a checkpoint is the "
        "one to use in operation."
    )

    # ---------------------------------------------------------------- 4.2 failure analysis
    E['failure_par'] = (
        "Three sources account for most of the remaining errors and failures. First, a constant camera offset: in the "
        "body frame of the vehicle the accepted fixes of the causal replay are displaced by about "
        f"{g('b_l1d_f_m4')} m along and {g('b_l1d_r_m4')} m across the heading with the Mavic 2 Pro and by about "
        f"{g('b_fc_f_m4')} m and {g('b_fc_r_m4')} m with the Mavic 2 Zoom (positive: forward and to the right; Model 4 "
        f"lineage, grayscale: {g('b_l1d_f_gray')}/{g('b_l1d_r_gray')} m and {g('b_fc_f_gray')}/{g('b_fc_r_gray')} m). The "
        "offset is the same on every route and with every representation, which points to a small attitude (boresight) "
        "offset of the camera or a timing offset of the position tag rather than to the matcher. Estimating the offset "
        "of each route from the other doctoral routes flown with the same camera and subtracting it raises the "
        f"precision of the accepted fixes of the Model 4 lineage from {g('p_before_m4')}% to {g('p_after_m4')}% on the "
        f"doctoral routes ({g('pa_before_m4')}% to {g('pa_after_m4')}% on the additional flights) and lowers their median "
        f"error from {g('med_before_m4')} m to {g('med_after_m4')} m. The offset of the Mavic 2 Zoom approaches the 30 m "
        "threshold, which explains the low success of Urban 1 in Table 4; the tables report uncorrected values. Second, "
        "heading: the flight-controller yaw used for north alignment differs from the gimbal yaw by 2-13° depending on "
        "the route, and neither source is consistently better (on a subset of frames the gimbal yaw would have raised "
        "the oracle success of Model 4 on Urban 8 and lowered it on Rural 1). Heading is also the dominant sensitivity "
        f"(Table 5): a heading error of 5° lowers the causal success of the Model 4 lineage from {g('ca_m4')}% to "
        f"{g('sens_yaw+5')}% or {g('sens_yaw-5')}% and an error of 10° to {g('sens_yaw+10')}% or {g('sens_yaw-10')}%, "
        f"whereas height errors of 5% leave it at {g('sens_alt+5pct')}% and {g('sens_alt-5pct')}%. Third, the coupling of position and scale in the causal replay: the "
        "height above ground is computed from the DEM under the predicted position. After a run of unresolved frames "
        "over homogeneous fields the predicted position drifts onto higher ground, the frame is scaled about 20% too "
        "small, often below the size of a crop, and the tracker cannot re-acquire the route. On Rural 1 and Rural 2 "
        "this accounts for most of the gap between the causal and the GNSS-aided replay (Fig. 6(b)); on Rural 3, where "
        "the track is kept, the two replays scale the frames within about 3% of each other. Holding the ground height of "
        "the last accepted fix while the position is uncertain would break this loop."
    )

    E['fig6_caption'] = (
        "Fig. 6. Causal replay (GNSS read only at the first frame) of two routes with the Model 4 lineage. Green: GPS "
        "track; yellow: accepted fixes within 30 m; red: accepted fixes beyond 30 m. (a) Urban 5 (Mavic 2 Pro): "
        f"{g('fig6_u5')} of the eligible frames are localized within 30 m. (b) Rural 1: after a run of unresolved frames over "
        "homogeneous fields the predicted position drifts, the frames are scaled too small and the track is not "
        "re-acquired (Section 4.2). Basemap: Google Maps (a) and Bing Maps (b) satellite imagery, © Google, © Microsoft; imagery © Maxar Technologies."
    )
    E['sens_note'] = (
        "Each row is a complete causal replay of the eleven doctoral routes with the perturbation applied to every frame "
        "(heading, height above ground) or to the initialization only (initial position, displaced to the north-east). "
        "The eligible frames are those of Table 3."
    )

    # ---------------------------------------------------------------- texture probe (Section 4.4)
    low = [f('low_gray'), f('low_sobel'), f('low_cae')]; reg = [f('reg_gray'), f('reg_sobel'), f('reg_cae')]
    parts = []
    if all(ok(l) and ok(r) and l >= r - 3.0 for l, r in zip(low, reg)):
        parts.append("Within a ±1 km window the low-texture tiles are not harder than the regular ones for any "
                     "representation, so in local search the gate does not remove queries that would change the comparison; its "
                     "effect on the global search was not measured, and the Google stratum is small.")
    elif ok(low[2]) and low[2] >= max(low[0], low[1]):
        parts.append("The learned representation keeps its advantage over grayscale on the tiles that the gate excludes.")
    else:
        parts.append("On these tiles the representations lose part of their accuracy, which the gate makes explicit.")
    if ok(reg[1]) and ok(reg[2]) and reg[1] > reg[2] + 1.5:
        parts.append("In this local search, as in the global benchmark of Section 4.3, the gradient-magnitude baseline "
                     f"is more accurate than the cross-autoencoder on regular tiles ({reg[1]:.1f}% against "
                     f"{reg[2]:.1f}%), and both are well above grayscale.")
    ts = ' '.join(parts)
    E['texture_par'] = (
        "Low-texture queries. The benchmark gate rejects tiles with a gray-level standard deviation below 12 or an "
        "entropy below 4 bits. To measure performance exactly where the gate applies, a regular 500 m grid over the "
        f"district mosaics was classified with the same gate, and all low-texture grid points ({g('n_low_g')} on the "
        f"Google and {g('n_low_b')} on the Bing mosaic, excluding missing-data tiles) were matched together with "
        f"{g('n_reg')} randomly drawn regular points per direction, with the flight-replay matcher in a ±1 km window "
        "around the true position (25 m threshold, clean tiles, f48-k4-s1 checkpoint 1850). On regular tiles grayscale "
        f"NCC, gradient magnitude and the cross-autoencoder reach {g('reg_gray')}%, {g('reg_sobel')}% and "
        f"{g('reg_cae')}% (mean of both directions); on low-texture tiles they reach {g('low_gray')}%, {g('low_sobel')}% "
        f"and {g('low_cae')}%. " + ts
    )

    # ---------------------------------------------------------------- reconstruction control (Section 4.5)
    rm, rr, rw = f('recon_map'), f('recon_rec'), f('recon_raw')
    recon_ok = ok(rr) and rr <= rw + 1.5
    if ok(rr):
        if rr < rw - 1.5:
            rs = ("The reconstruction network, which learns an almost exact copy of its input (validation SSIM 0.999), is "
                  "worse than untransformed grayscale, and none of its checkpoints reached the false-peak ratio of grayscale on "
                  "the training region: in this pair of runs the gain does not come from passing the tiles through a "
                  "convolutional encoder-decoder but from the cartographic target.")
        elif rr <= rw + 1.5:
            rs = ("The reconstruction network, which learns an almost exact copy of its input (validation SSIM 0.999), "
                  "performs like untransformed grayscale: the gain does not come from passing the tiles through a "
                  "convolutional encoder-decoder but from the cartographic target.")
        else:
            rs = (f"The reconstruction network also improves on grayscale, but the map-supervised network gains "
                  f"{rm - rw:.1f} points against {rr - rw:.1f}, so most of the gain is due to the cartographic target.")
    else:
        rs = ''
    E['recon_concl'] = (('matters: in one controlled pair of runs, the same network trained to reconstruct its input '
                         + ('fell below' if ok(rr) and rr < rw - 1.5 else 'performed like') + ' grayscale') if recon_ok
                        else 'contributes to the gain')
    E['recon_par'] = (
        "Cartographic supervision versus input reconstruction. To isolate the training target, the training recipe of "
        "the authors' current pipeline (seven-layer f32-k3 encoder-decoder without spatial reduction, sigmoid output, "
        "binary cross-entropy, batch 1, 12 epochs) was trained twice with the target as the only difference: with the "
        "cartographic rendering, and - in a later run with the same code, configuration and training tiles - with the "
        "input satellite tile itself (an autoencoder that reconstructs its input). For each run the checkpoint with the lowest false-peak ratio on a training region was chosen "
        "(the gate of Section 4.4) and "
        "evaluated with 1,000 queries per direction on the district mosaics. Global success at 25 m (mean of both "
        f"directions and both query variants) is {g('recon_map')}% for the map-supervised network, {g('recon_rec')}% "
        f"for the reconstruction network and {g('recon_raw')}% for grayscale NCC on the same queries. " + rs
    )

    # ---------------------------------------------------------------- discussion
    cae_ho_or = max(f(f'or_{c}_ho') for c in CAES); cae_ho_ca = max(f(f'ca_{c}_ho') for c in CAES)
    if f('or_sobel_ho') > cae_ho_or + 1.5 and f('ca_sobel_ho') >= cae_ho_ca - 2.5:
        sob_ho = 'matches or exceeds every checkpoint'
    elif f('or_sobel_ho') >= cae_ho_or - 2.5 and f('ca_sobel_ho') >= cae_ho_ca - 2.5:
        sob_ho = 'is on a par with the checkpoints'
    else:
        sob_ho = 'stays below the best checkpoint'
    d1850 = [f('or_f48') - f('or_gray'), f('ca_f48') - f('ca_gray')]
    f1850 = ('performed at the level of grayscale on the flights' if all(abs(x) <= 1.5 for x in d1850) else
             ('was below grayscale on the flights' if all(x < -1.5 for x in d1850) else
              ('was better than grayscale on the flights' if all(x > 1.5 for x in d1850) else
               'was not consistently better than grayscale on the flights')))
    # ---- gradient magnitude on the district benchmark (sobel_summary.json via fill_extra.py --sobel-bench)
    have_sb = all(ok(f(k)) for k in ('sb_cilo_cae_c', 'sb_cilo_cae_h', 'sb_sob_c', 'sb_cae_c', 'sb_gray_c'))
    if not have_sb:
        raise SystemExit('narrative.py: the benchmark result of gradient magnitude is missing (make_v15.py --sobel-bench)')
    if not (f('sb_cilo_cae_c') > 0 and f('sb_cilo_cae_h') > 0):
        raise SystemExit('narrative.py: the comparative wording assumes gradient magnitude above the cross-autoencoder on '
                         'both query variants (block-bootstrap interval above zero); revise the text for this result')
    every_ckpt = f('sb_sob_c') > 69.4 and f('sb_sob_h') > 62.7   # district maxima of the checkpoint sweep (Table 7)
    sb_short_c, sb_short_h = abs(f('sb_d_cae_c')), abs(f('sb_d_cae_h'))

    def f1(k):  # the value as printed in the tables (one decimal), so that stated differences match the tables
        try: return float(str(N[k]).replace(',', ''))
        except (KeyError, ValueError): return float('nan')

    def vs(a, b, obj=''):
        d_ = f1(a) - f1(b)
        o = f' {obj}' if obj else ''
        if abs(d_) < 0.05: return f'matches{o}'
        return f'{"leads" if d_ > 0 else "trails"}{o} by {abs(d_):.1f} points'

    # ---- paired block-bootstrap intervals of the flight differences (flight_ci.py via fill_extra.py --flight-ci)
    def fd(mode, a_, b_, ho=False):
        base = f'fd_{mode}_{a_}_{b_}' + ('_ho' if ho else '')
        if base + '_lo_25' not in N: return None
        return {s_: float(N[f'{base}_{s_}']) for s_ in ('diff_pp', 'lo_25', 'hi_25', 'lo_10', 'hi_10', 'lo_50', 'hi_50',
                                                        'lo_route', 'hi_route')}

    def sgn(mode, a_, b_, ho=False):  # sign of the stated (printed, one decimal) difference a - b
        s_ = '_ho' if ho else ''
        return 1 if f1(f'{mode}_{a_}{s_}') - f1(f'{mode}_{b_}{s_}') >= 0 else -1

    def oriented(mode, a_, b_, L='25', ho=False):  # interval of the stated lead or deficit, in points
        x = fd(mode, a_, b_, ho); lo, hi = x[f'lo_{L}'], x[f'hi_{L}']
        return (lo, hi) if sgn(mode, a_, b_, ho) > 0 else (-hi, -lo)

    def ci(mode, a_, b_, L='25', ho=False):
        lo, hi = oriented(mode, a_, b_, L, ho)
        return f'{lo:.1f} to {hi:.1f}'

    def excl(mode, a_, b_, L='25', ho=False):
        lo, hi = oriented(mode, a_, b_, L, ho)
        return lo > 0 or hi < 0

    def join(xs, conj='and'):
        return xs[0] if len(xs) == 1 else ', '.join(xs[:-1]) + f' {conj} ' + xs[-1]

    WHO = {'m4': 'the Model 4 lineage', 'f48': 'checkpoint 1850', 'c1550': 'checkpoint 1550',
           'sobel': 'gradient magnitude', 'gray': 'grayscale'}
    MODEW = {'or': 'oracle', 'ca': 'causal'}

    def lead(mode, a_, b_, ho=False):
        return 'lead' if sgn(mode, a_, b_, ho) > 0 else 'deficit'

    def phrase(mode, a_, b_, route=False):
        return f"the {MODEW[mode]} {lead(mode, a_, b_)} of {WHO[a_]} ({ci(mode, a_, b_, 'route' if route else '25')} points)"

    DISC = [('or', 'm4', 'sobel'), ('ca', 'm4', 'sobel'), ('or', 'f48', 'sobel'), ('ca', 'f48', 'sobel')]
    REPORTED = DISC + [('or', 'm4', 'gray'), ('or', 'sobel', 'gray'), ('ca', 'c1550', 'sobel'), ('ca', 'sobel', 'gray')]
    if not all(fd(*k) is not None for k in REPORTED):
        raise SystemExit('narrative.py: paired flight intervals missing (flight_ci.json; make_v15.py runs flight_ci.py)')
    ho_ck = [c for c in CAES if fd('or', c, 'sobel', True) is not None]
    ho_lead = [f1('or_sobel_ho') - f1(f'or_{c}_ho') for c in ho_ck]
    ho_all = bool(ho_ck) and all(d_ > 0 for d_ in ho_lead) and all(excl('or', c, 'sobel', ho=True) for c in ho_ck)
    E['ci_oracle'] = (
        "Paired on the same frames, with 95% intervals from a block bootstrap stratified by route (Section 3.7), the "
        f"Model 4 lineage {vs('or_m4', 'or_sobel', 'gradient magnitude')} ({ci('or', 'm4', 'sobel')}) and "
        f"{vs('or_m4', 'or_gray', 'grayscale')} ({ci('or', 'm4', 'gray')}), gradient magnitude "
        f"{vs('or_sobel', 'or_gray', 'grayscale')} ({ci('or', 'sobel', 'gray')}) and checkpoint 1850 "
        f"{vs('or_f48', 'or_sobel', 'gradient magnitude')} ({ci('or', 'f48', 'sobel')})" +
        (f"; on the additional flights gradient magnitude leads the three checkpoints by {min(ho_lead):.1f}-"
         f"{max(ho_lead):.1f} points, with intervals that exclude zero for all three." if ho_all else
         f"; on the additional flights the differences between gradient magnitude and the checkpoints range from "
         f"{min(ho_lead):.1f} to {max(ho_lead):.1f} points.")
    )
    flips = [k for k in REPORTED if len({excl(*k, L) for L in ('10', '25', '50')}) > 1]
    if flips:
        parts = []
        for m_, a_, b_ in flips:
            los = [oriented(m_, a_, b_, L)[0] for L in ('10', '25', '50')]
            rel_ = 'over' if lead(m_, a_, b_) == 'lead' else 'relative to'
            parts.append(f"the {MODEW[m_]} {lead(m_, a_, b_)} of {WHO[a_]} {rel_} {WHO[b_]}, whose lower limit lies "
                         f"between {min(los):.1f} and {max(los):.1f} points")
        sens = (" For the paired comparisons of both replays, blocks of 10 or 50 frames lead to the same conclusions "
                f"except for {join(parts)}.")
    else:
        sens = " For the paired comparisons of both replays, blocks of 10 or 50 frames lead to the same conclusions."
    E['ci_causal'] = (
        f"Paired on the same frames (95% intervals in parentheses), the Model 4 lineage {vs('ca_m4', 'ca_sobel', 'gradient magnitude')} "
        f"({ci('ca', 'm4', 'sobel')}), checkpoint 1850 {vs('ca_f48', 'ca_sobel', 'it')} ({ci('ca', 'f48', 'sobel')}) and "
        f"checkpoint 1550 {vs('ca_c1550', 'ca_sobel', 'it')} ({ci('ca', 'c1550', 'sobel')}), and gradient magnitude "
        f"{vs('ca_sobel', 'ca_gray', 'grayscale')} ({ci('ca', 'sobel', 'gray')})." + sens
    )
    ex = [phrase(*k) for k in DISC if excl(*k)]
    inc = [phrase(*k) for k in DISC if not excl(*k)]
    rex = [phrase(*k, route=True) for k in DISC if excl(*k, 'route')]
    ci_disc = (" The paired intervals of Sections 4.1 and 4.2 separate " +
               (f"{join(ex)} from zero, but not {join(inc, 'or')}" if ex and inc else
                ("all four differences from zero" if ex else "none of these differences from zero")) +
               "; when whole routes are resampled, a more conservative check with eleven routes, " +
               (f"only {join(rex)} {'remains' if len(rex) == 1 else 'remain'}" if rex else "none of them remains") + ".")
    small_txt = (f"{g('cov_small')} are too small at map resolution to contain a crop" +
                 (f" ({g('cov_small_r4')} of them the entire low-altitude Rural 4 flight)"
                  if g('cov_small_r4') == g('cov_r4_cap') and g('cov_small_r4') != '0' else ""))
    dem_txt = (f"{g('cov_dem')} frames of Rural 5 lie outside the DEM" if g('cov_dem') == g('cov_dem_r5')
               else f"{g('cov_dem')} frames lie outside the DEM")
    E['coverage'] = (
        f"Of the {g('cov_cap')} captured frames, {g('cov_elig')} are eligible ({g('cov_elig_doc')} on the eleven doctoral "
        f"routes and {g('cov_elig_ho')} on the additional flights); of the {g('cov_inel')} that are not, {small_txt} and "
        f"{dem_txt}" + ("." if g('cov_other') == '0' else f", and {g('cov_other')} fail for other reasons.")
    )
    E['ci_method'] = (
        "Differences between representations are computed on the same eligible frames (paired). Because consecutive "
        "frames overlap and a lost track persists over many frames, their 95% intervals come from a circular block "
        f"bootstrap stratified by route (blocks of {g('fci_block')} consecutive frames, {g('fci_reps')} replicates); "
        "blocks of 10 and 50 frames and resampling of whole routes serve as sensitivity checks."
    )

    flight_diffs = [abs(f1(f'{m}_{c}') - f1(f'{m}_sobel')) for m in ('or', 'ca') for c in CAES]
    within = max(flight_diffs) <= max(or_hi - or_lo, ca_hi - ca_lo)
    E['contrib_baselines'] = (
        "Controlled comparisons with the identical matcher, on the same flight frames and on the same benchmark "
        "queries, between the learned representation, untransformed grayscale and a classical gradient-magnitude "
        "representation, and between the thesis-selected stride-2 checkpoint and the benchmark-selected stride-1 "
        "checkpoint. The outcome is unfavorable to the learned representation: on the benchmark gradient magnitude "
        "exceeds " + ("every checkpoint" if every_ckpt else "the benchmark-selected checkpoint") + ", and on the flights "
        + ("the differences between representations are within the spread between checkpoints."
           if within else "the differences between representations are small and checkpoint-dependent.")
    )
    E['bench_sobel_par'] = (
        "Gradient-magnitude baseline. The classical gradient-magnitude representation of the flight replay (Section "
        "3.8), run through the district benchmark after the first round of review in global search on the identical "
        f"{g('sb_n')} queries per direction, localizes {g('sb_sob_c')}% of clean and {g('sb_sob_h')}% of degraded "
        f"queries within 25 m without any training (mean of both directions), against {g('sb_cae_c')}% and "
        f"{g('sb_cae_h')}% for the benchmark-selected cross-autoencoder and {g('sb_gray_c')}% and {g('sb_gray_h')}% for "
        "grayscale NCC; the grayscale run repeated alongside reproduced the grayscale rows of Table 6 query by query. "
        f"The difference to the cross-autoencoder, computed query by query, is {g('sb_d_cae_c')} points on clean "
        f"queries (95% block-bootstrap interval {g('sb_ci_cae_c')}) and {g('sb_d_cae_h')} points on degraded queries "
        f"({g('sb_ci_cae_h')}): gradient magnitude succeeds on {g('sb_only_c')} clean queries that the network misses "
        f"and fails on {g('sb_caeonly_c')} that the network solves. It leads in both matching directions "
        f"({g('sb_sob_gb_c')}% and {g('sb_sob_bg_c')}% on clean queries against {g('sb_cae_gb_c')}% and "
        f"{g('sb_cae_bg_c')}% for Google → Bing and Bing → Google), on both halves of the spatial split of Section 4.4 "
        f"(north {g('sb_n_sob_c')}% against {g('sb_n_cae_c')}%, south {g('sb_s_sob_c')}% against {g('sb_s_cae_c')}%) "
        f"and in every tertile of query entropy, with the largest margin on the least textured third "
        f"({g('sb_lo_sob_c')}% against {g('sb_lo_cae_c')}%)" +
        (", and it is above every checkpoint evaluated on the district mosaics (Table 7)" if every_ckpt else "") +
        f". It loses {g('sb_drop_sob')} points under photometric degradation where the network loses "
        f"{g('sb_drop_cae')}. Successful fixes are equally precise (median error {g('sb_med_sob')} m against "
        f"{g('sb_med_cae')} m), and {g('sb_rej')} of the {g('sb_searches')} gradient-magnitude searches were rejected "
        "for lack of template variance and count as failures. The smoothing scale was taken from the flight replay "
        "and not tuned on the benchmark, whereas the checkpoint was selected on these mosaics. The learned "
        "representation therefore improves on raw intensity by a wide margin but not on a classical structural "
        "representation: the gain of Table 6 is a gain over grayscale."
    )
    E['asym_par'] = (
        "The benchmark also exposes an asymmetry that is invisible in route tests. On the district mosaics with "
        f"grayscale NCC, searching Bing queries on the Google mosaic succeeds for only {g('sb_gray_bg_c_0')}% of clean "
        f"queries in global search, whereas Google queries on the Bing mosaic succeed for {g('sb_gray_gb_c_0')}%; both "
        f"structural representations nearly remove the asymmetry (cross-autoencoder {g('sb_cae_bg_c_0')}% versus "
        f"{g('sb_cae_gb_c_0')}%, gradient magnitude {g('sb_sob_bg_c_0')}% versus {g('sb_sob_gb_c_0')}%). The Bing "
        "mosaic, whose rendering is darker and less contrasted (Fig. 5), makes a poor NCC template, and either "
        "structural transform normalizes both sides to the same appearance."
    )
    E['t7_sentence'] = ("The gradient-magnitude baseline added after review (last row of Table 7) exceeds every one of "
                        "these checkpoints on the district mosaics (Section 4.3)." if every_ckpt else
                        "The gradient-magnitude baseline added after review (last row of Table 7) is compared with "
                        "them in Section 4.3.")
    E['t7_caption'] = (
        "Table 7. Best checkpoint per training lineage in the exhaustive sweep (Ürgüp core, 2,000 queries per "
        "direction, global search, mean of both directions) and performance of the same checkpoint on the wider "
        "district mosaics (4,000 queries per direction, global search). Grayscale NCC and the gradient-magnitude "
        "baseline (run on the district in global search only; Section 4.3) are shown for reference."
    )
    E['t7_row'] = ['Gradient magnitude (σ = 1.5 px), no training', '-', '-', '-', '-', f"{g('sb_sob_c')}%", f"{g('sb_sob_h')}%"]
    E['fig7_add'] = " Gradient magnitude was run on the district in global search only (Table 7 and Section 4.3)."

    E['disc_interp'] = (
        "Interpretation. Three results have to be read together. First, converting both images to a structural "
        "representation before correlation is what matters. On the district benchmark grayscale NCC localizes "
        f"{g('sb_gray_c')}% of clean queries in global search, the learned representation {g('sb_cae_c')}% and "
        f"gradient magnitude {g('sb_sob_c')}%; both structural representations remove the asymmetry between the "
        "providers, and the peak-separation probe shows the mechanism for the learned one: it suppresses false "
        "correlation peaks. Second, the learned representation does not improve on the classical one. On the "
        f"benchmark it falls {sb_short_c:.1f} points short of gradient magnitude on clean and {sb_short_h:.1f} points "
        "on degraded queries, on both halves of the spatial split and at every texture level; in the local "
        "low-texture probe of Section 4.4 it is also behind; and on the flights the picture is mixed. In independent "
        f"search over 3 × 3 km the three checkpoints localize {or_lo:.1f}-{or_hi:.1f}% of the eligible doctoral frames "
        f"within 30 m against {g('or_sobel')}% for gradient magnitude and {g('or_gray')}% for grayscale NCC, and in the "
        f"causal replay {ca_lo:.1f}-{ca_hi:.1f}% against {g('ca_sobel')}% and {g('ca_gray')}%. Relative to gradient "
        f"magnitude, the flown Model 4 lineage {vs('or_m4', 'or_sobel')} in the oracle replay and "
        f"{vs('ca_m4', 'ca_sobel')} in the causal replay, the benchmark-selected checkpoint 1850 "
        f"{vs('or_f48', 'or_sobel')} and {vs('ca_f48', 'ca_sobel')}, and on the additional flights gradient "
        f"magnitude {sob_ho}. The spread "
        f"between the three checkpoints ({ca_hi - ca_lo:.1f} points in the causal replay) is "
        + ("larger than any" if within else "of the same order as") + " of these differences." + ci_disc + " Third, the gain of the "
        "learned representation over grayscale is genuine - it persists when the checkpoint is selected on one half "
        "of the district and evaluated on the other" + (
            " and it does not arise when the same network is trained to reconstruct its input" if recon_ok else "") +
        " - but it is a gain over raw intensity. The networks that win the benchmark are shallow stride-1 networks "
        "with small kernels (Section 4.4), whose receptive field spans a few tens of pixels: functionally a learned "
        "local structural filter, and the fixed filter does the same job at least as well on this task. Whether a "
        "network trained for the matching objective itself, or applied to gradient images, could exceed the classical "
        "filter was not tested."
    )
    E['disc_stride'] = (
        "Benchmark ranking versus flight ranking. The checkpoint ranking of the satellite benchmark does not carry over "
        "to the flight frames. Checkpoint 1850 of the f48-k4-s1 run, best on the district mosaics, localizes "
        f"{g('or_f48')}% of the doctoral frames in the oracle replay and {g('ca_f48')}% in the causal replay; checkpoint "
        f"1550 of the same run, best on the core mosaics, localizes {g('or_c1550')}% and {g('ca_c1550')}%; and the "
        "stride-2 Model 4 lineage, whose archived checkpoints of epochs 450 and 500 are at or below grayscale on the "
        "benchmark (Sections 4.3 and 4.4), localizes "
        f"{g('or_m4')}% and {g('ca_m4')}%. UAV frames differ from both satellite providers in sensor, season, time of "
        "day and resampling, and the checkpoint sweep of Section 4.4 already showed swings of ten points or more between "
        "neighboring checkpoints. Selecting checkpoints on queries that resemble the operational imagery - for example "
        "on a separate set of flights recorded with the released manifest format - is the obvious remedy, and averaging "
        "neighboring checkpoints could reduce the variance."
    )
    E['disc_lit'] = (
        "Comparison with the literature. The results agree with the season-invariant transform of Fragoso et "
        "al.^{17} and the dense season-invariant features of Kinnari et al.^{19} in that pre-processing both images "
        "into a stable structural domain rescues classical matching; they add the qualification that, on a "
        "cross-source benchmark, an untrained gradient-magnitude filter achieved this at least as well as the learned "
        "transform, a control worth including in future comparisons. The closest method, the OpenStreetMap-supervised translation of Schleiss,^{20} "
        "was evaluated on a 560 m × 680 m area at about 300 m above ground; about 20% of the frames could not be matched "
        "and the median error of the remainder was about 40 m (22.7 m on a landmark-rich subset). In the causal replay of "
        f"the present work {g('ca_m4_unres')}% of the eligible doctoral frames remain unresolved with the Model 4 "
        f"lineage and the median error of accepted fixes is {g('ca_m4_med')} m ({g('med_after_m4')} m after removing the "
        "camera offset), over routes of 6-9 km at about 400-1,000 m above ground; areas, cameras and references differ, "
        "so the "
        "comparison is indicative only. The present work adds a transform applied to both the map and the frame, a "
        "network two orders of magnitude smaller than a pix2pix generator, controlled comparisons with grayscale, "
        "gradient-magnitude and input-reconstruction representations on the same queries and frames, a quantitative "
        "account of false-peak suppression "
        "and a reusable benchmark with spatial block bootstrap. Compared with retrieval-style cross-view "
        "geolocalization,^{14-16} the method returns a metric position at map resolution rather than a tile identity, "
        "at the cost of requiring near-nadir imagery and known heading and scale."
    )
    mags = [math.hypot(f(f'b_{c}_f_m4'), f(f'b_{c}_r_m4')) for c in ('l1d', 'fc')]
    off_rng = f'{min(mags):.0f}-{max(mags):.0f} m' if all(ok(m) for m in mags) else 'about 9-22 m'
    E['disc_valid'] = (
        "Validity boundaries. Several limits must be stated plainly. (i) Geographic scope. Ürgüp and Nevşehir are among "
        "the 125 cities of the training corpus and all evaluations are in the Ürgüp region; the spatial split of the "
        "district benchmark (Section 4.4) separates checkpoint selection from evaluation but not training from "
        "evaluation, so the results are in-domain, and a geographically held-out test with an explicit tile-exclusion "
        "buffer is required before claiming transfer to unseen regions. (ii) Reference accuracy. The consumer GPS "
        "reference has an observed error of 10-15 m and the cameras add a constant offset of " + off_rng + " (Section 4.2); "
        "median errors of about 10-20 m are therefore close to the measurement floor. (iii) Replay, not flight. The "
        "causal replay removes GNSS after the first frame but still processes recorded frames offline; it uses the "
        "recorded barometric altitude and compass heading, initializes from GNSS, propagates the position with a "
        "frame-to-frame visual-odometry prior instead of inertial sensing, and was not run on flight hardware, so "
        "latency and power were not measured. (iv) Geometry. The method assumes near-nadir imagery and a usable "
        "heading; heading errors of 5-10° are the dominant sensitivity (Table 5), and frames whose footprint at map "
        "resolution is smaller than a crop (low flights, steep terrain) are not eligible. (v) Model files and "
        "preprocessing. The flown Model 4 file of the doctoral study (epoch 250) was not archived; the earliest archived "
        "checkpoint of the same run (epoch 300) is used and identified by its SHA-256 hash; this file was not itself "
        "evaluated on the benchmark, whose entries for the lineage are the archived checkpoints of epochs 450 and 500. "
        "Flight results depend on the checkpoint and on input equalization by several points (Section 4.2). (vi) "
        "Baselines. Gradient magnitude was run on the district benchmark in global search only, with the smoothing "
        "scale of the flight replay and without tuning, and it was not included in the region-of-interest modes of "
        "Table 6, in the core-mosaic sweep or in the peak-separation probe; a tuned classical pipeline could be "
        "stronger still. Learned matchers (e.g., SuperPoint/SuperGlue or LoFTR) and other learned transforms were not "
        "re-implemented; the released frame manifest, replay program and benchmark allow such comparisons to be added."
    )
    E['disc_future'] = (
        "Future work. Five directions follow directly from the results. First, any learned representation for "
        "correlation-based localization should be measured against gradient magnitude rather than against raw "
        "intensity alone; training the network for the matching objective itself (peak separation rather than "
        "pixel-wise reconstruction), applying it to gradient images, or combining the two representations are "
        "untested. Second, select and average checkpoints on queries that resemble the operational imagery rather "
        "than on satellite pairs alone. Third, estimate the camera boresight offset in a short calibration flight and "
        "the heading from the imagery itself, the two error sources identified in Section 4.2. Fourth, fuse the visual "
        "fix with inertial and barometric states in a Kalman or particle filter, using the NCC peak margin and "
        "cross-crop agreement as an adaptive measurement covariance, and resolve scale over steep terrain with a "
        "multi-resolution reference or image-based height estimation.^{51} Fifth, extend the benchmark and the flight "
        "tests to geographically disjoint regions, seasons and providers."
    )
    sob_ho_abs = {'matches or exceeds every checkpoint': 'matched or exceeded all checkpoints',
                  'is on a par with the checkpoints': 'was on a par with the checkpoints',
                  'stays below the best checkpoint': 'stayed below the best checkpoint'}[sob_ho]
    E['conclusions'] = (
        "We asked whether a learned structural representation - a compact cross-autoencoder that renders satellite "
        "and nadir UAV imagery as map-like images - improves template-matching localization beyond a classical "
        "structural representation when both are matched by normalized cross-correlation with the same code. All "
        "results were regenerated from per-query and per-frame records with released programs. On a cross-source "
        "benchmark of about 500 km² with spatially blocked queries, both structural representations raise global "
        f"success at 25 m far above grayscale NCC ({g('sb_gray_c_0')}% on clean queries), but the untrained gradient "
        f"magnitude ({g('sb_sob_c_0')}% clean, {g('sb_sob_h_0')}% degraded) exceeds the benchmark-selected learned "
        f"checkpoint ({g('sb_cae_c_0')}%, {g('sb_cae_h_0')}%) by {sb_short_c:.1f} and {sb_short_h:.1f} points with "
        "block-bootstrap intervals that exclude zero, on both halves of a spatial split and at every texture level. "
        "The learned gain over grayscale is real - it persists under the spatial split" + (
            " and requires the cartographic target" if recon_ok else "") + " - but it is a gain over raw intensity. "
        "On sixteen DJI flights over Ürgüp the differences are small and checkpoint-dependent: in independent search "
        f"three checkpoints fixed in advance localized {or_lo:.1f}-{or_hi:.1f}% of the eligible doctoral frames within "
        f"30 m (gradient magnitude {g('or_sobel')}%, grayscale {g('or_gray')}%), and a causal replay that reads GNSS "
        f"only at the first frame localized {ca_lo:.1f}-{ca_hi:.1f}% with {cap_lo:.0f}-{cap_hi:.0f}% precision "
        f"({g('ca_sobel')}% and {g('ca_gray')}%); in the causal replay the flown lineage "
        f"{vs('ca_m4', 'ca_sobel', 'gradient magnitude')} (95% block-bootstrap interval {ci('ca', 'm4', 'sobel')} "
        f"points), the benchmark-selected checkpoint {vs('ca_f48', 'ca_sobel', 'it')} ({ci('ca', 'f48', 'sobel')}), "
        f"and on the additional flights gradient magnitude {sob_ho_abs}. The "
        "remaining errors are dominated by a constant camera offset, heading "
        "errors and a wrong ground height after a loss of track. Within its stated in-domain scope, structural "
        "template matching with a fixed gradient filter is an explainable and inexpensive candidate position-fix "
        "source for GNSS-denied UAV navigation, and a learned representation has to beat this baseline before its "
        "training cost is justified; the released benchmark and replay program make that test routine."
    )
    E['abstract'] = (
        "Unmanned aerial vehicles (UAVs) rely on global navigation satellite systems (GNSS), which can be jammed or "
        "spoofed. We ask whether a learned structural representation - a lightweight cross-autoencoder (50-78 thousand "
        "parameters, trained on 406,579 satellite-map tile pairs from 125 cities) that renders satellite and UAV images "
        "as map-like images - improves template matching against a satellite map by normalized cross-correlation "
        "beyond a classical structural representation, the gradient magnitude of the smoothed image. On a "
        "reproducible cross-source benchmark of Google and Bing mosaics with spatially blocked queries (about 500 "
        f"km²), global success at 25 m rose from {g('sb_gray_c_0')}% (grayscale) to "
        f"{g('sb_cae_c_0')}% with the benchmark-selected checkpoint and to {g('sb_sob_c_0')}% with untrained gradient "
        "magnitude, which led on both halves of a spatial split and under photometric degradation "
        f"({g('sb_sob_h_0')}% against {g('sb_cae_h_0')}%). Sixteen DJI flights over Ürgüp, Türkiye (2,951 frames), "
        "replayed with released code, showed small, checkpoint-dependent differences: three checkpoints fixed in "
        f"advance localized {or_lo:.1f}-{or_hi:.1f}% of eligible frames within 30 m in independent search (gradient "
        f"magnitude {g('or_sobel')}%, grayscale {g('or_gray')}%) and {ca_lo:.1f}-{ca_hi:.1f}% in a causal replay "
        f"reading GNSS only at the first frame ({g('ca_sobel')}% and {g('ca_gray')}%). Learned representations for "
        "this task must therefore be measured against gradient magnitude. All results are in-domain and offline."
    )
    n_words = len(E['abstract'].split())
    if n_words > 200: print(f'WARNING: abstract has {n_words} words (limit 200)')
    return E
